from dataclasses import dataclass
import math


MIB = 1024 * 1024


def _align_up(value, alignment):
    return max(alignment, math.ceil(int(value) / int(alignment)) * int(alignment))


@dataclass(frozen=True)
class WorkspaceEstimate:
    preparation_peak_bytes: int
    execution_peak_bytes: int
    peak_bytes: int

    @property
    def peak_mb(self):
        return self.peak_bytes / MIB


@dataclass(frozen=True)
class OutputBlockPlan:
    output_block_cols: int
    shard_count: int
    workspace_budget_bytes: int
    base_allocated_bytes: int
    resident_state_bytes: int
    safety_margin_bytes: int
    estimated_peak_bytes: int
    full_layer_peak_bytes: int
    manual_override: bool

    @property
    def workspace_budget_mb(self):
        return self.workspace_budget_bytes / MIB

    @property
    def estimated_peak_mb(self):
        return self.estimated_peak_bytes / MIB

    @property
    def base_allocated_mb(self):
        return self.base_allocated_bytes / MIB

    @property
    def resident_state_mb(self):
        return self.resident_state_bytes / MIB

    @property
    def safety_margin_mb(self):
        return self.safety_margin_bytes / MIB


def estimate_mode0_workspace(
    *,
    tokens,
    in_features,
    out_features,
    input_slices,
    weight_slices,
    array_rows,
    array_cols,
    weight_bytes=2,
    quantized_bytes=4,
    sliced_bytes=1,
    restored_conductance_bytes=None,
    partial_bytes=4,
    output_bytes=2,
    read_variation=0.0,
    seeded_read_noise=False,
    grouped_noisy_vmm=False,
):
    if restored_conductance_bytes is None:
        restored_conductance_bytes = (
            10
            if float(read_variation or 0.0) > 0.0 and bool(seeded_read_noise)
            else 4
        )
    padded_in = _align_up(in_features, array_rows)
    padded_out = _align_up(out_features, array_cols)
    weight_elements = padded_in * padded_out
    input_tiles = padded_in // int(array_rows)

    original_weight = weight_elements * int(weight_bytes)
    quantized_weight = weight_elements * int(quantized_bytes)
    sliced_weight = weight_elements * int(weight_slices) * int(sliced_bytes)
    preparation_peak = original_weight + quantized_weight + sliced_weight

    restored_conductance = (
        weight_elements * int(weight_slices) * int(restored_conductance_bytes)
    )
    strict_partial = (
        input_tiles
        * int(tokens)
        * padded_out
        * int(partial_bytes)
        * (min(3, int(weight_slices)) if grouped_noisy_vmm else 1)
    )
    activation_slices = (
        int(tokens) * padded_in * int(input_slices) * int(sliced_bytes)
    )
    output = int(tokens) * padded_out * int(output_bytes)
    execution_peak = (
        sliced_weight
        + restored_conductance
        + strict_partial
        + activation_slices
        + output
    )
    return WorkspaceEstimate(
        preparation_peak_bytes=preparation_peak,
        execution_peak_bytes=execution_peak,
        peak_bytes=max(preparation_peak, execution_peak),
    )


def plan_output_block(
    *,
    tokens,
    in_features,
    out_features,
    input_slices,
    weight_slices,
    array_rows,
    array_cols,
    cuda_peak_budget_mb=0.0,
    base_allocated_mb=0.0,
    resident_state_mb=0.0,
    safety_margin_mb=0.0,
    manual_output_block_cols=0,
    read_variation=0.0,
    seeded_read_noise=False,
    grouped_noisy_vmm=False,
):
    base_allocated_bytes = max(0, int(float(base_allocated_mb) * MIB))
    resident_state_bytes = max(0, int(float(resident_state_mb) * MIB))
    safety_margin_bytes = max(0, int(float(safety_margin_mb) * MIB))
    requested_budget_bytes = max(0, int(float(cuda_peak_budget_mb or 0.0) * MIB))
    net_workspace_budget = max(
        0,
        requested_budget_bytes
        - base_allocated_bytes
        - resident_state_bytes
        - safety_margin_bytes,
    )
    full_estimate = estimate_mode0_workspace(
        tokens=tokens,
        in_features=in_features,
        out_features=out_features,
        input_slices=input_slices,
        weight_slices=weight_slices,
        array_rows=array_rows,
        array_cols=array_cols,
        read_variation=read_variation,
        seeded_read_noise=seeded_read_noise,
        grouped_noisy_vmm=grouped_noisy_vmm,
    )
    if manual_output_block_cols:
        block_cols = min(
            int(out_features),
            _align_up(manual_output_block_cols, array_cols),
        )
        estimate = estimate_mode0_workspace(
            tokens=tokens,
            in_features=in_features,
            out_features=block_cols,
            input_slices=input_slices,
            weight_slices=weight_slices,
            array_rows=array_rows,
            array_cols=array_cols,
            read_variation=read_variation,
            seeded_read_noise=seeded_read_noise,
            grouped_noisy_vmm=grouped_noisy_vmm,
        )
        return OutputBlockPlan(
            output_block_cols=block_cols,
            shard_count=math.ceil(int(out_features) / block_cols),
            workspace_budget_bytes=(
                net_workspace_budget
                if requested_budget_bytes > 0
                else estimate.peak_bytes
            ),
            base_allocated_bytes=base_allocated_bytes,
            resident_state_bytes=resident_state_bytes,
            safety_margin_bytes=safety_margin_bytes,
            estimated_peak_bytes=estimate.peak_bytes,
            full_layer_peak_bytes=full_estimate.peak_bytes,
            manual_override=True,
        )

    if float(cuda_peak_budget_mb or 0.0) <= 0.0:
        return OutputBlockPlan(
            output_block_cols=int(out_features),
            shard_count=1,
            workspace_budget_bytes=full_estimate.peak_bytes,
            base_allocated_bytes=base_allocated_bytes,
            resident_state_bytes=resident_state_bytes,
            safety_margin_bytes=safety_margin_bytes,
            estimated_peak_bytes=full_estimate.peak_bytes,
            full_layer_peak_bytes=full_estimate.peak_bytes,
            manual_override=False,
        )

    workspace_budget = net_workspace_budget
    tile_count = math.ceil(int(out_features) / int(array_cols))
    low, high = 1, max(1, tile_count)
    best_tiles = 1
    while low <= high:
        mid = (low + high) // 2
        width = min(int(out_features), mid * int(array_cols))
        estimate = estimate_mode0_workspace(
            tokens=tokens,
            in_features=in_features,
            out_features=width,
            input_slices=input_slices,
            weight_slices=weight_slices,
            array_rows=array_rows,
            array_cols=array_cols,
            read_variation=read_variation,
            seeded_read_noise=seeded_read_noise,
            grouped_noisy_vmm=grouped_noisy_vmm,
        )
        if estimate.peak_bytes <= workspace_budget:
            best_tiles = mid
            low = mid + 1
        else:
            high = mid - 1

    block_cols = min(int(out_features), best_tiles * int(array_cols))
    estimate = estimate_mode0_workspace(
        tokens=tokens,
        in_features=in_features,
        out_features=block_cols,
        input_slices=input_slices,
        weight_slices=weight_slices,
        array_rows=array_rows,
        array_cols=array_cols,
        read_variation=read_variation,
        seeded_read_noise=seeded_read_noise,
        grouped_noisy_vmm=grouped_noisy_vmm,
    )
    return OutputBlockPlan(
        output_block_cols=block_cols,
        shard_count=math.ceil(int(out_features) / block_cols),
        workspace_budget_bytes=workspace_budget,
        base_allocated_bytes=base_allocated_bytes,
        resident_state_bytes=resident_state_bytes,
        safety_margin_bytes=safety_margin_bytes,
        estimated_peak_bytes=estimate.peak_bytes,
        full_layer_peak_bytes=full_estimate.peak_bytes,
        manual_override=False,
    )
