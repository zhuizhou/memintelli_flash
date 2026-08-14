from memintelli.NN_layers.state_planner import (
    estimate_mode0_workspace,
    plan_global_feasible_resident_budget,
    plan_global_resident_budget,
    plan_output_block,
)


def test_workspace_estimate_grows_with_output_width():
    small = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=16384,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
    )
    large = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=32768,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
    )

    assert large.peak_bytes > small.peak_bytes
    assert large.preparation_peak_bytes > small.preparation_peak_bytes
    assert large.execution_peak_bytes > small.execution_peak_bytes


def test_read_noise_accounts_for_restore_and_noise_temporaries_without_requiring_a_seed():
    clean = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=32768,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        read_variation=0.0,
        seeded_read_noise=False,
    )
    noisy = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=32768,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        seeded_read_noise=False,
    )

    assert noisy.execution_peak_bytes > clean.execution_peak_bytes


def test_grouped_noisy_vmm_accounts_for_all_adc_slice_partials():
    serial = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=16384,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        seeded_read_noise=True,
        grouped_noisy_vmm=False,
    )
    grouped = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=16384,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        seeded_read_noise=True,
        grouped_noisy_vmm=True,
    )

    assert grouped.execution_peak_bytes > serial.execution_peak_bytes


def test_direct_final_workspace_caps_restored_state_to_one_output_chunk():
    narrow = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=65536,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        streaming_prefetch_window=1,
        restored_conductance_bytes=2,
        output_bytes=4,
    )
    wide = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        streaming_prefetch_window=1,
        restored_conductance_bytes=2,
        output_bytes=4,
    )

    assert wide.execution_peak_bytes > narrow.execution_peak_bytes
    assert wide.execution_peak_bytes < narrow.execution_peak_bytes * 4


def test_wide_layer_preparation_accounts_for_conductance_index_mapping():
    in_features = 2560
    out_features = 248320
    weight_slices = 5

    estimate = estimate_mode0_workspace(
        tokens=128,
        in_features=in_features,
        out_features=out_features,
        input_slices=5,
        weight_slices=weight_slices,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        compressed_state_bytes=1,
        restored_conductance_bytes=2,
    )

    weight_elements = in_features * out_features
    expected_mapping_peak = weight_elements * weight_slices * (1 + 4 + 1)
    assert estimate.preparation_peak_bytes >= expected_mapping_peak


def test_24g_planner_blocks_wide_lm_head_when_resident_state_is_8gib():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=24576,
        base_allocated_mb=1214,
        resident_state_mb=8192,
        safety_margin_mb=256,
        read_variation=0.05,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        minimum_output_block_cols=16384,
        compressed_state_bytes=1,
        restored_conductance_bytes=2,
        output_bytes=4,
    )

    predicted_total = (
        plan.base_allocated_bytes
        + plan.resident_state_bytes
        + plan.safety_margin_bytes
        + plan.estimated_peak_bytes
    )
    assert plan.shard_count > 1
    assert predicted_total <= 24576 * 1024 * 1024


def test_direct_final_planner_can_cap_a_wide_layer_to_one_execution_window():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=8192,
        base_allocated_mb=1214,
        resident_state_mb=0,
        safety_margin_mb=256,
        read_variation=0.05,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        minimum_output_block_cols=16384,
        maximum_output_block_cols=16384,
        compressed_state_bytes=1,
        restored_conductance_bytes=2,
        output_bytes=4,
    )

    assert plan.output_block_cols == 16384
    assert plan.shard_count == 16


def test_direct_final_planner_does_not_charge_full_strict_partial_tensor():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=24576,
        base_allocated_mb=1224,
        resident_state_mb=0,
        safety_margin_mb=256,
        read_variation=0.05,
        seeded_read_noise=True,
        grouped_noisy_vmm=True,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        streaming_prefetch_window=0,
        restored_conductance_bytes=2,
        output_bytes=4,
    )

    assert plan.output_block_cols == 248320
    assert plan.estimated_peak_mb < 20 * 1024
    assert plan.execution_peak_mb < 4 * 1024
    assert plan.preparation_peak_mb > plan.execution_peak_mb


def test_planner_reserves_one_internal_chunk_when_resident_request_uses_total_budget():
    plan = plan_output_block(
        tokens=128,
        in_features=12288,
        out_features=4096,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=24576,
        base_allocated_mb=1224,
        resident_state_mb=24576,
        safety_margin_mb=256,
        read_variation=0.05,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
        minimum_output_block_cols=16384,
        streaming_prefetch_window=1,
        restored_conductance_bytes=2,
        output_bytes=4,
    )

    assert plan.output_block_cols == 4096
    assert plan.resident_state_mb < 24576
    assert plan.workspace_budget_mb >= plan.estimated_peak_mb


def test_unlimited_budget_keeps_the_full_layer():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=0.0,
    )

    assert plan.output_block_cols == 248320
    assert plan.shard_count == 1


def test_budgeted_plan_selects_largest_aligned_block_that_fits():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=4096.0,
        base_allocated_mb=1224.0,
        safety_margin_mb=512.0,
    )

    assert plan.output_block_cols % 64 == 0
    assert plan.shard_count >= 8
    assert plan.estimated_peak_mb <= plan.workspace_budget_mb

    next_width = min(248320, plan.output_block_cols + 64)
    if next_width > plan.output_block_cols:
        next_estimate = estimate_mode0_workspace(
            tokens=128,
            in_features=2560,
            out_features=next_width,
            input_slices=5,
            weight_slices=5,
            array_rows=64,
            array_cols=64,
        )
        assert next_estimate.peak_bytes > plan.workspace_budget_bytes


def test_manual_block_override_is_only_aligned_and_clamped():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=4096.0,
        manual_output_block_cols=16001,
    )

    assert plan.output_block_cols == 16064
    assert plan.manual_override is True


def test_plan_reports_each_reserved_budget_component():
    plan = plan_output_block(
        tokens=128,
        in_features=2560,
        out_features=248320,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=4096.0,
        base_allocated_mb=1024.0,
        resident_state_mb=512.0,
        safety_margin_mb=256.0,
    )

    assert plan.base_allocated_mb == 1024.0
    assert plan.resident_state_mb == 512.0
    assert plan.safety_margin_mb == 256.0
    assert plan.workspace_budget_mb == 2304.0


def test_global_residency_keeps_the_zero_resident_block_counts():
    layer_specs = [
        {
            "tokens": 128,
            "in_features": 2560,
            "out_features": 9216,
            "input_slices": 5,
            "weight_slices": 5,
            "array_rows": 64,
            "array_cols": 64,
            "read_variation": 0.05,
            "seeded_read_noise": True,
            "grouped_noisy_vmm": True,
            "execution_strategy": "framework",
            "restored_conductance_bytes": 2,
            "output_bytes": 2,
        },
        {
            "tokens": 128,
            "in_features": 2560,
            "out_features": 248320,
            "input_slices": 5,
            "weight_slices": 5,
            "array_rows": 64,
            "array_cols": 64,
            "read_variation": 0.05,
            "seeded_read_noise": True,
            "grouped_noisy_vmm": True,
            "execution_strategy": "framework",
            "restored_conductance_bytes": 2,
            "output_bytes": 2,
        },
    ]
    common = {
        "cuda_peak_budget_mb": 24576.0,
        "base_allocated_mb": 1214.0,
        "safety_margin_mb": 256.0,
    }
    zero_resident = [
        plan_output_block(**spec, resident_state_mb=0.0, **common)
        for spec in layer_specs
    ]

    resident_mb = plan_global_resident_budget(
        layer_specs=layer_specs,
        requested_resident_mb=24576.0,
        **common,
    )
    selected = [
        plan_output_block(**spec, resident_state_mb=resident_mb, **common)
        for spec in layer_specs
    ]

    assert 0.0 < resident_mb < 24576.0
    assert [plan.shard_count for plan in selected] == [
        plan.shard_count for plan in zero_resident
    ]


def test_global_feasible_residency_honors_a_requested_frontier_point_with_more_blocks():
    layer_specs = [
        {
            "tokens": 128,
            "in_features": 2560,
            "out_features": 248320,
            "input_slices": 5,
            "weight_slices": 5,
            "array_rows": 64,
            "array_cols": 64,
            "read_variation": 0.05,
            "seeded_read_noise": True,
            "grouped_noisy_vmm": True,
            "execution_strategy": "direct_final",
            "output_chunk_tiles": 256,
            "minimum_output_block_cols": 16384,
            "restored_conductance_bytes": 2,
            "output_bytes": 4,
        }
    ]
    common = {
        "cuda_peak_budget_mb": 24576.0,
        "base_allocated_mb": 1214.0,
        "safety_margin_mb": 256.0,
    }

    preserve_mb = plan_global_resident_budget(
        layer_specs=layer_specs,
        requested_resident_mb=24576.0,
        **common,
    )
    feasible_mb = plan_global_feasible_resident_budget(
        layer_specs=layer_specs,
        requested_resident_mb=24576.0,
        **common,
    )
    plan = plan_output_block(
        **layer_specs[0],
        resident_state_mb=feasible_mb,
        **common,
    )

    assert feasible_mb > preserve_mb
    assert plan.resident_state_mb == feasible_mb
    assert plan.shard_count > 1
