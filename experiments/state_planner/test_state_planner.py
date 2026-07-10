from memintelli.NN_layers.state_planner import (
    estimate_mode0_workspace,
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


def test_seeded_read_noise_accounts_for_restore_and_noise_temporaries():
    unseeded = estimate_mode0_workspace(
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
    seeded = estimate_mode0_workspace(
        tokens=128,
        in_features=2560,
        out_features=32768,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        seeded_read_noise=True,
    )

    assert seeded.execution_peak_bytes > unseeded.execution_peak_bytes


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
