from memintelli.NN_layers.state_planner import (
    estimate_mode1_workspace,
    plan_output_block,
)
import argparse
import importlib


benchmark = importlib.import_module("examples.20_real_llm_prefill_benchmark")


MIB = 1024 * 1024


def test_mode1_workspace_counts_pair_indices_and_gdiff_window():
    estimate = estimate_mode1_workspace(
        tokens=128,
        in_features=4096,
        out_features=12288,
        array_rows=64,
        array_cols=64,
        read_variation=0.05,
        index_bytes=1,
        gdiff_bytes=2,
        execution_window_cols=4096,
        output_bytes=4,
    )

    pair_index_bytes = 2 * 4096 * 12288
    gdiff_window_bytes = 4096 * 4096 * 2
    assert estimate.execution_peak_bytes >= pair_index_bytes + gdiff_window_bytes


def test_mode1_unlimited_plan_uses_full_layer_window():
    plan = plan_output_block(
        mode=1,
        tokens=128,
        in_features=4096,
        out_features=12288,
        input_slices=1,
        weight_slices=1,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=0,
        read_variation=0.05,
        execution_strategy="direct_final",
    )

    assert plan.output_block_cols == 12288
    assert plan.execution_window_cols == 12288
    assert plan.execution_window_bytes == 4096 * 12288 * 2


def test_mode1_budgeted_plan_reduces_execution_window():
    plan = plan_output_block(
        mode=1,
        tokens=128,
        in_features=4096,
        out_features=248320,
        input_slices=1,
        weight_slices=1,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=2500,
        base_allocated_mb=500,
        safety_margin_mb=128,
        read_variation=0.05,
        execution_strategy="direct_final",
    )

    assert 64 <= plan.execution_window_cols < 248320
    assert plan.execution_window_cols % 64 == 0
    assert plan.estimated_peak_bytes <= 2500 * MIB


def test_full_model_planner_passes_mode1_state_model():
    args = argparse.Namespace(
        mode=1,
        s1_stage="budgeted",
        s1_block_addressable=True,
        state_planner="analytical",
        output_block_cols=0,
        cuda_peak_budget_mb=2500.0,
        state_resident_budget_mb=0.0,
        weight_paral_size=(64, 64),
        input_slice=(1, 1, 1, 1, 1),
        weight_slice=(1, 1, 1, 1, 1),
        read_variation=0.05,
        mode0_vmm_compute_dtype="bfloat16",
        compute_dtype="bfloat16",
        s2_stage="full",
        triton_output_chunk_limit=256,
        triton_terminal_output_chunk_limit=0,
        streaming=False,
        g_level=16,
        batch=1,
        seq=128,
        planner_base_allocated_mb=500.0,
        planner_safety_margin_mb=128.0,
    )
    layer = argparse.Namespace(in_features=4096, out_features=248320)

    plan = benchmark.plan_linear_output_blocks(args, layer, terminal_layer=True)

    assert plan.execution_window_bytes > 0
    assert plan.execution_window_cols < layer.out_features


def test_unlimited_full_model_plan_keeps_full_mode1_execution_window():
    args = argparse.Namespace(
        mode=1,
        s1_stage="budgeted",
        s1_block_addressable=True,
        state_planner="analytical",
        output_block_cols=0,
        cuda_peak_budget_mb=0.0,
        state_resident_budget_mb=-1.0,
        weight_paral_size=(64, 64),
        input_slice=(1, 1, 1, 1, 1),
        weight_slice=(1, 1, 1, 1, 1),
        read_variation=0.05,
        mode0_vmm_compute_dtype="bfloat16",
        compute_dtype="bfloat16",
        s2_stage="full",
        triton_output_chunk_limit=256,
        triton_terminal_output_chunk_limit=0,
        streaming=False,
        g_level=16,
        batch=1,
        seq=128,
        planner_base_allocated_mb=0.0,
        planner_safety_margin_mb=256.0,
    )
    layer = argparse.Namespace(in_features=4096, out_features=248320)

    plan = benchmark.plan_linear_output_blocks(args, layer, terminal_layer=True)

    assert plan.output_block_cols == layer.out_features
    assert plan.execution_window_cols == layer.out_features
