import argparse
import importlib.util
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "examples" / "20_real_llm_prefill_benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("real_llm_prefill_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def s2_args(stage):
    return argparse.Namespace(
        s2_stage=stage,
        fuse_mlp_gate_up=True,
        fuse_common_input_projections=True,
        triton_fuse_restored_input_slices=True,
        triton_direct_final_output=True,
        triton_gidx_direct_final_output=True,
        triton_direct_final_exact_reduce=False,
        triton_overlap_restore_direct=True,
        triton_precompute_input_voltage=True,
        triton_fast_adc_scale=True,
        triton_direct_output_zero_once=True,
        triton_activation_slice_cache=True,
    )


def test_s2_off_disables_both_execution_compaction_levels():
    module = load_benchmark_module()
    args, fast_inference = module.apply_s2_stage(s2_args("off"))

    assert fast_inference is False
    assert args.triton_fuse_restored_input_slices is False
    assert args.triton_direct_final_output is False
    assert args.triton_direct_final_exact_reduce is False
    assert args.fuse_mlp_gate_up is False
    assert args.fuse_common_input_projections is False
    assert args.triton_activation_slice_cache is False


def test_s2_intra_enables_strict_intra_linear_compaction_only():
    module = load_benchmark_module()
    args, fast_inference = module.apply_s2_stage(s2_args("intra"))

    assert fast_inference is True
    assert args.triton_fuse_restored_input_slices is True
    assert args.triton_direct_final_output is True
    assert args.triton_gidx_direct_final_output is False
    assert args.triton_direct_final_exact_reduce is True
    assert args.fuse_mlp_gate_up is False
    assert args.fuse_common_input_projections is False
    assert args.triton_activation_slice_cache is False


def test_s2_full_adds_common_input_projection_coalescing():
    module = load_benchmark_module()
    args, fast_inference = module.apply_s2_stage(s2_args("full"))

    assert fast_inference is True
    assert args.triton_direct_final_output is True
    assert args.triton_direct_final_exact_reduce is True
    assert args.fuse_mlp_gate_up is True
    assert args.fuse_common_input_projections is True
    assert args.triton_activation_slice_cache is False


def test_s1_stages_form_resident_block_budgeted_hierarchy():
    module = load_benchmark_module()

    off = argparse.Namespace(s1_stage="off", state_resident_budget_mb=0.0)
    block = argparse.Namespace(s1_stage="block", state_resident_budget_mb=0.0)
    budgeted = argparse.Namespace(s1_stage="budgeted", state_resident_budget_mb=8192.0)

    module.apply_s1_stage(off)
    module.apply_s1_stage(block)
    module.apply_s1_stage(budgeted)

    assert off.s1_block_addressable is False
    assert off.state_planner == "off"
    assert off.state_resident_budget_mb == -1.0
    assert block.s1_block_addressable is True
    assert block.state_planner == "off"
    assert block.state_resident_budget_mb == 0.0
    assert budgeted.s1_block_addressable is True
    assert budgeted.state_planner == "analytical"
    assert budgeted.state_resident_budget_mb == 8192.0


def test_s1_budget_is_the_single_source_for_cross_layer_residency():
    module = load_benchmark_module()
    args = argparse.Namespace(
        s1_stage="budgeted",
        state_resident_budget_mb=8192.0,
        memory_resident_budget_mb=0.0,
    )

    module.apply_s1_stage(args)

    assert args.memory_resident_budget_mb == 8192.0


def mode_args(mode):
    return argparse.Namespace(
        execution_mode=mode,
        s1_stage="budgeted",
        memory_prepare_policy="lazy_release",
        memory_budget_mb=0.0,
        memory_resident_budget_mb=8192.0 if mode == "balanced" else 0.0,
        state_resident_budget_mb=8192.0 if mode == "balanced" else 0.0,
        cuda_peak_budget_mb=20000.0 if mode != "speed" else 0.0,
        fast_inference_backend="triton_gidx",
        triton_fuse_restored_input_slices=True,
        fuse_mlp_gate_up=False,
        fuse_common_input_projections=False,
        triton_overlap_restore_direct=False,
        triton_precompute_input_voltage=False,
        triton_fast_adc_scale=False,
        triton_direct_output_zero_once=False,
        triton_gidx_direct_final_output=False,
        triton_mode0_strict_intermediate=False,
        triton_mode0_strict_intermediate_backend="auto",
        mode=0,
        streaming=False,
        lazy_prepare=False,
        lazy_release_after_forward=False,
        free_weights=False,
        memory_runtime_diagnostic=False,
        inference_chunk_size=8 * 1024 * 1024,
        _inference_chunk_size_user_set=False,
        _triton_fuse_restored_input_slices_user_set=False,
        _fuse_mlp_gate_up_user_set=False,
        _fuse_common_input_projections_user_set=False,
        _triton_overlap_restore_direct_user_set=False,
        _triton_precompute_input_voltage_user_set=False,
        _triton_fast_adc_scale_user_set=False,
        _triton_direct_output_zero_once_user_set=False,
        _triton_gidx_direct_final_output_user_set=False,
    )


def test_execution_modes_do_not_change_s2_compute_policy():
    module = load_benchmark_module()
    compute_fields = (
        "fast_inference_backend",
        "triton_fuse_restored_input_slices",
        "fuse_mlp_gate_up",
        "fuse_common_input_projections",
        "triton_overlap_restore_direct",
        "triton_precompute_input_voltage",
        "triton_fast_adc_scale",
        "triton_direct_output_zero_once",
        "triton_gidx_direct_final_output",
    )

    for mode in ("speed", "balanced", "memory"):
        args = mode_args(mode)
        before = {field: getattr(args, field) for field in compute_fields}
        module.apply_launcher_execution_mode_defaults(args)
        after = {field: getattr(args, field) for field in compute_fields}
        assert after == before

    speed = mode_args("speed")
    memory = mode_args("memory")
    module.apply_launcher_execution_mode_defaults(speed)
    module.apply_launcher_execution_mode_defaults(memory)
    assert speed.state_resident_budget_mb == -1.0
    assert memory.state_resident_budget_mb == 0.0


def test_launcher_parser_exposes_independent_s1_s2_controls():
    module = load_benchmark_module()
    argv = [
        "benchmark",
        "--model-path",
        "dummy-model",
        "--s1-stage",
        "block",
        "--s2-stage",
        "intra",
        "--cuda-peak-budget-mb",
        "20000",
        "--state-resident-budget-mb",
        "8192",
        "--output-block-cols",
        "16384",
        "--no-include-hf",
        "--no-include-original",
        "--no-include-v2",
        "--include-v3-mode0",
    ]
    with mock.patch("sys.argv", argv):
        args = module.parse_args()

    assert args.s1_stage == "block"
    assert args.s2_stage == "intra"
    assert args.cuda_peak_budget_mb == 20000.0
    assert args.state_resident_budget_mb == 0.0
    assert args.output_block_cols == 16384
    assert args.inference_chunk_size == 16 * 1024 * 1024
    assert args.fuse_mlp_gate_up is False
    assert args.fuse_common_input_projections is False
