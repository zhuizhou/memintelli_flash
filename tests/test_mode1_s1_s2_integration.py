import argparse
import importlib


benchmark = importlib.import_module("examples.20_real_llm_prefill_benchmark")


def _args(stage):
    return argparse.Namespace(
        mode=1,
        s2_stage=stage,
        fast_inference_backend="auto",
        triton_gidx_direct_final_output=True,
        triton_mode1_gidx_direct_final=True,
        triton_mode1_gdiff_direct_final=False,
        mode1_gdiff_schedule="auto",
        state_resident_budget_mb=-1.0,
        mode0_semantic_policy="auto",
        read_variation=0.05,
        fuse_mlp_gate_up=False,
        fuse_common_input_projections=False,
    )


def test_mode1_s2_intra_enables_noisy_gdiff_direct_final():
    args, _ = benchmark.apply_s2_stage(_args("intra"))

    assert args.fast_inference is True
    assert args.triton_mode1_gdiff_direct_final is True
    assert args.triton_mode1_gidx_direct_final is True
    assert args.fast_inference_backend == "triton_gidx"


def test_mode1_s2_off_disables_both_direct_final_paths():
    args, _ = benchmark.apply_s2_stage(_args("off"))

    assert args.fast_inference is False
    assert args.triton_mode1_gdiff_direct_final is False
    assert args.triton_mode1_gidx_direct_final is False
