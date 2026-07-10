import argparse
import gc
import json
import time

import torch

from memintelli.NN_layers.linear import LinearMem
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def make_engine(stage, device, read_variation_seed):
    intra = stage == "intra"
    return DPETensorMultiMode(
        HGS=1e-5,
        LGS=1e-8,
        g_level=2,
        write_variation=0.0,
        read_variation=0.05,
        vnoise=0.0,
        rdac=2,
        radc=256,
        vread=0.2,
        rate_stuck_HGS=0.0,
        rate_stuck_LGS=0.0,
        mode=0,
        device=device,
        inference_chunk_size=8 * 1024 * 1024,
        fast_inference=intra,
        fast_inference_backend="triton_gidx" if intra else "torch",
        triton_input_precision="ieee",
        triton_block_r=64,
        triton_block_l=32,
        triton_block_k=64,
        triton_output_chunk_limit=512,
        triton_gidx_fused_restore_read_noise=True,
        triton_fuse_restored_input_slices=intra,
        triton_fuse_activation_slices=True,
        triton_direct_final_output=intra,
        triton_direct_final_exact_reduce=intra,
        triton_gidx_direct_final_output=False,
        triton_overlap_restore_direct=False,
        triton_precompute_input_voltage=False,
        triton_fast_adc_scale=False,
        triton_direct_output_zero_once=False,
        read_variation_seed=read_variation_seed,
        conductance_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        linear_output_dtype="input",
        mode0_semantic_policy="strict",
        mode0_vmm_compute_dtype=torch.bfloat16,
        runtime_counters=True,
    )


def run_stage(stage, weight_cpu, input_tensor, args):
    device = input_tensor.device
    engine = make_engine(stage, device, args.read_variation_seed)
    layer = LinearMem(
        engine=engine,
        in_features=args.in_features,
        out_features=args.out_features,
        input_slice=[1, 1, 1, 1, 1],
        weight_slice=[1, 1, 1, 1, 1],
        bias=False,
        device=device,
        dtype=torch.bfloat16,
        bw_e=None,
        input_paral_size=(1, 64),
        weight_paral_size=(64, 64),
        input_quant_gran=(1, 64),
        weight_quant_gran=(64, 64),
        skip_initial_mapping=True,
    )
    with torch.no_grad():
        layer.weight.copy_(weight_cpu.to(device))
    layer._prepare_inference_weight(streaming=False, free_weights=False)
    with torch.inference_mode():
        for _ in range(args.warmup):
            layer(input_tensor)
    torch.cuda.synchronize(device)
    engine.reset_fastpath_counters()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = layer(input_tensor)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    counters = engine.get_fastpath_counters()
    output_cpu = output.cpu()
    del output, layer, engine
    gc.collect()
    torch.cuda.empty_cache()
    return output_cpu, {
        "stage": stage,
        "forward_ms": elapsed_ms,
        "cuda_peak_mb": peak_mb,
        "counters": counters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--read-variation-seed", type=int, default=1234)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(23)
    device = torch.device("cuda")
    weight_cpu = torch.randn(
        args.out_features,
        args.in_features,
        dtype=torch.bfloat16,
    )
    input_tensor = torch.randn(
        args.tokens,
        args.in_features,
        device=device,
        dtype=torch.bfloat16,
    )

    off_output, off_result = run_stage("off", weight_cpu, input_tensor, args)
    intra_output, intra_result = run_stage("intra", weight_cpu, input_tensor, args)
    equal = torch.equal(off_output, intra_output)
    max_abs = (off_output.float() - intra_output.float()).abs().max().item()
    result = {
        "in_features": args.in_features,
        "out_features": args.out_features,
        "tokens": args.tokens,
        "read_variation": 0.05,
        "read_variation_seed": args.read_variation_seed,
        "torch_equal": equal,
        "max_abs": max_abs,
        "runs": [off_result, intra_result],
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))
    if not equal:
        raise AssertionError(f"S2 intra mismatch: max_abs={max_abs}")
    intra_counters = intra_result["counters"]
    if intra_counters.get("strict_adc_scale_accumulate_triton_success_count", 0) <= 0:
        raise AssertionError("strict postprocess Triton kernel was not used")
    if intra_counters.get("strict_adc_scale_accumulate_triton_fallback_count", 0) != 0:
        raise AssertionError("strict postprocess Triton kernel fell back")


if __name__ == "__main__":
    main()
