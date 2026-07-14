from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


NAMED_SHAPES = {
    "qkv": (128, 4096, 4096),
    "gate-up": (128, 4096, 11008),
    "down": (128, 11008, 4096),
    "lm-head": (128, 4096, 151936),
}


def build_mode1_engine(
    device: torch.device,
    *,
    backend: str,
    read_var: float,
    dtype: torch.dtype,
    require_fastpath: bool,
    input_tile_group: int = 1,
    gdiff_direct: bool = False,
    gdiff_schedule: str = "auto",
) -> DPETensorMultiMode:
    return DPETensorMultiMode(
        HGS=1e-5,
        LGS=1e-7,
        g_level=16,
        write_variation=0.0,
        read_variation=float(read_var),
        vnoise=0.0,
        rate_stuck_HGS=0.0,
        rate_stuck_LGS=0.0,
        rdac=16,
        radc=256,
        mode=1,
        mode1_paral_size=(64, 64),
        mode1_adc_per_tile=True,
        fast_inference=True,
        fast_inference_backend=backend,
        triton_input_precision="ieee",
        triton_auto_config=True,
        triton_mode1_gidx_direct_final=True,
        triton_mode1_gdiff_direct_final=bool(gdiff_direct),
        mode1_gdiff_schedule=str(gdiff_schedule),
        triton_mode1_chunked_direct_final=True,
        triton_mode1_input_tile_group=int(input_tile_group),
        mode1_grouped_tile_gemm=False,
        mode1_require_fastpath=require_fastpath,
        conductance_dtype=dtype,
        compute_dtype=dtype,
        device=device,
    )


def prepare_mode1_tensor(
    engine: DPETensorMultiMode,
    data: torch.Tensor,
    *,
    is_weight: bool,
) -> SlicedDataMultiMode:
    sliced = SlicedDataMultiMode(
        torch.tensor([1], device=data.device),
        is_weight=is_weight,
        paral_size=(64, 64),
        quant_gran=(64, 64),
        device=data.device,
        inference=True,
        mode=1,
    )
    sliced.slice_data_imp(engine, data)
    return sliced


def compare_mode1_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | bool]:
    actual_f = actual.detach().to(torch.float32)
    expected_f = expected.detach().to(torch.float32)
    diff = (actual_f - expected_f).abs()
    rel = diff / expected_f.abs().clamp_min(1e-12)
    cosine = F.cosine_similarity(
        actual_f.reshape(1, -1),
        expected_f.reshape(1, -1),
        dim=1,
    ).item()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "cosine_similarity": float(cosine),
        "torch_equal": bool(torch.equal(actual_f, expected_f)),
        "allclose_rtol1e_2_atol1e_2": bool(
            torch.allclose(actual_f, expected_f, rtol=1e-2, atol=1e-2)
        ),
        "finite": bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all()),
    }


@torch.no_grad()
def run_mode1_comparison(
    shape: tuple[int, int, int],
    *,
    read_var: float,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 7,
    input_tile_group: int = 1,
) -> dict:
    tokens, in_features, out_features = (int(v) for v in shape)
    torch.manual_seed(seed)
    x = torch.randn((tokens, in_features), device=device, dtype=dtype)
    weight = torch.randn((in_features, out_features), device=device, dtype=dtype)

    optimized = build_mode1_engine(
        device,
        backend="triton_gidx",
        read_var=read_var,
        dtype=dtype,
        require_fastpath=True,
        input_tile_group=input_tile_group,
    )
    reference = build_mode1_engine(
        device,
        backend="torch",
        read_var=read_var,
        dtype=dtype,
        require_fastpath=False,
        input_tile_group=input_tile_group,
    )
    x_sliced = prepare_mode1_tensor(optimized, x, is_weight=False)
    weight_sliced = prepare_mode1_tensor(optimized, weight, is_weight=True)

    expected = reference(x_sliced, weight_sliced)
    actual = optimized(x_sliced, weight_sliced)
    metrics = compare_mode1_outputs(actual, expected)
    metrics["shape"] = [tokens, in_features, out_features]
    metrics["read_var"] = float(read_var)
    metrics["dtype"] = str(dtype)
    metrics["fastpath_counters"] = optimized.get_fastpath_counters()
    return metrics


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_path(
    engine: DPETensorMultiMode,
    x_sliced: SlicedDataMultiMode,
    weight_sliced: SlicedDataMultiMode,
    *,
    warmup: int,
    repeat: int,
) -> dict:
    for _ in range(int(warmup)):
        engine(x_sliced, weight_sliced)
    _synchronize(engine.device)

    samples_ms = []
    for _ in range(int(repeat)):
        _synchronize(engine.device)
        start = time.perf_counter()
        engine(x_sliced, weight_sliced)
        _synchronize(engine.device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "samples_ms": samples_ms,
        "fastpath_counters": engine.get_fastpath_counters(),
    }


def parse_shape(value: str) -> tuple[int, int, int]:
    if value in NAMED_SHAPES:
        return NAMED_SHAPES[value]
    parts = value.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be a name or TOKENS,IN,OUT")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Mode1 differential-pair microbenchmark")
    parser.add_argument("--shape", type=parse_shape, default=NAMED_SHAPES["qkv"])
    parser.add_argument(
        "--path",
        choices=["reference", "pair-direct", "gdiff-direct", "both", "all"],
        default="both",
    )
    parser.add_argument("--read-var", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--input-tile-group", type=int, default=1)
    parser.add_argument("--gdiff-schedule", choices=["auto", "owner", "grouped"], default="auto")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokens, in_features, out_features = args.shape
    torch.manual_seed(7)
    x = torch.randn((tokens, in_features), device=device, dtype=dtype)
    weight = torch.randn((in_features, out_features), device=device, dtype=dtype)

    preparation_engine = build_mode1_engine(
        device,
        backend="triton_gidx",
        read_var=args.read_var,
        dtype=dtype,
        require_fastpath=False,
        input_tile_group=args.input_tile_group,
    )
    x_sliced = prepare_mode1_tensor(preparation_engine, x, is_weight=False)
    weight_sliced = prepare_mode1_tensor(preparation_engine, weight, is_weight=True)

    result = {
        "shape": [tokens, in_features, out_features],
        "read_var": float(args.read_var),
        "dtype": str(dtype),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "paths": {},
    }
    if args.path in ("reference", "both", "all"):
        reference = build_mode1_engine(
            device,
            backend="torch",
            read_var=args.read_var,
            dtype=dtype,
            require_fastpath=False,
            input_tile_group=args.input_tile_group,
        )
        result["paths"]["reference"] = benchmark_path(
            reference,
            x_sliced,
            weight_sliced,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    if args.path in ("pair-direct", "both", "all"):
        optimized = build_mode1_engine(
            device,
            backend="triton_gidx",
            read_var=args.read_var,
            dtype=dtype,
            require_fastpath=True,
            input_tile_group=args.input_tile_group,
        )
        result["paths"]["pair_direct"] = benchmark_path(
            optimized,
            x_sliced,
            weight_sliced,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    if args.path in ("gdiff-direct", "all"):
        optimized = build_mode1_engine(
            device,
            backend="triton_gidx",
            read_var=args.read_var,
            dtype=dtype,
            require_fastpath=True,
            input_tile_group=args.input_tile_group,
            gdiff_direct=True,
            gdiff_schedule=args.gdiff_schedule,
        )
        result["paths"]["gdiff_direct"] = benchmark_path(
            optimized,
            x_sliced,
            weight_sliced,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    if "reference" in result["paths"] and "pair_direct" in result["paths"]:
        result["speedup"] = (
            result["paths"]["reference"]["mean_ms"]
            / result["paths"]["pair_direct"]["mean_ms"]
        )
    if "pair_direct" in result["paths"] and "gdiff_direct" in result["paths"]:
        result["gdiff_speedup_vs_pair"] = (
            result["paths"]["pair_direct"]["mean_ms"]
            / result["paths"]["gdiff_direct"]["mean_ms"]
        )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
