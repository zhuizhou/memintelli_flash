# -*- coding: utf-8 -*-
"""Memintelli example 13: Llama-3.1-8B WikiText inference with multimode CIM.

This example replaces Hugging Face ``nn.Linear`` layers with ``LinearMem`` and
evaluates token-level perplexity on WikiText-2. It is intentionally written as a
single-file example so users can inspect and modify the mode/execution knobs.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memintelli.NN_layers.linear import LinearMem, tensor_bytes
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def parse_int_tuple(value: str) -> tuple[int, ...]:
    items = value.replace(",", " ").split()
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer tuple: {value!r}") from exc


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device, *, for_model: bool = False) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value != "auto":
        raise ValueError(f"unknown dtype {value!r}")
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if for_model else torch.float32


def format_gb(num_bytes: int | float) -> str:
    return f"{float(num_bytes) / 1024**3:.2f} GB"


def iter_linearmem(model: nn.Module) -> list[LinearMem]:
    return [module for module in model.modules() if isinstance(module, LinearMem)]


def replace_linear_with_linearmem(
    module: nn.Module,
    mem_args: dict,
    *,
    skip_lm_head: bool,
    linearmem_device: str,
    prefix: str = "",
    replaced: list[str] | None = None,
) -> list[str]:
    """Recursively replace ``nn.Linear`` modules with CPU-created ``LinearMem``."""
    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if skip_lm_head and (name == "lm_head" or full_name.endswith(".lm_head")):
                continue

            new_layer = LinearMem(
                engine=mem_args["engine"],
                in_features=child.in_features,
                out_features=child.out_features,
                input_slice=mem_args["input_slice"],
                weight_slice=mem_args["weight_slice"],
                bias=child.bias is not None,
                device=mem_args["engine"].device if linearmem_device == "engine" else "cpu",
                dtype=child.weight.dtype,
                bw_e=mem_args["bw_e"],
                input_paral_size=mem_args["input_paral_size"],
                weight_paral_size=mem_args["weight_paral_size"],
                input_quant_gran=mem_args["input_quant_gran"],
                weight_quant_gran=mem_args["weight_quant_gran"],
                skip_initial_mapping=True,
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.detach().cpu())
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias.detach().cpu())

            setattr(module, name, new_layer)
            replaced.append(full_name)
            del child
            gc.collect()
        else:
            replace_linear_with_linearmem(
                child,
                mem_args,
                skip_lm_head=skip_lm_head,
                linearmem_device=linearmem_device,
                prefix=full_name,
                replaced=replaced,
            )

    return replaced


def pinned_g_bytes(layer: LinearMem) -> int:
    return sum(tensor_bytes(value) for value in getattr(layer, "_pinned_buffers", {}).values())


def prepare_linearmem_layers(
    model: nn.Module,
    *,
    execution: str,
    free_weights: bool,
    gpu_memory_reserve_gb: float,
    pin_policy: str,
) -> None:
    """Prepare LinearMem layers without accumulating every mapped G tensor on GPU."""
    layers = iter_linearmem(model)
    if not layers:
        print("[Memintelli] No LinearMem layers found.")
        return

    engine_device = layers[0].engine.device
    for idx, layer in enumerate(tqdm(layers, desc="Preparing LinearMem layers")):
        layer._prepare_inference_weight(
            streaming=True,
            free_weights=free_weights,
            pin_policy=pin_policy,
        )
        if torch.cuda.is_available() and torch.device(engine_device).type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    sizes = [pinned_g_bytes(layer) for layer in layers]
    total_g = sum(sizes)
    print(f"[Memintelli] Prepared {len(layers)} LinearMem layers; compressed G size: {format_gb(total_g)}")

    streaming_set: set[int]
    if execution == "memory":
        streaming_set = set(range(len(layers)))
    elif execution == "speed":
        streaming_set = set()
    elif execution == "auto":
        if not (torch.cuda.is_available() and torch.device(engine_device).type == "cuda"):
            streaming_set = set(range(len(layers)))
        else:
            props = torch.cuda.get_device_properties(engine_device)
            gpu_total = props.total_memory
            gpu_used = torch.cuda.memory_allocated(engine_device)
            pending_cpu_params = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
                if p.device.type == "cpu" and p.numel() > 0
            )
            budget = gpu_total - gpu_used - pending_cpu_params - int(gpu_memory_reserve_gb * 1024**3)
            print(
                "[Memintelli] auto execution budget for resident G: "
                f"{format_gb(max(0, budget))} "
                f"(total={format_gb(gpu_total)}, used={format_gb(gpu_used)}, "
                f"pending_params={format_gb(pending_cpu_params)}, reserve={gpu_memory_reserve_gb:.1f} GB)"
            )
            if total_g <= budget:
                streaming_set = set()
            else:
                streaming_set = set()
                resident = total_g
                for idx in sorted(range(len(layers)), key=lambda i: sizes[i], reverse=True):
                    if resident <= budget:
                        break
                    streaming_set.add(idx)
                    resident -= sizes[idx]
    else:
        raise ValueError(f"unknown execution mode {execution!r}")

    for idx, layer in enumerate(layers):
        if idx in streaming_set:
            layer._streaming = True
        else:
            layer._load_to_device(engine_device)
            layer._pinned_buffers.clear()
            layer._streaming = False

    streaming_layers = [layer for layer in layers if layer._streaming]
    for layer in layers:
        object.__setattr__(layer, "_next_streaming_layer", None)
    if streaming_layers:
        for idx in range(len(streaming_layers) - 1):
            object.__setattr__(streaming_layers[idx], "_next_streaming_layer", streaming_layers[idx + 1])
        object.__setattr__(streaming_layers[-1], "_next_streaming_layer", streaming_layers[0])

    resident_g = sum(size for idx, size in enumerate(sizes) if idx not in streaming_set)
    streaming_g = total_g - resident_g
    print(
        "[Memintelli] execution layout: "
        f"{len(layers) - len(streaming_set)} resident layers ({format_gb(resident_g)}), "
        f"{len(streaming_set)} streaming layers ({format_gb(streaming_g)})"
    )


def build_engine(args: argparse.Namespace, device: torch.device) -> DPETensorMultiMode:
    rdac_bits = args.rdac_bits if args.rdac_bits is not None else (4 if args.mode == 1 else 1)
    g_level = args.g_level if args.g_level is not None else (16 if args.mode == 1 else 2)
    engine_dtype = resolve_dtype(args.engine_dtype, device, for_model=False)
    backend = args.backend
    if backend == "auto":
        backend = "triton_gidx" if device.type == "cuda" else "torch"

    return DPETensorMultiMode(
        HGS=args.hgs,
        LGS=args.lgs,
        write_variation=args.write_variation,
        read_variation=args.read_variation,
        vnoise=args.vnoise,
        rate_stuck_HGS=args.rate_stuck_hgs,
        rate_stuck_LGS=args.rate_stuck_lgs,
        rdac=2**rdac_bits,
        g_level=g_level,
        radc=2**args.radc_bits,
        mode=args.mode,
        mode2_input_mode=args.mode2_input_mode,
        mode1_paral_size=(args.array_size, args.array_size),
        mode1_adc_per_tile=True,
        inference_chunk_size=int(args.chunk_mb * 1024 * 1024),
        fast_inference=not args.no_fast_inference,
        fast_inference_backend=backend,
        triton_input_precision=args.triton_precision,
        triton_block_r=args.triton_block_r,
        triton_block_l=args.triton_block_l,
        triton_block_k=args.triton_block_k,
        triton_output_chunk_limit=args.triton_output_chunk_limit,
        triton_auto_config=args.triton_auto_config,
        triton_gidx_fuse_input_slices=True,
        triton_reuse_input_voltage=True,
        triton_gidx_direct_final_output=True,
        triton_direct_final_output=True,
        triton_mode1_gidx_direct_final=True,
        triton_mode1_input_tile_group=args.mode1_input_tile_group,
        triton_mode1_chunked_direct_final=args.mode1_chunked_direct_final,
        triton_mode2_diff_direct_final=True,
        triton_mode2_diff_fuse_input_slices=True,
        direct_output_chunk_write=True,
        read_variation_seed=args.read_variation_seed,
        write_variation_mode=args.write_variation_mode,
        conductance_dtype=engine_dtype,
        compute_dtype=engine_dtype,
        profile=args.profile_engine,
        device=device,
    )


def load_wikitext2_text(max_lines: int, cache_dir: str | None) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to run this example: pip install datasets") from exc

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir)
    lines = [line for line in dataset["text"] if line and line.strip()]
    if max_lines > 0:
        lines = lines[:max_lines]
    return "\n\n".join(lines)


def load_model_and_tokenizer(args: argparse.Namespace, model_dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to run this example: pip install transformers accelerate") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        token=args.hf_token,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=model_dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype, **kwargs)
    return model, tokenizer


@torch.no_grad()
def evaluate_token_ppl(
    model: nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_length: int,
    stride: int,
    max_windows: int,
) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0
    window_count = 0

    for start in tqdm(range(0, seq_len, stride), desc="Evaluating WikiText PPL"):
        begin_loc = max(start + stride - max_length, 0)
        end_loc = min(start + stride, seq_len)
        target_len = end_loc - start
        if target_len <= 0:
            continue

        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-target_len] = -100

        outputs = model(input_ids=input_ids_chunk, labels=target_ids, use_cache=False)
        total_nll += float(outputs.loss.item()) * target_len
        total_tokens += target_len
        del outputs

        window_count += 1
        if max_windows > 0 and window_count >= max_windows:
            break
        if torch.cuda.is_available() and window_count % 8 == 0:
            torch.cuda.empty_cache()

    return math.exp(total_nll / max(1, total_tokens))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated models.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--engine-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--mode2-input-mode", choices=["signed", "differential"], default="differential")
    parser.add_argument("--execution", choices=["speed", "memory", "auto"], default="speed")
    parser.add_argument(
        "--linearmem-device",
        choices=["cpu", "engine"],
        default="cpu",
        help="Where LinearMem Parameters are created during replacement. Use engine on large GPUs.",
    )
    parser.add_argument("--array-size", type=int, default=64)
    parser.add_argument("--input-slice", type=parse_int_tuple, default=(1, 1, 1, 1, 1))
    parser.add_argument("--weight-slice", type=parse_int_tuple, default=(1, 1, 1, 1, 1))
    parser.add_argument("--rdac-bits", type=int, default=None)
    parser.add_argument("--radc-bits", type=int, default=8)
    parser.add_argument("--g-level", type=int, default=None)
    parser.add_argument("--read-variation", type=float, default=0.05)
    parser.add_argument("--read-variation-seed", type=int, default=None)
    parser.add_argument("--write-variation", type=float, default=0.0)
    parser.add_argument("--write-variation-mode", choices=["materialized", "virtual"], default="materialized")
    parser.add_argument("--vnoise", type=float, default=0.0)
    parser.add_argument("--rate-stuck-hgs", type=float, default=0.0)
    parser.add_argument("--rate-stuck-lgs", type=float, default=0.0)
    parser.add_argument("--hgs", type=float, default=1e-5)
    parser.add_argument("--lgs", type=float, default=1e-8)
    parser.add_argument("--backend", choices=["auto", "torch", "triton", "triton_gidx"], default="auto")
    parser.add_argument("--no-fast-inference", action="store_true")
    parser.add_argument("--triton-precision", choices=["ieee", "tf32", "tf32x3"], default="tf32")
    parser.add_argument("--triton-block-r", type=int, default=32)
    parser.add_argument("--triton-block-l", type=int, default=16)
    parser.add_argument("--triton-block-k", type=int, default=64)
    parser.add_argument("--triton-output-chunk-limit", type=int, default=256)
    parser.add_argument("--triton-auto-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode1-input-tile-group", type=int, default=1)
    parser.add_argument("--mode1-chunked-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-mb", type=float, default=64.0)
    parser.add_argument("--execution-reserve-gb", type=float, default=4.0)
    parser.add_argument("--pin-policy", choices=["persistent", "window"], default="persistent")
    parser.add_argument("--keep-fp-weights", action="store_true")
    parser.add_argument("--simulate-lm-head", action="store_true")
    parser.add_argument("--max-lines", type=int, default=64, help="WikiText non-empty lines. 0 means full test split.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=8, help="0 means evaluate all windows.")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--profile-engine", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the engine and print config without downloading a model.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    model_dtype = resolve_dtype(args.model_dtype, device, for_model=True)
    engine = build_engine(args, device)
    rdac_bits = args.rdac_bits if args.rdac_bits is not None else (4 if args.mode == 1 else 1)
    g_level = args.g_level if args.g_level is not None else (16 if args.mode == 1 else 2)

    print(
        "[Memintelli] config: "
        f"model={args.model}, mode={args.mode}, execution={args.execution}, "
        f"device={device}, model_dtype={model_dtype}, engine_dtype={engine.compute_dtype}, "
        f"array={args.array_size}, slices={args.input_slice}/{args.weight_slice}, "
        f"rdac={rdac_bits}b, radc={args.radc_bits}b, g_level={g_level}"
    )
    if args.dry_run:
        return

    torch.set_float32_matmul_precision("high")
    model, tokenizer = load_model_and_tokenizer(args, model_dtype)
    mem_args = {
        "engine": engine,
        "input_slice": args.input_slice,
        "weight_slice": args.weight_slice,
        "bw_e": None,
        "input_paral_size": (1, args.array_size),
        "weight_paral_size": (args.array_size, args.array_size),
        "input_quant_gran": (1, args.array_size),
        "weight_quant_gran": (args.array_size, args.array_size),
    }
    replaced = replace_linear_with_linearmem(
        model,
        mem_args,
        skip_lm_head=not args.simulate_lm_head,
        linearmem_device=args.linearmem_device,
    )
    print(f"[Memintelli] replaced {len(replaced)} nn.Linear modules with LinearMem")
    if not args.simulate_lm_head:
        print("[Memintelli] lm_head is kept as torch nn.Linear; pass --simulate-lm-head to include it.")

    prepare_linearmem_layers(
        model,
        execution=args.execution,
        free_weights=not args.keep_fp_weights,
        gpu_memory_reserve_gb=args.execution_reserve_gb,
        pin_policy=args.pin_policy,
    )

    model = model.eval().to(device)
    text = load_wikitext2_text(args.max_lines, args.dataset_cache_dir)
    ppl = evaluate_token_ppl(
        model,
        tokenizer,
        text,
        device,
        max_length=args.max_length,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    print(f"Token-level PPL on WikiText-2: {ppl:.4f}")


if __name__ == "__main__":
    main()
