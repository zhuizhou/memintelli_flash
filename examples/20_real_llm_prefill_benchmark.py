# -*- coding: utf-8 -*-
"""Benchmark real HuggingFace LLM prefill with MemIntelli Linear replacement.

Rows run in isolated subprocesses with independent PYTHONPATHs. This lets us
compare real-weight HF float inference, v2/v3 memristive replacement, and the
original MemIntelli replacement attempt without cross-import contamination.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


ROOT = Path(__file__).resolve().parents[1]


WORKER_CODE = r"""
import argparse
import concurrent.futures
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer

try:
    from memintelli.NN_layers.output_blocking import OutputBlockedLinearMem as GenericOutputBlockedLinearMem
except ImportError:
    GenericOutputBlockedLinearMem = None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def cuda_profiler_start(device):
    if device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop(device):
    if device.type == "cuda":
        torch.cuda.cudart().cudaProfilerStop()


def config_vocab_size(config):
    vocab = getattr(config, "vocab_size", None)
    if vocab is None and hasattr(config, "text_config"):
        vocab = getattr(config.text_config, "vocab_size", None)
    return int(vocab or 32000)


ARC_EASY_SMOKE_EXAMPLES = [
    {
        "id": "arc_easy_smoke_0001",
        "question": "Which planet do people live on?",
        "choices": ["Mars", "Earth", "Venus", "Jupiter"],
        "answer": "B",
    },
    {
        "id": "arc_easy_smoke_0002",
        "question": "What gas do plants take in from the air for photosynthesis?",
        "choices": ["Oxygen", "Carbon dioxide", "Helium", "Nitrogen"],
        "answer": "B",
    },
    {
        "id": "arc_easy_smoke_0003",
        "question": "Which tool is used to measure temperature?",
        "choices": ["Thermometer", "Ruler", "Scale", "Clock"],
        "answer": "A",
    },
    {
        "id": "arc_easy_smoke_0004",
        "question": "Water freezes when it becomes which state of matter?",
        "choices": ["Gas", "Liquid", "Solid", "Plasma"],
        "answer": "C",
    },
]


def load_classification_examples(args):
    if args.classification_task not in {"arc_easy_smoke", "multi_choice_json"}:
        return []
    if not args.classification_examples_json:
        if args.classification_task == "multi_choice_json":
            raise ValueError("multi_choice_json classification requires --classification-examples-json")
        return list(ARC_EASY_SMOKE_EXAMPLES)
    with open(args.classification_examples_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    if not isinstance(data, list):
        raise ValueError("--classification-examples-json must be a JSON list or contain an 'examples' list")
    return [item for item in data if isinstance(item, dict)]


def normalize_answer_label(example, labels):
    answer = example.get("answer", example.get("label", example.get("answerKey")))
    if answer is None and "answer_index" in example:
        idx = int(example["answer_index"])
        return labels[idx] if 0 <= idx < len(labels) else ""
    answer = str(answer).strip()
    if answer in labels:
        return answer
    if answer.isdigit():
        idx = int(answer)
        return labels[idx] if 0 <= idx < len(labels) else ""
    return answer[:1].upper()


def choice_texts_from_example(example, labels):
    choices = example.get("choices") or []
    if isinstance(choices, dict):
        choices = [choices.get(label, "") for label in labels]
    choices = [str(choice) for choice in choices]
    if len(choices) < len(labels):
        choices = choices + [""] * (len(labels) - len(choices))
    return choices[: len(labels)]


def prompt_from_example(example, labels):
    choices = choice_texts_from_example(example, labels)
    lines = [f"Question: {str(example.get('question', '')).strip()}"]
    for label, choice in zip(labels, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def token_id_for_choice_label(tokenizer, label):
    candidates = [f" {label}", label]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return int(ids[-1])
    return 0


def build_multi_choice_classification_batch(args, tokenizer, device):
    labels = [str(item) for item in (args.classification_choice_labels or ["A", "B", "C", "D"])]
    examples = load_classification_examples(args)
    if not examples:
        raise ValueError(f"{args.classification_task} classification requires at least one example")
    selected = [examples[(args.seed + idx) % len(examples)] for idx in range(max(1, int(args.batch)))]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    encoded_rows = []
    attention_rows = []
    prompts = []
    choice_text_rows = []
    gold_labels = []
    example_ids = []
    for example in selected:
        prompt = prompt_from_example(example, labels)
        choice_text_rows.append(choice_texts_from_example(example, labels))
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) > int(args.seq):
            ids = ids[-int(args.seq):]
        pad_len = max(0, int(args.seq) - len(ids))
        encoded_rows.append([pad_token_id] * pad_len + ids)
        attention_rows.append([0] * pad_len + [1] * len(ids))
        prompts.append(prompt)
        gold_labels.append(normalize_answer_label(example, labels))
        example_ids.append(str(example.get("id", "")))
    input_ids = torch.tensor(encoded_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
    choice_token_ids = [token_id_for_choice_label(tokenizer, label) for label in labels]
    return input_ids, attention_mask, {
        "classification_task": args.classification_task,
        "classification_examples_json": args.classification_examples_json,
        "classification_example_ids": example_ids,
        "classification_choice_labels": labels,
        "classification_choice_texts": choice_text_rows,
        "classification_choice_token_ids": choice_token_ids,
        "classification_gold_labels": gold_labels,
        "classification_prompt_count": len(prompts),
        "classification_prompt_preview": prompts[: min(2, len(prompts))],
    }


def resolve_classification_token_ids(args, tokenizer=None):
    if args.classification_token_ids:
        ids = [int(token_id) for token_id in args.classification_token_ids]
    elif args.classification_task in {"arc_easy_smoke", "multi_choice_json"}:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        labels = [str(item) for item in (args.classification_choice_labels or ["A", "B", "C", "D"])]
        ids = [token_id_for_choice_label(tokenizer, label) for label in labels]
    else:
        ids = list(range(max(1, int(args.classification_candidate_count))))
    if not ids:
        raise ValueError("classification token-id selection produced no token ids")
    ids = sorted({int(token_id) for token_id in ids if int(token_id) >= 0})
    if not ids:
        raise ValueError("classification token-id selection produced no valid token ids")
    return ids


def resolve_lm_head_output_token_ids(args):
    if args.lm_head_output_select == "all":
        return []
    if args.lm_head_output_select != "label_tokens":
        raise ValueError(f"unsupported lm_head output select: {args.lm_head_output_select}")
    if args.workload != "classification":
        raise ValueError("--lm-head-output-select label_tokens requires --workload classification")
    if args.classification_scoring != "label_token":
        raise ValueError("--lm-head-output-select label_tokens only supports --classification-scoring label_token")
    return resolve_classification_token_ids(args)


def prepare_config(config, args):
    if args.language_model_only and hasattr(config, "language_model_only"):
        setattr(config, "language_model_only", True)
    return config


def should_use_multimodal_loader(config, args):
    if args.model_loader == "multimodal":
        return True
    if args.model_loader == "causal":
        return False
    arch = getattr(config, "architectures", []) or []
    return hasattr(config, "vision_config") or any("ConditionalGeneration" in item for item in arch)


def load_device_map(args):
    if not args.load_on_device:
        return None
    if args.device == "auto":
        return "auto"
    if args.device == "cuda":
        return {"": "cuda:0"}
    if args.device.startswith("cuda:"):
        return {"": args.device}
    return None


def load_hf_model(args, config, dtype):
    kwargs = dict(
        dtype=dtype,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    device_map = load_device_map(args)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if should_use_multimodal_loader(config, args):
        from transformers import AutoModelForMultimodalLM

        return AutoModelForMultimodalLM.from_pretrained(args.model_path, **kwargs)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)


def alloc_mb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def tensor_bytes(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def tensor_pinned_bytes(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        return tensor_bytes(value) if value.device.type == "cpu" and value.is_pinned() else 0
    if isinstance(value, (tuple, list)):
        return sum(tensor_pinned_bytes(item) for item in value)
    return 0


def tensor_device_bytes(value):
    if value is None:
        return {}
    if torch.is_tensor(value):
        return {value.device.type: value.numel() * value.element_size()}
    out = {}
    if isinstance(value, (tuple, list)):
        for item in value:
            for key, size in tensor_device_bytes(item).items():
                out[key] = out.get(key, 0) + size
    return out


STREAM_WEIGHT_ATTRS = ("G_indices", "G", "max_data", "e_bias", "mode1_w_max")


def module_stream_weight_bytes(module) -> int:
    ws = getattr(module, "weight_sliced", None)
    if ws is None:
        return 0
    return sum(tensor_bytes(getattr(ws, attr, None)) for attr in STREAM_WEIGHT_ATTRS)


def module_stream_weight_bytes_from_hints(name: str, hints: dict) -> int:
    hint = hints.get(name) or {}
    try:
        return int(float(hint.get("stream_weight_bytes") or 0.0))
    except (TypeError, ValueError):
        return 0


def select_budgeted_window_pin_cache_names(named_layers, budget_bytes, pin_hints, args):
    if budget_bytes <= 0 or not pin_hints:
        return None, 0
    scored = []
    for name, module in named_layers:
        hint = pin_hints.get(name)
        if not hint:
            continue
        layer_bytes = module_stream_weight_bytes_from_hints(name, pin_hints)
        if layer_bytes <= 0:
            layer_bytes = estimate_stream_weight_bytes(module, args)
        if layer_bytes <= 0:
            continue
        benefit = float(hint.get("benefit_ms") or 0.0)
        score = benefit / max(float(layer_bytes), 1.0)
        scored.append((score, benefit, -layer_bytes, name, layer_bytes))
    selected = set()
    used = 0
    for _score, _benefit, neg_bytes, name, layer_bytes in sorted(scored, reverse=True):
        layer_bytes = int(layer_bytes or -neg_bytes)
        if used + layer_bytes <= budget_bytes:
            selected.add(name)
            used += layer_bytes
    return selected, used


def merge_device_bytes(dst, value):
    for key, size in tensor_device_bytes(value).items():
        dst[key] = dst.get(key, 0) + size


def format_shape(value):
    if value is None or not torch.is_tensor(value):
        return []
    return [int(dim) for dim in value.shape]


def element_size(dtype, fallback=4):
    try:
        return torch.empty((), dtype=dtype).element_size()
    except Exception:
        return int(fallback)


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def slice_method_len(value):
    try:
        return int(len(value))
    except TypeError:
        return 0


def resolve_quant_geometry(rows, cols, paral_size, quant_gran):
    qg = quant_gran
    if qg == "per-matrix":
        qg = (rows, cols)
    elif qg == "per-row":
        qg = (1, cols)
    elif qg == "per-col":
        qg = (rows, 1)
    qg = list(qg)
    ps = list(paral_size)
    qg[0] = math.ceil(max(1, int(qg[0])) / max(1, int(ps[0]))) * max(1, int(ps[0]))
    qg[1] = math.ceil(max(1, int(qg[1])) / max(1, int(ps[1]))) * max(1, int(ps[1]))
    ngr = math.ceil(max(1, int(rows)) / qg[0])
    ngc = math.ceil(max(1, int(cols)) / qg[1])
    ndr = qg[0] // max(1, int(ps[0]))
    ndc = qg[1] // max(1, int(ps[1]))
    return qg, ps, ngr, ngc, ndr, ndc


def infer_sliced_tensor_numel(rows, cols, paral_size, quant_gran, slice_count, batch_positions=1):
    _qg, ps, ngr, ngc, ndr, ndc = resolve_quant_geometry(rows, cols, paral_size, quant_gran)
    return int(batch_positions) * ngr * ndr * ngc * ndc * int(slice_count) * ps[0] * ps[1]


def infer_scale_numel(rows, cols, paral_size, quant_gran, batch_positions=1):
    _qg, _ps, ngr, ngc, ndr, ndc = resolve_quant_geometry(rows, cols, paral_size, quant_gran)
    return int(batch_positions) * ngr * ndr * ngc * ndc


def collect_layer_buffer_accounting(model, LinearMem, args, prep_info=None):
    if LinearMem is None or args.kind == "hf":
        return {
            "enabled": bool(args.collect_layer_buffer_accounting),
            "complete": False,
            "reason": "not a MemIntelli row",
        }

    layers = []
    totals = {
        "resident_weight_state_bytes": 0,
        "resident_weight_state_device_bytes": {},
        "pinned_buffer_bytes": 0,
        "estimated_weight_sliced_data_bytes": 0,
        "estimated_weight_quantized_bytes": 0,
        "estimated_input_sliced_data_bytes_per_layer_forward": 0,
        "estimated_input_quantized_bytes_per_layer_forward": 0,
        "estimated_input_scale_bytes_per_layer_forward": 0,
        "estimated_adc_current_bytes_per_layer_forward": 0,
        "estimated_intermediate_output_bytes_per_layer_forward": 0,
        "estimated_restored_conductance_chunk_bytes_per_layer_forward": 0,
        "estimated_max_transient_bytes_per_layer_forward": 0,
    }
    field_counts = {}
    top_limit = max(0, int(getattr(args, "layer_buffer_accounting_limit", 12) or 0))
    positions = int(args.batch) * (1 if args.workload == "generation" and args.decode_steps == 0 else int(args.seq))
    if args.workload == "generation":
        positions = int(args.batch) * int(args.seq)

    for name, module in model.named_modules():
        if not isinstance(module, LinearMem):
            continue
        ws = getattr(module, "weight_sliced", None)
        if ws is None:
            continue
        layer_name = name or "<root>"
        weight_shape = [int(module.out_features), int(module.in_features)]
        weight_paral = tuple(getattr(ws, "paral_size", args.weight_paral_size))
        weight_quant = getattr(ws, "quant_gran", args.weight_quant_gran)
        input_paral = tuple(getattr(module, "input_paral_size", args.input_paral_size))
        input_quant = tuple(getattr(module, "input_quant_gran", args.input_quant_gran))
        weight_slice_count = max(1, slice_method_len(getattr(ws, "slice_method", args.weight_slice)))
        input_slice_count = max(1, slice_method_len(getattr(module, "input_slice_method", args.input_slice)))
        mode = int(getattr(getattr(module, "engine", None), "mode", args.mode if args.kind == "v3" else 0) or 0)
        input_dtype_bytes = element_size(getattr(module.weight, "dtype", torch.float32), 2)
        branch_multiplier = 2 if mode == 2 else 1

        resident_by_attr = {}
        resident_total = 0
        resident_devices = {}
        for attr in (
            "G_indices",
            "G",
            "max_data",
            "e_bias",
            "mode1_w_max",
            "sliced_data",
            "quantized_data",
            "sliced_data_p",
            "sliced_data_n",
            "quantized_data_p",
            "quantized_data_n",
            "max_data_p",
            "max_data_n",
            "e_bias_p",
            "e_bias_n",
        ):
            value = getattr(ws, attr, None)
            size = tensor_bytes(value)
            if size:
                resident_by_attr[attr] = {
                    "bytes": size,
                    "mb": size / (1024 ** 2),
                    "device_bytes": tensor_device_bytes(value),
                    "shape": format_shape(value) if torch.is_tensor(value) else [],
                }
                resident_total += size
                merge_device_bytes(resident_devices, value)
                field_counts[attr] = field_counts.get(attr, 0) + 1

        pinned_bytes = 0
        pinned_by_attr = {}
        for attr, value in getattr(module, "_pinned_buffers", {}).items():
            size = tensor_bytes(value)
            if size:
                pinned_size = tensor_pinned_bytes(value)
                pinned_by_attr[attr] = {
                    "bytes": size,
                    "pinned_bytes": pinned_size,
                    "mb": size / (1024 ** 2),
                    "pinned_mb": pinned_size / (1024 ** 2),
                    "shape": format_shape(value) if torch.is_tensor(value) else [],
                }
                pinned_bytes += pinned_size
                field_counts[f"pinned_{attr}"] = field_counts.get(f"pinned_{attr}", 0) + 1
        for attr, value in getattr(module, "_active_pinned_buffers", {}).items():
            pinned_size = tensor_pinned_bytes(value)
            if pinned_size:
                pinned_bytes += pinned_size
                field_counts[f"active_pinned_{attr}"] = field_counts.get(f"active_pinned_{attr}", 0) + 1

        in_features = int(module.in_features)
        out_features = int(module.out_features)
        weight_sliced_numel = infer_sliced_tensor_numel(
            in_features,
            out_features,
            weight_paral,
            weight_quant,
            weight_slice_count,
        )
        weight_scale_numel = infer_scale_numel(in_features, out_features, weight_paral, weight_quant)
        input_sliced_numel = infer_sliced_tensor_numel(
            int(positions),
            in_features,
            input_paral,
            input_quant,
            input_slice_count,
        )
        input_scale_numel = infer_scale_numel(int(positions), in_features, input_paral, input_quant)
        weight_sliced_bytes = weight_sliced_numel * branch_multiplier * element_size(torch.uint8, 1)
        weight_quantized_bytes = in_features * out_features * input_dtype_bytes
        input_sliced_bytes = input_sliced_numel * branch_multiplier * element_size(torch.uint8, 1)
        input_quantized_bytes = int(positions) * in_features * input_dtype_bytes
        input_scale_bytes = input_scale_numel * 4
        output_bytes = int(positions) * out_features * 4
        adc_current_bytes = input_slice_count * weight_slice_count * int(positions) * out_features * 4 * branch_multiplier

        if resident_by_attr.get("G") and not resident_by_attr.get("G_indices"):
            restored_conductance_chunk_bytes = 0
        else:
            restored_conductance_chunk_bytes = max(1, weight_sliced_numel // max(1, input_slice_count)) * 4 * branch_multiplier
        max_transient_bytes = max(
            input_sliced_bytes + input_scale_bytes,
            restored_conductance_chunk_bytes + adc_current_bytes,
            output_bytes,
        )

        estimated = {
            "weight_sliced_data_bytes": weight_sliced_bytes,
            "weight_quantized_bytes": weight_quantized_bytes,
            "input_sliced_data_bytes_per_forward": input_sliced_bytes,
            "input_quantized_bytes_per_forward": input_quantized_bytes,
            "input_scale_bytes_per_forward": input_scale_bytes,
            "adc_current_bytes_per_forward": adc_current_bytes,
            "intermediate_output_bytes_per_forward": output_bytes,
            "restored_conductance_chunk_bytes_per_forward": restored_conductance_chunk_bytes,
            "max_transient_bytes_per_forward": max_transient_bytes,
        }

        totals["resident_weight_state_bytes"] += resident_total
        totals["pinned_buffer_bytes"] += pinned_bytes
        for key, value in resident_devices.items():
            totals["resident_weight_state_device_bytes"][key] = totals["resident_weight_state_device_bytes"].get(key, 0) + value
        for key, value in estimated.items():
            total_key = "estimated_" + key.replace("_per_forward", "_per_layer_forward")
            if total_key in totals:
                totals[total_key] += value

        layers.append({
            "name": layer_name,
            "in_features": in_features,
            "out_features": out_features,
            "mode": mode,
            "weight_paral_size": list(weight_paral),
            "weight_quant_gran": list(weight_quant) if isinstance(weight_quant, (list, tuple)) else weight_quant,
            "input_paral_size": list(input_paral),
            "input_quant_gran": list(input_quant) if isinstance(input_quant, (list, tuple)) else input_quant,
            "weight_slice_count": weight_slice_count,
            "input_slice_count": input_slice_count,
            "resident_weight_state_bytes": resident_total,
            "resident_weight_state_mb": resident_total / (1024 ** 2),
            "resident_weight_state_device_bytes": resident_devices,
            "pinned_buffer_bytes": pinned_bytes,
            "pinned_buffer_mb": pinned_bytes / (1024 ** 2),
            "resident_by_attr": resident_by_attr,
            "pinned_by_attr": pinned_by_attr,
            "estimated_buffers": {key: value for key, value in estimated.items()},
            "estimated_buffers_mb": {key.replace("_bytes", "_mb"): value / (1024 ** 2) for key, value in estimated.items()},
        })

    top_layers = sorted(
        layers,
        key=lambda item: (
            item["resident_weight_state_bytes"]
            + item["pinned_buffer_bytes"]
            + item["estimated_buffers"]["max_transient_bytes_per_forward"]
        ),
        reverse=True,
    )[:top_limit]
    totals_mb = {
        key.replace("_bytes", "_mb"): value / (1024 ** 2)
        for key, value in totals.items()
        if isinstance(value, (int, float))
    }
    device_mb = {
        key: value / (1024 ** 2)
        for key, value in totals["resident_weight_state_device_bytes"].items()
    }
    layer_count = len(layers)
    observed_resident = totals["resident_weight_state_bytes"] > 0 or totals["pinned_buffer_bytes"] > 0
    estimated_transient = totals["estimated_max_transient_bytes_per_layer_forward"] > 0
    return {
        "enabled": True,
        "complete": bool(layer_count and estimated_transient),
        "layer_count": layer_count,
        "accounted_layer_count": sum(1 for item in layers if item["estimated_buffers"]["max_transient_bytes_per_forward"] > 0),
        "observation_positions": positions,
        "execution_mode": args.execution_mode or "manual",
        "lazy_prepare": bool(args.lazy_prepare),
        "streaming": bool(args.streaming),
        "lazy_release_after_forward": bool(args.lazy_release_after_forward),
        "observed_resident_state_after_prepare_or_forward": bool(observed_resident),
        "estimated_transient_state_available": bool(estimated_transient),
        "field_counts": field_counts,
        "totals": totals,
        "totals_mb": totals_mb,
        "resident_weight_state_device_mb": device_mb,
        "top_layers": top_layers,
        "summary_note": (
            "Resident tensor bytes are directly observed from LinearMem buffers; transient input, ADC, "
            "restored conductance, and output bytes are shape-derived estimates for the audited workload."
        ),
        "prepare_info": prep_info or {},
    }


def collect_runtime_mechanism_counters(model, LinearMem, args):
    if LinearMem is None or args.kind == "hf":
        return {"enabled": False, "reason": "not a MemIntelli row"}
    totals = {
        "layer_count": 0,
        "offload_to_cpu_count": 0,
        "offload_to_cpu_bytes": 0,
        "sync_load_count": 0,
        "sync_load_bytes": 0,
        "prefetch_started_count": 0,
        "prefetch_wait_count": 0,
        "prefetch_bytes": 0,
        "prefetch_oom_count": 0,
        "prefetch_complete_on_arrival_count": 0,
        "prefetch_pending_on_arrival_count": 0,
        "next_restore_prefetch_started_count": 0,
        "next_restore_prefetch_skip_count": 0,
        "next_restore_prefetch_skip_streaming_count": 0,
        "next_restore_prefetch_skip_unprepared_count": 0,
        "release_gpu_tensor_count": 0,
        "release_gpu_tensor_bytes": 0,
        "lazy_prepare_count": 0,
        "input_slice_count": 0,
        "mapreduce_count": 0,
        "postprocess_count": 0,
        "release_prepared_count": 0,
        "pin_cpu_buffer_ms": 0.0,
        "sync_load_ms": 0.0,
        "prefetch_schedule_ms": 0.0,
        "prefetch_copy_ms": 0.0,
        "prefetch_copy_event_count": 0,
        "release_gpu_tensor_ms": 0.0,
        "window_pin_cache_hit_count": 0,
        "window_pin_cache_miss_count": 0,
        "window_pin_cache_store_count": 0,
        "window_pin_cache_bytes": 0,
        "lazy_prepare_ms": 0.0,
        "input_slice_ms": 0.0,
        "mapreduce_ms": 0.0,
        "postprocess_ms": 0.0,
        "release_prepared_ms": 0.0,
        "lazy_prepare_wall_ms": 0.0,
        "input_slice_wall_ms": 0.0,
        "mapreduce_wall_ms": 0.0,
        "postprocess_wall_ms": 0.0,
        "release_prepared_wall_ms": 0.0,
        "pinned_buffer_peak_bytes": 0,
        "current_pinned_buffer_bytes": 0,
        "pinned_buffer_count": 0,
        "prefetch_link_count": 0,
        "pending_prefetch_event_count": 0,
        "lazy_prepared_layers": 0,
        "streaming_layers": 0,
        "persistent_pin_layers": 0,
        "window_pin_layers": 0,
        "window_pin_width": 0,
        "window_pinned_peak_bytes_estimate": 0,
        "persistent_pinned_peak_bytes": 0,
        "gidx_attempt_count": 0,
        "gidx_success_count": 0,
        "gidx_fallback_count": 0,
        "gidx_read_noise_fallback_count": 0,
        "gidx_virtual_write_fallback_count": 0,
        "gidx_import_fallback_count": 0,
        "gidx_exception_fallback_count": 0,
        "gidx_read_noise_aware_attempt_count": 0,
        "gidx_read_noise_aware_success_count": 0,
        "activation_slice_fused_success_count": 0,
        "activation_slice_fused_fallback_count": 0,
        "output_finalize_fused_attempt_count": 0,
        "output_finalize_fused_success_count": 0,
        "output_finalize_fused_fallback_count": 0,
        "direct_final_output_attempt_count": 0,
        "direct_final_output_success_count": 0,
        "direct_final_output_fallback_count": 0,
        "direct_final_output_store_success_count": 0,
        "gidx_direct_final_output_attempt_count": 0,
        "gidx_direct_final_output_success_count": 0,
        "gidx_direct_final_output_fallback_count": 0,
        "gidx_direct_final_output_store_success_count": 0,
        "mode1_gidx_direct_final_attempt_count": 0,
        "mode1_gidx_direct_final_success_count": 0,
        "mode1_gidx_direct_final_fallback_count": 0,
        "mode1_gidx_direct_final_grouped_success_count": 0,
        "mode1_chunked_direct_final_attempt_count": 0,
        "mode1_chunked_direct_final_success_count": 0,
        "mode1_chunked_direct_final_fallback_count": 0,
        "mode2_diff_direct_final_attempt_count": 0,
        "mode2_diff_direct_final_success_count": 0,
        "mode2_diff_direct_final_fallback_count": 0,
        "mode2_diff_direct_final_store_success_count": 0,
    }
    layer_rows = []
    window_peak_candidates = []
    persistent_peak_bytes = 0
    seen_engines = set()
    for name, module in model.named_modules():
        if not isinstance(module, LinearMem):
            continue
        totals["layer_count"] += 1
        engine = getattr(module, "engine", None)
        if engine is not None and id(engine) not in seen_engines and hasattr(engine, "get_fastpath_counters"):
            seen_engines.add(id(engine))
            fastpath = engine.get_fastpath_counters()
            for key, value in fastpath.items():
                if isinstance(value, float):
                    totals[key] = float(totals.get(key, 0.0) or 0.0) + value
                else:
                    totals[key] = int(totals.get(key, 0) or 0) + int(value or 0)
        if hasattr(module, "runtime_mechanism_counters"):
            counters = module.runtime_mechanism_counters()
        else:
            pinned = getattr(module, "_pinned_buffers", {}) or {}
            active_pinned = getattr(module, "_active_pinned_buffers", {}) or {}
            window_cached = getattr(module, "_window_cached_buffers", {}) or {}
            current_pinned = sum(tensor_pinned_bytes(value) for value in pinned.values())
            current_pinned += sum(tensor_pinned_bytes(value) for value in active_pinned.values())
            current_pinned += sum(tensor_pinned_bytes(value) for value in window_cached.values())
            counters = {
                "current_pinned_buffer_bytes": current_pinned,
                "pinned_buffer_peak_bytes": current_pinned,
                "pinned_buffer_count": len(pinned) + len(active_pinned) + len(window_cached),
                "has_prefetch_link": getattr(module, "_next_streaming_layer", None) is not None,
                "has_pending_prefetch_event": getattr(module, "_prefetch_event", None) is not None,
                "lazy_prepared": bool(getattr(module, "_lazy_prepared", False)),
                "streaming": bool(getattr(module, "_streaming", False)),
                "streaming_pin_policy": getattr(module, "_streaming_pin_policy", "persistent"),
            }
        pin_policy = str(counters.get("streaming_pin_policy", "persistent") or "persistent")
        is_streaming_layer = bool(counters.get("streaming"))
        if is_streaming_layer and pin_policy == "window":
            totals["window_pin_layers"] += 1
        elif is_streaming_layer:
            totals["persistent_pin_layers"] += 1
        for key in [
            "offload_to_cpu_count",
            "offload_to_cpu_bytes",
            "sync_load_count",
            "sync_load_bytes",
            "prefetch_started_count",
            "prefetch_wait_count",
            "prefetch_bytes",
            "prefetch_oom_count",
            "prefetch_complete_on_arrival_count",
            "prefetch_pending_on_arrival_count",
            "next_restore_prefetch_started_count",
            "next_restore_prefetch_skip_count",
            "next_restore_prefetch_skip_streaming_count",
            "next_restore_prefetch_skip_unprepared_count",
            "release_gpu_tensor_count",
            "release_gpu_tensor_bytes",
            "window_pin_cache_hit_count",
            "window_pin_cache_miss_count",
            "window_pin_cache_store_count",
            "window_pin_cache_bytes",
            "lazy_prepare_count",
            "input_slice_count",
            "mapreduce_count",
            "postprocess_count",
            "release_prepared_count",
            "activation_slice_fused_success_count",
            "activation_slice_fused_fallback_count",
            "current_pinned_buffer_bytes",
            "pinned_buffer_count",
        ]:
            totals[key] += int(counters.get(key, 0) or 0)
        for key in [
            "pin_cpu_buffer_ms",
            "sync_load_ms",
            "prefetch_schedule_ms",
            "prefetch_copy_ms",
            "release_gpu_tensor_ms",
            "lazy_prepare_ms",
            "input_slice_ms",
            "mapreduce_ms",
            "postprocess_ms",
            "release_prepared_ms",
            "lazy_prepare_wall_ms",
            "input_slice_wall_ms",
            "mapreduce_wall_ms",
            "postprocess_wall_ms",
            "release_prepared_wall_ms",
        ]:
            totals[key] += float(counters.get(key, 0.0) or 0.0)
        totals["prefetch_link_count"] += int(bool(counters.get("has_prefetch_link")))
        totals["pending_prefetch_event_count"] += int(bool(counters.get("has_pending_prefetch_event")))
        totals["lazy_prepared_layers"] += int(bool(counters.get("lazy_prepared")))
        totals["streaming_layers"] += int(bool(counters.get("streaming")))
        peak = int(counters.get("pinned_buffer_peak_bytes", 0) or 0)
        if is_streaming_layer and pin_policy == "window":
            window_peak_candidates.append(peak)
        elif is_streaming_layer:
            persistent_peak_bytes += peak
        runtime_activity = any(
            float(counters.get(key, 0.0) or 0.0) > 0
            for key in [
                "pin_cpu_buffer_ms",
                "sync_load_ms",
                "prefetch_schedule_ms",
                "prefetch_copy_ms",
                "release_gpu_tensor_ms",
                "lazy_prepare_ms",
                "input_slice_ms",
                "mapreduce_ms",
                "postprocess_ms",
                "release_prepared_ms",
                "lazy_prepare_wall_ms",
                "input_slice_wall_ms",
                "mapreduce_wall_ms",
                "postprocess_wall_ms",
                "release_prepared_wall_ms",
            ]
        ) or any(
            int(counters.get(key, 0) or 0) > 0
            for key in [
                "lazy_prepare_count",
                "input_slice_count",
                "mapreduce_count",
                "postprocess_count",
                "release_prepared_count",
                "activation_slice_fused_success_count",
                "activation_slice_fused_fallback_count",
            ]
        )
        if peak or runtime_activity:
            layer_rows.append(
                {
                    "name": name or "<root>",
                    "pinned_buffer_peak_bytes": peak,
                    "streaming_pin_policy": pin_policy,
                    "stream_weight_bytes": module_stream_weight_bytes(module),
                    "prefetch_started_count": counters.get("prefetch_started_count", 0),
                    "prefetch_wait_count": counters.get("prefetch_wait_count", 0),
                    "sync_load_count": counters.get("sync_load_count", 0),
                    "prefetch_complete_on_arrival_count": counters.get("prefetch_complete_on_arrival_count", 0),
                    "prefetch_pending_on_arrival_count": counters.get("prefetch_pending_on_arrival_count", 0),
                    "pin_cpu_buffer_ms": counters.get("pin_cpu_buffer_ms", 0.0),
                    "sync_load_ms": counters.get("sync_load_ms", 0.0),
                    "prefetch_schedule_ms": counters.get("prefetch_schedule_ms", 0.0),
                    "prefetch_copy_ms": counters.get("prefetch_copy_ms", 0.0),
                    "release_gpu_tensor_ms": counters.get("release_gpu_tensor_ms", 0.0),
                    "window_pin_cache_hit_count": counters.get("window_pin_cache_hit_count", 0),
                    "window_pin_cache_miss_count": counters.get("window_pin_cache_miss_count", 0),
                    "window_pin_cache_store_count": counters.get("window_pin_cache_store_count", 0),
                    "window_pin_cache_bytes": counters.get("window_pin_cache_bytes", 0),
                    "lazy_prepare_count": counters.get("lazy_prepare_count", 0),
                    "input_slice_count": counters.get("input_slice_count", 0),
                    "mapreduce_count": counters.get("mapreduce_count", 0),
                    "postprocess_count": counters.get("postprocess_count", 0),
                    "release_prepared_count": counters.get("release_prepared_count", 0),
                    "activation_slice_fused_success_count": counters.get("activation_slice_fused_success_count", 0),
                    "activation_slice_fused_fallback_count": counters.get("activation_slice_fused_fallback_count", 0),
                    "lazy_prepare_ms": counters.get("lazy_prepare_ms", 0.0),
                    "input_slice_ms": counters.get("input_slice_ms", 0.0),
                    "mapreduce_ms": counters.get("mapreduce_ms", 0.0),
                    "postprocess_ms": counters.get("postprocess_ms", 0.0),
                    "release_prepared_ms": counters.get("release_prepared_ms", 0.0),
                    "lazy_prepare_wall_ms": counters.get("lazy_prepare_wall_ms", 0.0),
                    "input_slice_wall_ms": counters.get("input_slice_wall_ms", 0.0),
                    "mapreduce_wall_ms": counters.get("mapreduce_wall_ms", 0.0),
                    "postprocess_wall_ms": counters.get("postprocess_wall_ms", 0.0),
                    "release_prepared_wall_ms": counters.get("release_prepared_wall_ms", 0.0),
                }
            )
    window_peak_candidates = sorted(window_peak_candidates, reverse=True)
    window_width = 0
    if window_peak_candidates:
        # At most the current layer and one prefetched successor are intended
        # to be pinned simultaneously in window mode.
        window_width = 2 if totals["prefetch_link_count"] else 1
    window_peak_estimate = sum(window_peak_candidates[:window_width])
    totals["window_pin_width"] = window_width
    totals["window_pinned_peak_bytes_estimate"] = window_peak_estimate
    totals["persistent_pinned_peak_bytes"] = persistent_peak_bytes
    # The static window estimate assumes module execution follows the simple
    # prefetch chain. Real transformer graphs can leave additional prefetched
    # layers pending, so keep the report conservative by also considering the
    # directly observed current pinned footprint.
    totals["pinned_buffer_peak_bytes"] = max(
        persistent_peak_bytes + window_peak_estimate,
        int(totals.get("current_pinned_buffer_bytes", 0) or 0),
    )
    for key in [
        "offload_to_cpu_bytes",
        "sync_load_bytes",
        "prefetch_bytes",
        "release_gpu_tensor_bytes",
        "window_pin_cache_bytes",
        "pinned_buffer_peak_bytes",
        "current_pinned_buffer_bytes",
        "window_pinned_peak_bytes_estimate",
        "persistent_pinned_peak_bytes",
    ]:
        totals[key.replace("_bytes", "_mb")] = totals[key] / (1024 ** 2)
    top_layers = sorted(layer_rows, key=lambda item: item["pinned_buffer_peak_bytes"], reverse=True)[:8]
    layer_rows = sorted(layer_rows, key=lambda item: item["name"])
    return {
        "enabled": True,
        "reason": "ok" if totals["layer_count"] else "no LinearMem layers observed",
        **totals,
        "top_layers": top_layers,
        "layer_rows": layer_rows,
    }


def reset_runtime_mechanism_counters(model, LinearMem):
    if LinearMem is None:
        return
    for module in model.modules():
        if isinstance(module, LinearMem) and hasattr(module, "reset_runtime_mechanism_counters"):
            module.reset_runtime_mechanism_counters()


def merge_profile_summaries(summaries):
    merged = {}
    for summary in summaries:
        for label, item in summary.items():
            dst = merged.setdefault(label, {
                "count": 0,
                "total_ms": 0.0,
                "mean_ms": 0.0,
                "statuses": {},
                "reasons": {},
                "messages": {},
                "metadata": {},
            })
            dst["count"] += item.get("count", 0)
            dst["total_ms"] += item.get("total_ms", 0.0)
            for key in ("statuses", "reasons", "messages"):
                values = item.get(key) or {}
                if isinstance(values, dict):
                    for name, count in values.items():
                        name = str(name)
                        dst[key][name] = dst[key].get(name, 0) + int(count)
            values = item.get("metadata") or {}
            if isinstance(values, dict):
                for name, group in values.items():
                    if not isinstance(group, dict):
                        continue
                    name = str(name)
                    dst_group = dst["metadata"].setdefault(name, {
                        "count": 0,
                        "total_ms": 0.0,
                        "mean_ms": 0.0,
                    })
                    dst_group["count"] += int(group.get("count", 0) or 0)
                    dst_group["total_ms"] += float(group.get("total_ms", 0.0) or 0.0)
    for item in merged.values():
        item["mean_ms"] = item["total_ms"] / max(item["count"], 1)
        for group in item.get("metadata", {}).values():
            group["mean_ms"] = group["total_ms"] / max(group["count"], 1)
    return merged


def top_profile_events(summary, limit=12):
    items = sorted(summary.items(), key=lambda kv: kv[1].get("total_ms", 0.0), reverse=True)
    return [
        {
            "label": label,
            "count": int(item.get("count", 0)),
            "total_ms": float(item.get("total_ms", 0.0)),
            "mean_ms": float(item.get("mean_ms", 0.0)),
        }
        for label, item in items[:limit]
    ]


def _module_timing_category(name, module, LinearMem):
    if isinstance(module, LinearMem):
        return "linearmem_lm_head" if is_lm_head_name(name) or name.startswith("lm_head.") else "linearmem_body"
    if re.fullmatch(r"model\.layers\.\d+", name or ""):
        return "decoder_layer"
    if re.fullmatch(r"model\.layers\.\d+\.(self_attn|linear_attn)", name or ""):
        return "attention_block"
    if re.fullmatch(r"model\.layers\.\d+\.mlp", name or ""):
        return "mlp_block"
    if is_lm_head_name(name or ""):
        return "lm_head_wrapper"
    return None


class ModuleTimingContext:
    def __init__(self, model, LinearMem, enabled=False):
        self.model = model
        self.LinearMem = LinearMem
        self.enabled = bool(enabled)
        self.records = []
        self.handles = []
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        if not self.enabled:
            return self
        for name, module in self.model.named_modules():
            category = _module_timing_category(name, module, self.LinearMem)
            if category is None:
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name, category)))
            self.handles.append(module.register_forward_hook(self._post_hook(name, category)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _pre_hook(self, name, category):
        def hook(module, inputs):
            stack = getattr(module, "_memintelli_module_timing_stack", None)
            if stack is None:
                stack = []
                setattr(module, "_memintelli_module_timing_stack", stack)
            if self.cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                stack.append(("cuda", event, time.perf_counter()))
            else:
                stack.append(("cpu", None, time.perf_counter()))

        return hook

    def _post_hook(self, name, category):
        def hook(module, inputs, output):
            stack = getattr(module, "_memintelli_module_timing_stack", None) or []
            if not stack:
                return
            mode, start_event, start_wall = stack.pop()
            end_wall = time.perf_counter()
            if mode == "cuda":
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self.records.append(
                    {
                        "name": name or "<root>",
                        "category": category,
                        "mode": "cuda_event",
                        "start_event": start_event,
                        "end_event": end_event,
                        "wall_ms": (end_wall - start_wall) * 1000.0,
                    }
                )
            else:
                self.records.append(
                    {
                        "name": name or "<root>",
                        "category": category,
                        "mode": "wall",
                        "elapsed_ms": (end_wall - start_wall) * 1000.0,
                    }
                )

        return hook

    def summary(self, total_ms=None, limit=16):
        if not self.enabled:
            return {"enabled": False}
        if self.cuda:
            torch.cuda.synchronize()
        finalized = []
        for item in self.records:
            elapsed = item.get("elapsed_ms")
            if elapsed is None and item.get("mode") == "cuda_event":
                try:
                    elapsed = float(item["start_event"].elapsed_time(item["end_event"]))
                except Exception:
                    elapsed = float(item.get("wall_ms", 0.0) or 0.0)
            finalized.append(
                {
                    "name": item["name"],
                    "category": item["category"],
                    "elapsed_ms": float(elapsed or 0.0),
                    "wall_ms": float(item.get("wall_ms", elapsed or 0.0) or 0.0),
                }
            )
        by_category = {}
        for item in finalized:
            cat = item["category"]
            dst = by_category.setdefault(cat, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            dst["count"] += 1
            dst["total_ms"] += item["elapsed_ms"]
            dst["max_ms"] = max(dst["max_ms"], item["elapsed_ms"])
        if total_ms:
            top_level = by_category.get("decoder_layer", {}).get("total_ms", 0.0)
            top_level += by_category.get("lm_head_wrapper", {}).get("total_ms", 0.0)
            by_category["unattributed_top_level"] = {
                "count": 1,
                "total_ms": max(0.0, float(total_ms) - top_level),
                "max_ms": max(0.0, float(total_ms) - top_level),
            }
        for dst in by_category.values():
            dst["mean_ms"] = dst["total_ms"] / max(1, dst["count"])
            dst["share_of_forward"] = dst["total_ms"] / float(total_ms) if total_ms else None
        top_modules = sorted(finalized, key=lambda item: item["elapsed_ms"], reverse=True)[:limit]
        return {
            "enabled": True,
            "note": "Module timings are inclusive. decoder_layer and lm_head_wrapper are intended as top-level, non-overlapping categories; attention/mlp/linearmem timings are nested diagnostics and should not be added to decoder_layer.",
            "record_count": len(finalized),
            "by_category": by_category,
            "top_modules": top_modules,
        }


class LayerPeakTraceContext:
    def __init__(self, model, LinearMem, enabled=False):
        self.model = model
        self.LinearMem = LinearMem
        self.enabled = bool(enabled and LinearMem is not None and torch.cuda.is_available())
        self.handles = []
        self.rows = []

    def __enter__(self):
        if not self.enabled:
            return self
        for name, module in self.model.named_modules():
            if not isinstance(module, self.LinearMem):
                continue
            module_name = name or "<root>"
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(module_name)))
            self.handles.append(module.register_forward_hook(self._post_hook(module_name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def _device(module):
        engine = getattr(module, "engine", None)
        return torch.device(getattr(engine, "device", "cuda"))

    def _pre_hook(self, name):
        def hook(module, _inputs):
            device = self._device(module)
            torch.cuda.synchronize(device)
            allocated_before = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            object.__setattr__(module, "_memintelli_layer_peak_trace", allocated_before)

        return hook

    def _post_hook(self, name):
        def hook(module, _inputs, _output):
            device = self._device(module)
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
            allocated_after = int(torch.cuda.memory_allocated(device))
            allocated_before = int(getattr(module, "_memintelli_layer_peak_trace", 0) or 0)
            self.rows.append(
                {
                    "index": len(self.rows),
                    "name": name,
                    "in_features": int(getattr(module, "in_features", 0) or 0),
                    "out_features": int(getattr(module, "out_features", 0) or 0),
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "peak_bytes": peak,
                    "incremental_peak_bytes": max(0, peak - allocated_before),
                }
            )
            object.__setattr__(module, "_memintelli_layer_peak_trace", 0)

        return hook


class SemanticOutputProbe:
    def __init__(self, model, LinearMem, enabled=False):
        self.model = model
        self.LinearMem = LinearMem
        self.enabled = bool(enabled and LinearMem is not None)
        self.handles = []
        self.rows = []

    def __enter__(self):
        if not self.enabled:
            return self
        for name, module in self.model.named_modules():
            if not isinstance(module, self.LinearMem):
                continue
            module_name = name or "<root>"

            def hook(_module, _inputs, output, module_name=module_name):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(value):
                    return
                cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
                self.rows.append(
                    {
                        "index": len(self.rows),
                        "name": module_name,
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    }
                )

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_dtype(name):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


VMM_LOWP_CHOICES = ("auto", "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8")
VMM_LOWP_STABLE_DTYPES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}
VMM_LOWP_EXPERIMENTAL_8BIT = {"fp8_e4m3", "fp8_e5m2", "int8"}


def apply_vmm_lowp_format_defaults(args, *, argv_has_option):
    fmt = getattr(args, "vmm_lowp_format", "auto")
    if fmt in (None, "auto"):
        args.vmm_lowp_format = "auto"
        args.vmm_lowp_status = "auto_maps_to_compute_dtype"
        return args
    if fmt in VMM_LOWP_STABLE_DTYPES:
        dtype = VMM_LOWP_STABLE_DTYPES[fmt]
        if not argv_has_option("--conductance-dtype"):
            args.conductance_dtype = dtype
        if not argv_has_option("--compute-dtype"):
            args.compute_dtype = dtype
        if not argv_has_option("--mode0-vmm-compute-dtype"):
            args.mode0_vmm_compute_dtype = "auto"
        args.vmm_lowp_status = f"mapped_to_{dtype}"
        return args
    if fmt in VMM_LOWP_EXPERIMENTAL_8BIT:
        args.vmm_lowp_status = "unsupported_needs_scaled_8bit_kernel"
        return args
    raise ValueError(f"Unsupported --vmm-lowp-format: {fmt}")


def unsupported_8bit_vmm_reason(fmt):
    return (
        f"--vmm-lowp-format {fmt} is an experimental interface only. "
        "Current Memintelli Triton VMM kernels support FP32/FP16/BF16 DOT_DTYPE only; "
        "FP8/INT8 needs a scaled current-preserving kernel that de-scales before ADC. "
        "Use examples/21_probe_vmm_lowp_support.py for capability probing, or choose "
        "--vmm-lowp-format bf16/fp16/fp32 for runnable benchmarks."
    )


def make_engine(args, device):
    if args.kind == "v3":
        from memintelli.pimpy import DPETensorMultiMode

        return DPETensorMultiMode(
            HGS=args.hgs,
            LGS=args.lgs,
            g_level=args.g_level,
            write_variation=args.write_variation,
            read_variation=args.read_variation,
            vnoise=args.vnoise,
            write_variation_mode=args.write_variation_mode,
            conductance_dtype=args.conductance_dtype,
            compute_dtype=args.compute_dtype,
            linear_output_dtype=args.linear_output_dtype,
            mode0_semantic_policy=args.mode0_semantic_policy,
            mode0_vmm_compute_dtype=args.mode0_vmm_compute_dtype,
            rdac=args.rdac,
            radc=args.radc,
            vread=args.vread,
            rate_stuck_HGS=args.rate_stuck_hgs,
            rate_stuck_LGS=args.rate_stuck_lgs,
            mode=args.mode,
            mode2_input_mode=args.mode2_input_mode,
            inference_chunk_size=args.inference_chunk_size,
            fast_inference=args.fast_inference,
            fast_inference_backend=args.fast_inference_backend,
            triton_input_precision=args.triton_input_precision,
            triton_block_r=args.triton_block_r,
            triton_block_l=args.triton_block_l,
            triton_block_k=args.triton_block_k,
            triton_output_chunk_limit=args.triton_output_chunk_limit,
            triton_auto_config=args.triton_auto_config,
            triton_mode0_input_tile_group=args.triton_mode0_input_tile_group,
            triton_gidx_read_noise=args.triton_gidx_read_noise,
            triton_gidx_fused_restore_read_noise=args.triton_gidx_fused_restore_read_noise,
            triton_gidx_restore_block=args.triton_gidx_restore_block,
            triton_gidx_restore_block_auto=args.triton_gidx_restore_block_auto,
            triton_gidx_restore_small_block=args.triton_gidx_restore_small_block,
            triton_gidx_restore_auto_in_features_threshold=args.triton_gidx_restore_auto_in_features_threshold,
            triton_gidx_restore_num_warps=args.triton_gidx_restore_num_warps,
            triton_gidx_restore_strided=args.triton_gidx_restore_strided,
            triton_gidx_restore_m_slab=args.triton_gidx_restore_m_slab,
            triton_gidx_restore_approx_linear_noise=args.triton_gidx_restore_approx_linear_noise,
            triton_gidx_restore_exp2_noise=args.triton_gidx_restore_exp2_noise,
            triton_gidx_restore_fast_noise=args.triton_gidx_restore_fast_noise,
            triton_gidx_fuse_input_slices=args.triton_gidx_fuse_input_slices,
            triton_mode0_strict_intermediate=args.triton_mode0_strict_intermediate,
            triton_mode0_strict_intermediate_backend=args.triton_mode0_strict_intermediate_backend,
            triton_reuse_input_voltage=args.triton_reuse_input_voltage,
            triton_reuse_weight_tile=args.triton_reuse_weight_tile,
            triton_precompute_input_voltage=args.triton_precompute_input_voltage,
            triton_fast_adc_scale=args.triton_fast_adc_scale,
            triton_fuse_restored_input_slices=args.triton_fuse_restored_input_slices,
            triton_fuse_activation_slices=args.triton_fuse_activation_slices,
            triton_reuse_activation_slice_buffer=args.triton_reuse_activation_slice_buffer,
            triton_probe_activation_slice_reuse=args.triton_probe_activation_slice_reuse,
            triton_probe_activation_density=args.triton_probe_activation_density,
            triton_activation_slice_cache=args.triton_activation_slice_cache,
            triton_activation_slice_cache_max_entries=args.triton_activation_slice_cache_max_entries,
            triton_binary_input_slice_dac=args.triton_binary_input_slice_dac,
            triton_direct_final_num_warps=args.triton_direct_final_num_warps,
            triton_direct_final_partial_m_group=args.triton_direct_final_partial_m_group,
            triton_direct_final_exact_reduce=args.triton_direct_final_exact_reduce,
            triton_fuse_output_finalize=args.triton_fuse_output_finalize,
            triton_direct_final_output=args.triton_direct_final_output,
            triton_direct_output_zero_once=args.triton_direct_output_zero_once,
            triton_gidx_direct_final_output=args.triton_gidx_direct_final_output,
            triton_gidx_direct_final_deterministic=args.triton_gidx_direct_final_deterministic,
            triton_overlap_restore_direct=args.triton_overlap_restore_direct,
            triton_cross_linear_restore_prefetch=args.triton_cross_linear_restore_prefetch,
            triton_mode1_gidx_direct_final=args.triton_mode1_gidx_direct_final,
            triton_mode1_input_tile_group=args.triton_mode1_input_tile_group,
            triton_mode1_chunked_direct_final=args.triton_mode1_chunked_direct_final,
            triton_mode2_diff_direct_final=args.triton_mode2_diff_direct_final,
            triton_mode2_diff_presubtract=args.triton_mode2_diff_presubtract,
            triton_mode2_diff_fuse_input_slices=args.triton_mode2_diff_fuse_input_slices,
            triton_mode2_diff_block_r_cap=args.triton_mode2_diff_block_r_cap,
            triton_mode2_diff_block_l_cap=args.triton_mode2_diff_block_l_cap,
            mode1_grouped_tile_gemm=args.mode1_grouped_tile_gemm,
            direct_output_chunk_write=args.direct_output_chunk_write,
            read_variation_seed=args.read_variation_seed,
            profile=args.profile,
            profile_sync_cuda=args.profile_sync_cuda,
            runtime_stage_timing=bool(args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
            runtime_counters=bool(args.runtime_counters or args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
            device=device,
        )

    from memintelli.pimpy.memmat_tensor import DPETensor

    kwargs = dict(
        HGS=args.hgs,
        LGS=args.lgs,
        g_level=args.g_level,
        write_variation=args.write_variation,
        read_variation=args.read_variation,
        vnoise=args.vnoise,
        rdac=args.rdac,
        radc=args.radc,
        vread=args.vread,
        rate_stuck_HGS=args.rate_stuck_hgs,
        rate_stuck_LGS=args.rate_stuck_lgs,
        device=device,
    )
    if "inference_chunk_size" in inspect.signature(DPETensor).parameters:
        kwargs["inference_chunk_size"] = args.inference_chunk_size
    return DPETensor(**kwargs)


def linearmem_kwargs(LinearMem, args, engine, child, device, supports_skip):
    input_slice = args.mode2_input_slice if args.kind == "v3" and args.mode == 2 else args.input_slice
    weight_slice = args.mode2_weight_slice if args.kind == "v3" and args.mode == 2 else args.weight_slice
    kwargs = dict(
        engine=engine,
        in_features=child.in_features,
        out_features=child.out_features,
        input_slice=input_slice,
        weight_slice=weight_slice,
        bias=child.bias is not None,
        device=device,
        dtype=child.weight.dtype,
        bw_e=None,
        input_paral_size=tuple(args.input_paral_size),
        weight_paral_size=tuple(args.weight_paral_size),
        input_quant_gran=tuple(args.input_quant_gran),
        weight_quant_gran=tuple(args.weight_quant_gran),
    )
    if supports_skip:
        kwargs["skip_initial_mapping"] = True
    return kwargs


def plan_linear_output_blocks(args, child, manual_shard_count=0):
    stage = str(getattr(args, "s1_stage", "off") or "off")
    block_addressable = bool(getattr(args, "s1_block_addressable", stage != "off"))
    if not block_addressable:
        return argparse.Namespace(
            output_block_cols=int(child.out_features),
            shard_count=1,
            estimated_peak_mb=0.0,
            workspace_budget_mb=0.0,
            base_allocated_mb=0.0,
            resident_state_mb=0.0,
            safety_margin_mb=0.0,
            manual_override=False,
        )

    from memintelli.NN_layers.state_planner import plan_output_block

    planner_enabled = str(getattr(args, "state_planner", "off")) == "analytical"
    manual_cols = int(getattr(args, "output_block_cols", 0) or 0) if block_addressable else 0
    if block_addressable and not manual_cols and int(manual_shard_count or 0) > 1:
        manual_cols = math.ceil(int(child.out_features) / int(manual_shard_count))
    cuda_budget_mb = (
        float(getattr(args, "cuda_peak_budget_mb", 0.0) or 0.0)
        if block_addressable and planner_enabled
        else 0.0
    )
    resident_mb = max(0.0, float(getattr(args, "state_resident_budget_mb", 0.0) or 0.0))
    weight_tiles = tuple(int(v) for v in getattr(args, "weight_paral_size", (64, 64)))
    input_slices = getattr(args, "input_slice", (1,))
    weight_slices = getattr(args, "weight_slice", (1,))
    read_variation = float(getattr(args, "read_variation", 0.0) or 0.0)
    vmm_dtype = str(getattr(args, "mode0_vmm_compute_dtype", "auto") or "auto")
    if vmm_dtype == "auto":
        vmm_dtype = str(getattr(args, "compute_dtype", "float32") or "float32")
    grouped_noisy_vmm = read_variation > 0.0 and vmm_dtype in {"float16", "bfloat16"}
    return plan_output_block(
        tokens=max(1, int(getattr(args, "batch", 1))) * max(1, int(getattr(args, "seq", 1))),
        in_features=int(child.in_features),
        out_features=int(child.out_features),
        input_slices=max(1, len(input_slices)),
        weight_slices=max(1, len(weight_slices)),
        array_rows=weight_tiles[0],
        array_cols=weight_tiles[1],
        cuda_peak_budget_mb=cuda_budget_mb,
        base_allocated_mb=max(0.0, float(getattr(args, "planner_base_allocated_mb", 0.0) or 0.0)),
        resident_state_mb=resident_mb,
        safety_margin_mb=max(0.0, float(getattr(args, "planner_safety_margin_mb", 256.0) or 0.0)),
        manual_output_block_cols=manual_cols,
        read_variation=read_variation,
        seeded_read_noise=getattr(args, "read_variation_seed", None) is not None,
        grouped_noisy_vmm=grouped_noisy_vmm,
    )


class ShardedLinearMem(nn.Module):
    def __init__(self, LinearMem, args, engine, child, device, supports_skip, shard_count):
        super().__init__()
        if child.bias is not None:
            raise NotImplementedError("ShardedLinearMem currently supports bias=False only.")
        self.out_features = child.out_features
        self.in_features = child.in_features
        self.shard_count = int(shard_count)
        self.parallel = bool(args.lm_head_shard_parallel)
        self.shard_ranges = []
        shard_devices = parse_shard_devices(args.lm_head_shard_devices, engine.device)
        self.shard_devices = []
        self.shards = nn.ModuleList()
        rows_per_shard = math.ceil(child.out_features / self.shard_count)
        for start in range(0, child.out_features, rows_per_shard):
            end = min(start + rows_per_shard, child.out_features)
            shard_device = shard_devices[len(self.shards) % len(shard_devices)]
            shard_engine = engine if shard_device == engine.device else make_engine(args, shard_device)
            build_device = torch.device("cpu") if supports_skip else shard_device
            shard = LinearMem(**linearmem_kwargs(LinearMem, args, shard_engine, child, build_device, supports_skip))
            shard.out_features = end - start
            shard.weight = nn.Parameter(torch.empty((end - start, child.in_features), device=build_device, dtype=child.weight.dtype))
            shard.register_parameter("bias", None)
            self.shards.append(shard)
            self.shard_ranges.append((start, end))
            self.shard_devices.append(shard_device)

    @property
    def weight_sliced(self):
        return self.shards[0].weight_sliced

    @property
    def engine(self):
        return self.shards[0].engine

    def _run_shard(self, shard, shard_device, shard_range, input, return_device):
        if shard_device.type == "cuda":
            torch.cuda.set_device(shard_device)
        shard_input = input if input.device == shard_device else input.to(shard_device, non_blocking=True)
        out = shard(shard_input)
        if not torch.isfinite(out).all():
            raise RuntimeError(f"Non-finite lm_head shard output rows={shard_range} device={shard_device}")
        if out.device != return_device:
            out = out.to(return_device, non_blocking=True)
            if shard_device.type == "cuda":
                torch.cuda.synchronize(shard_device)
                torch.cuda.empty_cache()
            if return_device.type == "cuda":
                torch.cuda.synchronize(return_device)
        del shard_input
        return out

    def _run_shard_group(self, group, input, return_device):
        outs = []
        for index, shard, shard_device, shard_range in group:
            outs.append((index, self._run_shard(shard, shard_device, shard_range, input, return_device)))
        return outs

    def _forward_parallel(self, input, return_device):
        groups = {}
        for index, (shard, shard_device, shard_range) in enumerate(zip(self.shards, self.shard_devices, self.shard_ranges)):
            groups.setdefault(str(shard_device), []).append((index, shard, shard_device, shard_range))
        if len(groups) <= 1:
            return [self._run_shard(shard, shard_device, shard_range, input, return_device)
                    for shard, shard_device, shard_range in zip(self.shards, self.shard_devices, self.shard_ranges)]
        outs = [None] * len(self.shards)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = [
                executor.submit(self._run_shard_group, group, input, return_device)
                for group in groups.values()
            ]
            for future in concurrent.futures.as_completed(futures):
                for index, out in future.result():
                    outs[index] = out
        return outs

    def forward(self, input):
        return_device = input.device
        if self.parallel:
            outs = self._forward_parallel(input, return_device)
        else:
            outs = [self._run_shard(shard, shard_device, shard_range, input, return_device)
                    for shard, shard_device, shard_range in zip(self.shards, self.shard_devices, self.shard_ranges)]
        output = torch.cat(outs, dim=-1)
        if return_device.type == "cuda":
            torch.cuda.synchronize(return_device)
        return output

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True):
        for shard in self.shards:
            shard.enable_lazy_inference(streaming=streaming, free_weights=free_weights, release_after_forward=release_after_forward)
        return self


def build_output_blocked_linear(LinearMem, args, engine, child, supports_skip, plan):
    if GenericOutputBlockedLinearMem is None:
        raise ImportError("v3 output-block execution requires memintelli.NN_layers.output_blocking")
    if child.bias is not None and int(getattr(args, "mode", 0)) != 0:
        raise NotImplementedError("output-block execution with bias is currently supported for mode0 only")
    block_devices = [engine.device] * int(plan.shard_count)
    blocks = []
    block_ranges = []
    for block_index, start in enumerate(range(0, int(child.out_features), int(plan.output_block_cols))):
        end = min(start + int(plan.output_block_cols), int(child.out_features))
        block_device = torch.device(block_devices[block_index])
        block_engine = engine if block_device == engine.device else make_engine(args, block_device)
        build_device = torch.device("cpu") if supports_skip else block_device
        block_spec = argparse.Namespace(
            in_features=int(child.in_features),
            out_features=end - start,
            bias=child.bias,
            weight=child.weight,
        )
        block = LinearMem(
            **linearmem_kwargs(
                LinearMem,
                args,
                block_engine,
                block_spec,
                build_device,
                supports_skip,
            )
        )
        blocks.append(block)
        block_ranges.append((start, end))
    wrapped = GenericOutputBlockedLinearMem(
        blocks=blocks,
        block_ranges=block_ranges,
        in_features=child.in_features,
        out_features=child.out_features,
        block_devices=block_devices,
        parallel=False,
    )
    object.__setattr__(wrapped, "output_block_plan", plan)
    return wrapped


class LastTokenLinearMem(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.out_features = inner.out_features
        self.in_features = inner.in_features

    @property
    def weight_sliced(self):
        return self.inner.weight_sliced

    @property
    def engine(self):
        return self.inner.engine

    def forward(self, input):
        selected = input
        if input.dim() >= 3 and input.shape[-2] > 1:
            selected = input.narrow(-2, input.shape[-2] - 1, 1)
        return self.inner(selected)

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True):
        if hasattr(self.inner, "enable_lazy_inference"):
            self.inner.enable_lazy_inference(
                streaming=streaming,
                free_weights=free_weights,
                release_after_forward=release_after_forward,
            )
        return self


class SelectedOutputLinearMem(nn.Module):
    def __init__(self, inner, output_token_ids):
        super().__init__()
        self.inner = inner
        self.output_token_ids = [int(item) for item in output_token_ids]
        self.output_token_id_to_index = {int(token_id): idx for idx, token_id in enumerate(self.output_token_ids)}
        self.out_features = inner.out_features
        self.in_features = inner.in_features

    @property
    def weight_sliced(self):
        return self.inner.weight_sliced

    @property
    def engine(self):
        return self.inner.engine

    def forward(self, input):
        return self.inner(input)

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True):
        if hasattr(self.inner, "enable_lazy_inference"):
            self.inner.enable_lazy_inference(
                streaming=streaming,
                free_weights=free_weights,
                release_after_forward=release_after_forward,
            )
        return self


def unwrap_lm_head_wrapper(layer):
    while isinstance(layer, (LastTokenLinearMem, SelectedOutputLinearMem)):
        layer = layer.inner
    return layer


def is_linearmem_wrapper(module):
    is_generic_output_block = (
        GenericOutputBlockedLinearMem is not None
        and isinstance(module, GenericOutputBlockedLinearMem)
    )
    return is_generic_output_block or isinstance(
        module,
        (
            LastTokenLinearMem,
            SelectedOutputLinearMem,
            ShardedLinearMem,
            FusedGateUpLinearMem,
            FusedProjectionGroupLinearMem,
            GateProjectionView,
            UpProjectionView,
            FusedProjectionView,
        ),
    )


def disable_linearmem_for_torch_compile(model, LinearMem):
    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "disable"):
        return {"enabled": False, "reason": "torch.compiler.disable unavailable", "wrapped_modules": 0}
    if LinearMem is None:
        return {"enabled": False, "reason": "LinearMem unavailable", "wrapped_modules": 0}
    wrapped = 0
    seen = set()
    for module in model.modules():
        if not (isinstance(module, LinearMem) or is_linearmem_wrapper(module)):
            continue
        key = id(module)
        if key in seen:
            continue
        seen.add(key)
        forward = getattr(module, "forward", None)
        if forward is None:
            continue
        if getattr(forward, "_memintelli_compile_disabled", False):
            continue
        disabled_forward = torch.compiler.disable(forward)
        try:
            setattr(disabled_forward, "_memintelli_compile_disabled", True)
        except Exception:
            pass
        module.forward = disabled_forward
        wrapped += 1
    return {"enabled": True, "reason": "ok", "wrapped_modules": wrapped}


def copy_source_linear_to_mem(target, source):
    with torch.no_grad():
        if (
            GenericOutputBlockedLinearMem is not None
            and isinstance(target, GenericOutputBlockedLinearMem)
        ):
            for block, (start, end) in zip(target.blocks, target.block_ranges):
                block.weight.copy_(source.weight.detach()[start:end].to(block.weight.device))
                if source.bias is not None:
                    block.bias.copy_(source.bias.detach()[start:end].to(block.bias.device))
            return
        target.weight.copy_(source.weight.detach().to(target.weight.device))
        if source.bias is not None:
            target.bias.copy_(source.bias.detach().to(target.bias.device))


def build_mem_linear_for_source(LinearMem, args, engine, source, device, supports_skip):
    plan = plan_linear_output_blocks(args, source)
    if plan.shard_count > 1:
        target = build_output_blocked_linear(
            LinearMem,
            args,
            engine,
            source,
            supports_skip,
            plan,
        )
    else:
        target = LinearMem(
            **linearmem_kwargs(
                LinearMem,
                args,
                engine,
                source,
                device,
                supports_skip,
            )
        )
        object.__setattr__(target, "output_block_plan", plan)
    copy_source_linear_to_mem(target, source)
    return target


def build_coalesced_mem_linear(
    LinearMem,
    args,
    engine,
    projections,
    device,
    supports_skip,
    *,
    output_align=1,
):
    first = projections[0]
    align = max(1, int(output_align))
    out_splits = [int(layer.out_features) for layer in projections]
    padded_splits = [int(math.ceil(size / align) * align) for size in out_splits]
    out_offsets = []
    offset = 0
    for padded in padded_splits:
        out_offsets.append(offset)
        offset += padded

    fused_weight = first.weight.detach().new_zeros((offset, int(first.in_features)))
    fused_bias = first.bias.detach().new_zeros((offset,)) if first.bias is not None else None
    for layer, start, size in zip(projections, out_offsets, out_splits):
        fused_weight[start:start + size].copy_(layer.weight.detach())
        if fused_bias is not None:
            fused_bias[start:start + size].copy_(layer.bias.detach())
    source = argparse.Namespace(
        in_features=int(first.in_features),
        out_features=int(offset),
        weight=fused_weight,
        bias=fused_bias,
    )
    inner = build_mem_linear_for_source(
        LinearMem,
        args,
        engine,
        source,
        device,
        supports_skip,
    )
    return inner, out_splits, padded_splits, out_offsets


class FusedGateUpLinearMem(nn.Module):
    def __init__(self, LinearMem, args, engine, gate, up, device, supports_skip):
        super().__init__()
        if gate.in_features != up.in_features:
            raise ValueError("gate_proj and up_proj must have the same input width.")
        if gate.weight.dtype != up.weight.dtype:
            raise ValueError("gate_proj and up_proj must have the same dtype.")
        if (gate.bias is None) != (up.bias is None):
            raise ValueError("gate_proj and up_proj must either both have bias or both omit bias.")
        self.gate_out_features = int(gate.out_features)
        self.up_out_features = int(up.out_features)
        self.in_features = int(gate.in_features)
        self.out_features = self.gate_out_features + self.up_out_features
        self.inner, _, _, _ = build_coalesced_mem_linear(
            LinearMem,
            args,
            engine,
            [gate, up],
            device,
            supports_skip,
        )
        object.__setattr__(self, "_gate_bias", gate.bias)
        object.__setattr__(self, "_up_bias", up.bias)
        self._cache_key = None
        self._cache_gate = None
        self._cache_up = None

    @property
    def weight_sliced(self):
        return self.inner.weight_sliced

    @property
    def engine(self):
        return self.inner.engine

    def _input_key(self, input):
        return (id(input), tuple(input.shape), int(input.data_ptr()))

    def _clear_cache(self):
        self._cache_key = None
        self._cache_gate = None
        self._cache_up = None

    def _run_group(self, input):
        fused = self.inner(input)
        return torch.split(fused, [self.gate_out_features, self.up_out_features], dim=-1)

    def gate(self, input):
        key = self._input_key(input)
        if self._cache_key == key and self._cache_gate is not None:
            gate = self._cache_gate
            self._clear_cache()
            return gate
        self._clear_cache()
        gate, up = self._run_group(input)
        self._cache_key = key
        self._cache_up = up
        return gate

    def up(self, input):
        key = self._input_key(input)
        if self._cache_key == key and self._cache_up is not None:
            up = self._cache_up
            self._clear_cache()
            return up
        self._clear_cache()
        gate, up = self._run_group(input)
        self._cache_key = key
        self._cache_gate = gate
        return up

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True):
        if hasattr(self.inner, "enable_lazy_inference"):
            self.inner.enable_lazy_inference(
                streaming=streaming,
                free_weights=free_weights,
                release_after_forward=release_after_forward,
            )
        return self


class GateProjectionView(nn.Module):
    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.in_features = owner.in_features
        self.out_features = owner.gate_out_features
        self.bias = owner._gate_bias

    @property
    def weight_sliced(self):
        return self._owner.weight_sliced

    @property
    def engine(self):
        return self._owner.engine

    def forward(self, input):
        return self._owner.gate(input)


class UpProjectionView(nn.Module):
    def __init__(self, owner):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.in_features = owner.in_features
        self.out_features = owner.up_out_features
        self.bias = owner._up_bias

    @property
    def weight_sliced(self):
        return self._owner.weight_sliced

    @property
    def engine(self):
        return self._owner.engine

    def forward(self, input):
        return self._owner.up(input)


class FusedProjectionGroupLinearMem(nn.Module):
    def __init__(self, LinearMem, args, engine, projections, device, supports_skip, output_align=None):
        super().__init__()
        if not projections:
            raise ValueError("At least one projection is required for fusion.")
        first_name, first_layer = projections[0]
        first_bias = first_layer.bias is not None
        for name, layer in projections:
            if layer.in_features != first_layer.in_features:
                raise ValueError(f"{name} does not share the fused projection input width.")
            if layer.weight.dtype != first_layer.weight.dtype:
                raise ValueError(f"{name} does not share the fused projection dtype.")
            if (layer.bias is not None) != first_bias:
                raise ValueError("Fused projections must either all have bias or all omit bias.")
        self.names = [name for name, _ in projections]
        layers = [layer for _, layer in projections]
        self.in_features = int(first_layer.in_features)
        self.inner, self.out_splits, self.padded_splits, self.out_offsets = build_coalesced_mem_linear(
            LinearMem,
            args,
            engine,
            layers,
            device,
            supports_skip,
            output_align=output_align or 1,
        )
        self.out_features = int(sum(self.padded_splits))
        object.__setattr__(self, "_projection_biases", [layer.bias for layer in layers])
        self._cache_key = None
        self._cache_outputs = None

    @property
    def weight_sliced(self):
        return self.inner.weight_sliced

    @property
    def engine(self):
        return self.inner.engine

    def _input_key(self, input):
        return (id(input), tuple(input.shape), int(input.data_ptr()))

    def _clear_cache(self):
        self._cache_key = None
        self._cache_outputs = None

    def project(self, index, input):
        key = self._input_key(input)
        if self._cache_key != key or self._cache_outputs is None or self._cache_outputs[index] is None:
            self._clear_cache()
            self._cache_key = key
            fused = self.inner(input)
            self._cache_outputs = [
                fused.narrow(-1, start, size)
                for start, size in zip(self.out_offsets, self.out_splits)
            ]
        out = self._cache_outputs[index]
        self._cache_outputs[index] = None
        if all(item is None for item in self._cache_outputs):
            self._clear_cache()
        return out

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True):
        if hasattr(self.inner, "enable_lazy_inference"):
            self.inner.enable_lazy_inference(
                streaming=streaming,
                free_weights=free_weights,
                release_after_forward=release_after_forward,
            )
        return self


class FusedProjectionView(nn.Module):
    def __init__(self, owner, index):
        super().__init__()
        object.__setattr__(self, "_owner", owner)
        self.index = int(index)
        self.in_features = owner.in_features
        self.out_features = owner.out_splits[self.index]
        self.bias = owner._projection_biases[self.index]

    @property
    def weight_sliced(self):
        return self._owner.weight_sliced

    @property
    def engine(self):
        return self._owner.engine

    def forward(self, input):
        return self._owner.project(self.index, input)


def normalize_indexed_cuda_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(f"cuda:{index}")
    return device


def parse_shard_devices(value, default_device):
    if not value:
        return [normalize_indexed_cuda_device(default_device)]
    devices = [normalize_indexed_cuda_device(item.strip()) for item in value.split(",") if item.strip()]
    return devices or [normalize_indexed_cuda_device(default_device)]


def matches_any(patterns, value):
    return any(re.search(pattern, value) for pattern in patterns)


def is_lm_head_name(value):
    return value == "lm_head" or value.endswith(".lm_head")


def list_linear_module_names(module):
    return [name or "<root>" for name, child in module.named_modules() if isinstance(child, nn.Linear)]


def estimate_non_linear_model_bytes(model):
    seen = set()
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            continue
        for value in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            total += int(value.numel()) * int(value.element_size())
    return total


def build_linear_manifest(original_names, remaining_names, replaced, replaced_lm_head, args):
    original_lm_head_names = [name for name in original_names if is_lm_head_name(name)]
    remaining_lm_head_names = [name for name in remaining_names if is_lm_head_name(name)]
    original_count = len(original_names)
    remaining_count = len(remaining_names)
    replaced_count = int(replaced)
    coverage = 1.0 if original_count == 0 else replaced_count / float(original_count)
    valid = (
        args.kind == "hf"
        or (
            replaced_count == original_count
            and remaining_count == 0
            and (not original_lm_head_names or replaced_lm_head == len(original_lm_head_names))
        )
    )
    return {
        "require_all_linears": bool(args.require_all_linears),
        "full_model_deployment_valid": bool(valid),
        "original_linear_count": original_count,
        "original_lm_head_linear_count": len(original_lm_head_names),
        "replaced_linear_count": replaced_count,
        "replaced_lm_head_linear_count": int(replaced_lm_head),
        "remaining_linear_count": remaining_count,
        "remaining_lm_head_linear_count": len(remaining_lm_head_names),
        "replacement_coverage": coverage,
        "lm_head_simulated": bool(original_lm_head_names and replaced_lm_head == len(original_lm_head_names)),
        "fuse_mlp_gate_up": bool(getattr(args, "fuse_mlp_gate_up", False)),
        "fuse_common_input_projections": bool(getattr(args, "fuse_common_input_projections", False)),
        "original_linear_names_sample": original_names[: args.linear_name_limit],
        "remaining_linear_names_sample": remaining_names[: args.linear_name_limit],
        "original_lm_head_names": original_lm_head_names[: args.linear_name_limit],
        "remaining_lm_head_names": remaining_lm_head_names[: args.linear_name_limit],
    }


def validate_full_model_request(args, original_names):
    if not args.require_all_linears or args.kind == "hf":
        return
    lm_head_names = [name for name in original_names if is_lm_head_name(name)]
    problems = []
    if args.only_lm_head:
        problems.append("--only-lm-head is incompatible with --require-all-linears")
    if args.linear_include_regex:
        problems.append("--linear-include-regex is incompatible with --require-all-linears")
    if args.linear_exclude_regex:
        problems.append("--linear-exclude-regex is incompatible with --require-all-linears")
    if args.max_linears is not None:
        problems.append("--max-linears is incompatible with --require-all-linears")
    if args.max_non_lm_head_linears is not None:
        problems.append("--max-non-lm-head-linears is incompatible with --require-all-linears")
    if lm_head_names and not args.simulate_lm_head:
        problems.append("--simulate-lm-head is required by --require-all-linears when lm_head is an nn.Linear")
    if problems:
        raise RuntimeError("; ".join(problems))


def should_replace_linear(full_name, args):
    is_head = is_lm_head_name(full_name)
    if args.only_lm_head and not is_head:
        return False
    if (not args.simulate_lm_head) and is_head:
        return False
    if args.linear_include_regex and not (is_head and args.always_include_lm_head) and not matches_any(args.linear_include_regex, full_name):
        return False
    if args.linear_exclude_regex and matches_any(args.linear_exclude_regex, full_name):
        return False
    return True


def projection_group_preserves_weight_quant(args, linears):
    try:
        quant_col = int(args.weight_quant_gran[1])
        tile_col = int(args.weight_paral_size[1])
    except Exception:
        return False
    quant_col = max(1, math.ceil(max(1, quant_col) / max(1, tile_col)) * max(1, tile_col))
    offset = 0
    for layer in linears[:-1]:
        offset += int(layer.out_features)
        if offset % quant_col != 0:
            return False
    return True


def projection_group_output_alignment(args):
    try:
        quant_col = int(args.weight_quant_gran[1])
        tile_col = int(args.weight_paral_size[1])
    except Exception:
        return 1
    return max(1, math.ceil(max(1, quant_col) / max(1, tile_col)) * max(1, tile_col))


def can_fuse_common_input_projection_group(args, linears):
    if args.kind != "v3" or int(getattr(args, "mode", 0)) != 0:
        return False
    if not linears:
        return False
    if float(getattr(args, "write_variation", 0.0) or 0.0) != 0.0:
        return False
    if (
        float(getattr(args, "read_variation", 0.0) or 0.0) > 0.0
        and getattr(args, "read_variation_seed", None) is not None
    ):
        return False
    if tuple(int(v) for v in getattr(args, "weight_quant_gran", ())) != tuple(
        int(v) for v in getattr(args, "weight_paral_size", ())
    ):
        return False
    first = linears[0]
    return projection_group_preserves_weight_quant(args, linears) and all(
        int(layer.in_features) == int(first.in_features)
        and layer.weight.dtype == first.weight.dtype
        and (layer.bias is None) == (first.bias is None)
        for layer in linears
    )


def replace_linear(module, LinearMem, args, engine, runtime_device, prefix="", state=None):
    if state is None:
        state = {
            "replaced": 0,
            "replaced_names": [],
            "non_lm_head_replaced": 0,
            "lm_head_replaced": 0,
            "fused_gate_up_groups": 0,
            "fused_common_projection_groups": 0,
            "fused_common_projection_linears": 0,
            "output_blocked_linears": 0,
            "output_block_plans": [],
        }
    state.setdefault("output_blocked_linears", 0)
    state.setdefault("output_block_plans", [])
    supports_skip = "skip_initial_mapping" in inspect.signature(LinearMem).parameters
    skipped = 0
    if bool(getattr(args, "fuse_common_input_projections", False)):
        projection_groups = [
            ("_memintelli_fused_qkv", ("q_proj", "k_proj", "v_proj")),
            (
                "_memintelli_fused_linear_attn_qkv_z_b_a",
                ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
            ),
            ("_memintelli_fused_linear_attn_qkv_z", ("in_proj_qkv", "in_proj_z")),
        ]
        for fused_attr, names in projection_groups:
            if not all(hasattr(module, name) and isinstance(getattr(module, name), nn.Linear) for name in names):
                continue
            full_names = [f"{prefix}.{name}" if prefix else name for name in names]
            if not all(should_replace_linear(full_name, args) for full_name in full_names):
                continue
            linears = [getattr(module, name) for name in names]
            output_align = None
            can_fuse = can_fuse_common_input_projection_group(args, linears)
            if (
                not can_fuse
                and fused_attr == "_memintelli_fused_linear_attn_qkv_z_b_a"
                and args.kind == "v3"
                and int(getattr(args, "mode", 0)) == 0
                and float(getattr(args, "write_variation", 0.0) or 0.0) == 0.0
                and tuple(int(v) for v in getattr(args, "weight_quant_gran", ())) == tuple(
                    int(v) for v in getattr(args, "weight_paral_size", ())
                )
                and (
                    float(getattr(args, "read_variation", 0.0) or 0.0) <= 0.0
                    or getattr(args, "read_variation_seed", None) is None
                )
            ):
                output_align = projection_group_output_alignment(args)
                can_fuse = True
            if not can_fuse:
                continue
            extra_count = len(linears)
            if (
                args.max_linears is not None
                and args.max_linears >= 0
                and state["replaced"] + extra_count > args.max_linears
            ):
                continue
            if (
                args.max_non_lm_head_linears is not None
                and args.max_non_lm_head_linears >= 0
                and state["non_lm_head_replaced"] + extra_count > args.max_non_lm_head_linears
            ):
                continue
            build_device = torch.device("cpu") if supports_skip else runtime_device
            fused = FusedProjectionGroupLinearMem(
                LinearMem,
                args,
                engine,
                list(zip(names, linears)),
                build_device,
                supports_skip,
                output_align=output_align,
            )
            for index, name in enumerate(names):
                setattr(module, name, FusedProjectionView(fused, index))
            module.add_module(fused_attr, fused)
            state["replaced"] += extra_count
            state["non_lm_head_replaced"] += extra_count
            state["fused_common_projection_groups"] += 1
            state["fused_common_projection_linears"] += extra_count
            for fused_name in full_names:
                if len(state["replaced_names"]) < args.linear_name_limit:
                    state["replaced_names"].append(fused_name)
    if (
        bool(getattr(args, "fuse_mlp_gate_up", False))
        and hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and isinstance(module.gate_proj, nn.Linear)
        and isinstance(module.up_proj, nn.Linear)
        and can_fuse_common_input_projection_group(args, [module.gate_proj, module.up_proj])
    ):
        gate_name = f"{prefix}.gate_proj" if prefix else "gate_proj"
        up_name = f"{prefix}.up_proj" if prefix else "up_proj"
        if should_replace_linear(gate_name, args) and should_replace_linear(up_name, args):
            if (
                args.max_linears is None
                or args.max_linears < 0
                or state["replaced"] + 2 <= args.max_linears
            ) and (
                args.max_non_lm_head_linears is None
                or args.max_non_lm_head_linears < 0
                or state["non_lm_head_replaced"] + 2 <= args.max_non_lm_head_linears
            ):
                build_device = torch.device("cpu") if supports_skip else runtime_device
                fused = FusedGateUpLinearMem(
                    LinearMem,
                    args,
                    engine,
                    module.gate_proj,
                    module.up_proj,
                    build_device,
                    supports_skip,
                )
                module.gate_proj = GateProjectionView(fused)
                module.up_proj = UpProjectionView(fused)
                module.add_module("_memintelli_fused_gate_up", fused)
                state["replaced"] += 2
                state["non_lm_head_replaced"] += 2
                state["fused_gate_up_groups"] += 1
            for fused_name in (gate_name, up_name):
                if len(state["replaced_names"]) < args.linear_name_limit:
                    state["replaced_names"].append(fused_name)
    for name, child in list(module.named_children()):
        if name in {
            "_memintelli_fused_gate_up",
            "_memintelli_fused_qkv",
            "_memintelli_fused_linear_attn_qkv_z_b_a",
            "_memintelli_fused_linear_attn_qkv_z",
        }:
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if not should_replace_linear(full_name, args):
                skipped += 1
                continue
            is_head = is_lm_head_name(full_name)
            if args.max_linears is not None and args.max_linears >= 0 and state["replaced"] >= args.max_linears:
                skipped += 1
                continue
            if (
                (not is_head)
                and args.max_non_lm_head_linears is not None
                and args.max_non_lm_head_linears >= 0
                and state["non_lm_head_replaced"] >= args.max_non_lm_head_linears
            ):
                skipped += 1
                continue
            build_device = torch.device("cpu") if supports_skip else runtime_device
            lm_head_output_token_ids = list(getattr(args, "lm_head_output_token_ids", []) or []) if is_head else []
            if is_head and lm_head_output_token_ids and args.lm_head_output_shards > 1:
                raise ValueError("--lm-head-output-select label_tokens is incompatible with --lm-head-output-shards > 1")
            if is_head and lm_head_output_token_ids:
                if child.bias is not None:
                    raise NotImplementedError("Reduced lm_head output currently supports bias=False only.")
                reduced_child = nn.Linear(child.in_features, len(lm_head_output_token_ids), bias=False, device=build_device, dtype=child.weight.dtype)
                new_layer = LinearMem(**linearmem_kwargs(LinearMem, args, engine, reduced_child, build_device, supports_skip))
                new_layer = SelectedOutputLinearMem(new_layer, lm_head_output_token_ids)
            else:
                manual_shard_count = int(args.lm_head_output_shards) if is_head else 0
                output_block_plan = plan_linear_output_blocks(
                    args,
                    child,
                    manual_shard_count=manual_shard_count,
                )
                if output_block_plan.shard_count > 1:
                    new_layer = build_output_blocked_linear(
                        LinearMem,
                        args,
                        engine,
                        child,
                        supports_skip,
                        output_block_plan,
                    )
                    state["output_blocked_linears"] += 1
                    state["output_block_plans"].append(
                        {
                            "name": full_name,
                            "in_features": int(child.in_features),
                            "out_features": int(child.out_features),
                            "output_block_cols": int(output_block_plan.output_block_cols),
                            "block_count": int(output_block_plan.shard_count),
                            "estimated_peak_mb": float(output_block_plan.estimated_peak_mb),
                            "workspace_budget_mb": float(output_block_plan.workspace_budget_mb),
                            "base_allocated_mb": float(output_block_plan.base_allocated_mb),
                            "resident_state_mb": float(output_block_plan.resident_state_mb),
                            "safety_margin_mb": float(output_block_plan.safety_margin_mb),
                            "predicted_total_peak_mb": float(
                                output_block_plan.estimated_peak_mb
                                + output_block_plan.base_allocated_mb
                                + output_block_plan.resident_state_mb
                            ),
                            "manual_override": bool(output_block_plan.manual_override),
                        }
                    )
                else:
                    new_layer = LinearMem(**linearmem_kwargs(LinearMem, args, engine, child, build_device, supports_skip))
            if is_head and args.lm_head_input_select == "last":
                new_layer = LastTokenLinearMem(new_layer)
            with torch.no_grad():
                target_layer = unwrap_lm_head_wrapper(new_layer) if is_head else new_layer
                if (
                    GenericOutputBlockedLinearMem is not None
                    and isinstance(target_layer, GenericOutputBlockedLinearMem)
                ):
                    for block, (start, end) in zip(target_layer.blocks, target_layer.block_ranges):
                        block.weight.copy_(child.weight.detach()[start:end].to(block.weight.device))
                        if child.bias is not None:
                            block.bias.copy_(child.bias.detach()[start:end].to(block.bias.device))
                else:
                    if is_head and lm_head_output_token_ids:
                        selected_weight = child.weight.detach()[lm_head_output_token_ids]
                        target_layer.weight.copy_(selected_weight.to(target_layer.weight.device))
                    else:
                        target_layer.weight.copy_(child.weight.detach().to(target_layer.weight.device))
                    if child.bias is not None:
                        if is_head and lm_head_output_token_ids:
                            target_layer.bias.copy_(child.bias.detach()[lm_head_output_token_ids].to(target_layer.bias.device))
                        else:
                            target_layer.bias.copy_(child.bias.detach().to(target_layer.bias.device))
            setattr(module, name, new_layer)
            del child
            state["replaced"] += 1
            if not is_head:
                state["non_lm_head_replaced"] += 1
            else:
                state["lm_head_replaced"] += 1
            if len(state["replaced_names"]) < args.linear_name_limit:
                state["replaced_names"].append(full_name)
        else:
            _, s = replace_linear(child, LinearMem, args, engine, runtime_device, full_name, state)
            skipped += s
    return state["replaced"], skipped


def apply_execution_mode_defaults(args):
    # Map execution mode to state placement without changing compute policy.
    memory_prepare_policy = getattr(args, "memory_prepare_policy", "lazy_release")
    memory_budget_mb = float(getattr(args, "memory_budget_mb", 0.0) or 0.0)
    resident_budget_mb = float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0)
    if memory_budget_mb > 0.0:
        if resident_budget_mb > 0.0 and abs(resident_budget_mb - memory_budget_mb) > 1e-6:
            raise ValueError("--memory-budget-mb and --memory-resident-budget-mb must match when both are set.")
        args.memory_resident_budget_mb = memory_budget_mb
    args.memory_budget_mb = float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0)
    if not hasattr(args, "state_resident_budget_mb"):
        args.state_resident_budget_mb = resident_budget_mb
    if args.execution_mode == "speed":
        args.state_resident_budget_mb = -1.0
        args.streaming = False
        args.lazy_prepare = False
        args.lazy_release_after_forward = False
    elif args.execution_mode == "balanced":
        args.state_resident_budget_mb = float(
            getattr(args, "state_resident_budget_mb", 0.0) or resident_budget_mb
        )
        args.streaming = True
        if memory_prepare_policy == "lazy_release":
            args.memory_prepare_policy = "eager_streaming"
            memory_prepare_policy = args.memory_prepare_policy
        args.lazy_prepare = False
        args.lazy_release_after_forward = False
        args.free_weights = True
    elif args.execution_mode == "memory":
        args.state_resident_budget_mb = 0.0
        args.streaming = True
        if memory_prepare_policy == "eager_streaming":
            args.lazy_prepare = False
            args.lazy_release_after_forward = False
            args.free_weights = True
        elif memory_prepare_policy == "lazy_keep":
            args.lazy_prepare = True
            args.lazy_release_after_forward = False
            args.free_weights = False
        else:
            args.lazy_prepare = True
            args.lazy_release_after_forward = True
            args.free_weights = False
        if getattr(args, "memory_runtime_diagnostic", False):
            args.lazy_prepare = False
    return args


def apply_worker_s1_stage(args):
    stage = getattr(args, "s1_stage", "budgeted")
    if stage not in {"off", "block", "budgeted"}:
        raise ValueError(f"unsupported S1 stage: {stage}")
    args.s1_block_addressable = stage in {"block", "budgeted"}
    if stage == "off":
        args.state_planner = "off"
        args.state_resident_budget_mb = -1.0
    elif stage == "block":
        args.state_planner = "off"
        args.state_resident_budget_mb = 0.0
    else:
        args.state_planner = "analytical"
    if float(getattr(args, "state_resident_budget_mb", -1.0) or 0.0) >= 0.0:
        args.memory_resident_budget_mb = float(args.state_resident_budget_mb)
    return args


def apply_worker_s2_stage(args):
    stage = getattr(args, "s2_stage", "full")
    if stage not in {"off", "intra", "full"}:
        raise ValueError(f"unsupported S2 stage: {stage}")
    fast_inference = stage != "off"
    args.fast_inference = fast_inference
    args.triton_fuse_restored_input_slices = fast_inference
    args.triton_direct_final_output = fast_inference
    args.triton_gidx_direct_final_output = False
    args.triton_direct_final_exact_reduce = fast_inference
    args.triton_overlap_restore_direct = False
    args.triton_precompute_input_voltage = False
    args.triton_fast_adc_scale = False
    args.triton_direct_output_zero_once = False
    args.fuse_mlp_gate_up = stage == "full"
    args.fuse_common_input_projections = stage == "full"
    args.triton_activation_slice_cache = False
    if args.fast_inference_backend == "auto":
        args.fast_inference_backend = "triton_gidx" if fast_inference else "torch"
    return args


def _worker_argv_has_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def configure_cuda_memory_fraction(args, device):
    fraction = float(getattr(args, "cuda_memory_fraction", 0.0) or 0.0)
    if device.type != "cuda" or fraction <= 0.0:
        return {"enabled": False, "fraction": fraction}
    if fraction > 1.0:
        raise ValueError("--cuda-memory-fraction must be in the range (0, 1].")

    if device.index is not None:
        torch.cuda.set_device(device)
        target_device = device
    else:
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_per_process_memory_fraction(fraction, target_device)
    props = torch.cuda.get_device_properties(target_device)
    total_mb = props.total_memory / (1024 ** 2)
    return {
        "enabled": True,
        "fraction": fraction,
        "device": str(target_device),
        "gpu_name": props.name,
        "gpu_total_mb": total_mb,
        "allocator_limit_mb": total_mb * fraction,
    }


def iter_mem_layers(model, LinearMem):
    if LinearMem is None:
        return
    for module in model.modules():
        if isinstance(module, LinearMem):
            yield module


def iter_named_mem_layers(model, LinearMem):
    if LinearMem is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, LinearMem):
            yield name or "<root>", module


def load_streaming_pin_hints(path):
    if not path:
        return {}, "disabled"
    p = Path(path)
    if not p.exists():
        return {}, f"missing:{path}"
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"load_failed:{exc}"

    containers = []
    if isinstance(obj, list):
        containers.extend(item for item in obj if isinstance(item, dict))
    elif isinstance(obj, dict):
        containers.append(obj)
        for key in ("rows", "records", "cells"):
            value = obj.get(key)
            if isinstance(value, list):
                containers.extend(item for item in value if isinstance(item, dict))

    hints = {}
    for item in containers:
        mech = item.get("runtime_mechanism_counters") if isinstance(item.get("runtime_mechanism_counters"), dict) else item
        layer_rows = mech.get("layer_rows") if isinstance(mech, dict) else None
        if not isinstance(layer_rows, list):
            continue
        for layer in layer_rows:
            if not isinstance(layer, dict):
                continue
            name = str(layer.get("name") or "")
            if not name:
                continue
            stream_bytes = float(layer.get("stream_weight_bytes") or layer.get("pinned_buffer_peak_bytes") or 0.0)
            pin_ms = float(layer.get("pin_cpu_buffer_ms") or 0.0)
            sync_ms = float(layer.get("sync_load_ms") or 0.0)
            prefetch_ms = float(layer.get("prefetch_schedule_ms") or 0.0)
            lazy_prepare_ms = float(layer.get("lazy_prepare_ms") or 0.0)
            release_prepared_ms = float(layer.get("release_prepared_ms") or 0.0)
            pending = float(layer.get("prefetch_pending_on_arrival_count") or 0.0)
            benefit_ms = pin_ms + sync_ms + prefetch_ms + lazy_prepare_ms + release_prepared_ms
            if pending > 0:
                benefit_ms += 0.05 * pending
            hints[name] = {
                "stream_weight_bytes": stream_bytes,
                "benefit_ms": benefit_ms,
                "pin_cpu_buffer_ms": pin_ms,
                "sync_load_ms": sync_ms,
                "prefetch_schedule_ms": prefetch_ms,
                "lazy_prepare_ms": lazy_prepare_ms,
                "release_prepared_ms": release_prepared_ms,
                "prefetch_pending_on_arrival_count": pending,
                "score_per_byte": benefit_ms / max(stream_bytes, 1.0),
            }
    return hints, f"loaded:{len(hints)}:{path}"


def collect_engine_profile(model, LinearMem):
    summaries = []
    profile_layers = 0
    seen_engines = set()
    for module in iter_mem_layers(model, LinearMem):
        engine = getattr(module, "engine", None)
        if engine is None or not getattr(engine, "profile", False):
            continue
        engine_id = id(engine)
        if engine_id in seen_engines:
            profile_layers += 1
            continue
        seen_engines.add(engine_id)
        summaries.append(engine.get_profile_summary())
        profile_layers += 1
    return profile_layers, merge_profile_summaries(summaries)


def reset_engine_profiles(model, LinearMem):
    seen_engines = set()
    for module in iter_mem_layers(model, LinearMem):
        engine = getattr(module, "engine", None)
        if engine is not None and getattr(engine, "profile", False):
            engine_id = id(engine)
            if engine_id in seen_engines:
                continue
            seen_engines.add(engine_id)
            engine.reset_profile()


def reset_engine_fastpath_counters(model, LinearMem):
    seen_engines = set()
    for module in iter_mem_layers(model, LinearMem):
        engine = getattr(module, "engine", None)
        if engine is None or not hasattr(engine, "reset_fastpath_counters"):
            continue
        engine_id = id(engine)
        if engine_id in seen_engines:
            continue
        seen_engines.add(engine_id)
        engine.reset_fastpath_counters()


def configure_cross_restore_prefetch_from_execution_trace(model, LinearMem, trace):
    if LinearMem is None:
        return {"enabled": False, "trace_length": 0, "unique_layers": 0, "links": 0}
    if not trace:
        return {"enabled": False, "trace_length": 0, "unique_layers": 0, "links": 0}
    mem_layers = {id(module): module for module in iter_mem_layers(model, LinearMem)}
    unique_layers = []
    seen = set()
    for module in trace:
        if id(module) not in mem_layers:
            continue
        if id(module) in seen:
            continue
        unique_layers.append(module)
        seen.add(id(module))
    for module in mem_layers.values():
        object.__setattr__(module, "_next_restore_prefetch_layer", None)
    links = 0
    for current, target in zip(unique_layers, unique_layers[1:]):
        object.__setattr__(current, "_next_restore_prefetch_layer", target)
        links += 1
    return {
        "enabled": True,
        "trace_length": len(trace),
        "unique_layers": len(unique_layers),
        "links": links,
    }


def move_non_mem_tensors_to_device(model, LinearMem, device):
    # Keep LinearMem weights on CPU for lazy-release while moving the rest.
    for module in model.modules():
        if LinearMem is not None and isinstance(module, LinearMem):
            continue
        for param in module.parameters(recurse=False):
            if param is not None and param.device != device:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is not None and buf.device != device:
                setattr(module, name, buf.to(device))
    return model


def keep_linear_weights_on_cpu(args):
    return bool(args.lazy_prepare and args.lazy_release_after_forward)


def module_weight_cost_hint(module) -> int:
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return int(weight.numel())
    return 0


def estimate_stream_weight_bytes(module, args) -> int:
    # Estimate compressed mapped-state bytes before materializing a layer.
    weight_numel = module_weight_cost_hint(module)
    if weight_numel <= 0:
        return 0
    try:
        weight_slice_count = max(1, len(getattr(module, "weight_slice_method", args.weight_slice)))
    except TypeError:
        weight_slice_count = max(1, len(args.weight_slice))
    mode = int(getattr(getattr(module, "engine", None), "mode", args.mode if args.kind == "v3" else 0) or 0)
    branch_multiplier = 2 if mode == 2 else 1
    # In the optimized SLC path, the dominant resident state is compressed
    # uint8 conductance level indices. Scale tensors are much smaller; add a
    # small margin so budget selection does not overfill on rounded tiles.
    return int(weight_numel * weight_slice_count * branch_multiplier * 1.08)


def select_budgeted_resident_names(named_layers, budget_bytes, select, pin_hints, args):
    if budget_bytes <= 0:
        return set(), 0
    selected = set()
    used = 0
    for name, module in lazy_resident_candidate_order(named_layers, select, pin_hints):
        layer_bytes = module_stream_weight_bytes_from_hints(name, pin_hints)
        if layer_bytes <= 0:
            layer_bytes = estimate_stream_weight_bytes(module, args)
        if layer_bytes <= 0:
            continue
        if used + layer_bytes <= budget_bytes:
            selected.add(name)
            used += layer_bytes
    return selected, used


def lazy_resident_candidate_order(named_layers, resident_select, pin_hints):
    if resident_select == "largest":
        return sorted(named_layers, key=lambda item: module_weight_cost_hint(item[1]), reverse=True)
    if resident_select == "runtime" and pin_hints:
        scored = []
        for name, module in named_layers:
            hint = pin_hints.get(name) or {}
            benefit = float(hint.get("benefit_ms") or 0.0)
            cost = int(module_weight_cost_hint(module))
            hint_bytes = float(hint.get("stream_weight_bytes") or 0.0)
            score_denominator = hint_bytes if hint_bytes > 0.0 else float(max(cost, 1))
            score = benefit / score_denominator
            scored.append((score, benefit, cost, name, module))
        return [(name, module) for _score, _benefit, _cost, name, module in sorted(scored, reverse=True)]
    return list(named_layers)


def set_lazy_module_common(module, args, module_device, *, release_after_forward):
    object.__setattr__(
        module,
        "_empty_cache_after_release",
        bool(getattr(args, "memory_empty_cache_after_offload", False)),
    )
    object.__setattr__(
        module,
        "_empty_cache_after_release_interval",
        max(1, int(getattr(args, "memory_empty_cache_after_offload_interval", 1) or 1)),
    )
    module.enable_lazy_inference(
        streaming=args.streaming if release_after_forward else False,
        free_weights=args.free_weights if release_after_forward else False,
        release_after_forward=release_after_forward,
        pin_policy=args.streaming_pin_policy,
    )
    if module.bias is not None and module.bias.device != module_device:
        module.bias.data = module.bias.data.to(module_device)
    if hasattr(module, "input_slice_method") and module.input_slice_method.device != module_device:
        module.input_slice_method = module.input_slice_method.to(module_device)
    if hasattr(module, "weight_slice_method") and module.weight_slice_method.device != module_device:
        module.weight_slice_method = module.weight_slice_method.to(module_device)


def offload_module_to_cpu(module, *, pin_policy):
    method = module._offload_to_cpu
    try:
        supports_pin_policy = "pin_policy" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        supports_pin_policy = False
    if supports_pin_policy:
        return method(pin_policy=pin_policy)
    return method()


def prepare_mem_model(model, LinearMem, args, device):
    cache_budget_mb = float(getattr(args, "streaming_window_pin_cache_mb", 0.0) or 0.0)
    cache_budget_bytes = int(cache_budget_mb * (1024 ** 2)) if cache_budget_mb > 0.0 else 0
    if hasattr(LinearMem, "configure_window_pin_cache"):
        LinearMem.configure_window_pin_cache(cache_budget_bytes)
    named_layers = list(iter_named_mem_layers(model, LinearMem))
    layers = [module for _name, module in named_layers]
    supports_inference = False
    lazy_layers = 0
    pin_budget_mb = float(getattr(args, "streaming_persistent_pin_budget_mb", 0.0) or 0.0)
    pin_budget_bytes = int(pin_budget_mb * (1024 ** 2)) if pin_budget_mb > 0.0 else 0
    pin_budget_used_bytes = 0
    pin_select = str(getattr(args, "streaming_persistent_pin_select", "sequential") or "sequential")
    pin_hints, pin_hints_status = load_streaming_pin_hints(getattr(args, "streaming_pin_hints_json", ""))
    resident_budget_mb = float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0)
    resident_budget_bytes = int(resident_budget_mb * (1024 ** 2)) if resident_budget_mb > 0.0 else 0
    resident_select = str(getattr(args, "memory_resident_select", "largest") or "largest")
    resident_budget_used_bytes = 0
    resident_budget_estimated_used_bytes = 0
    resident_names = set()
    resident_errors = []
    runtime_pin_names = set()
    window_cache_names = None
    window_cache_budget_used_bytes = 0
    if pin_select == "runtime" and pin_budget_bytes > 0 and pin_hints:
        scored = []
        for name, _module in named_layers:
            hint = pin_hints.get(name)
            if not hint:
                continue
            layer_bytes = module_stream_weight_bytes_from_hints(name, pin_hints)
            if layer_bytes <= 0:
                continue
            score = float(hint.get("score_per_byte") or 0.0)
            benefit = float(hint.get("benefit_ms") or 0.0)
            scored.append((score, benefit, -layer_bytes, name, layer_bytes))
        for _score, _benefit, _neg_bytes, name, layer_bytes in sorted(scored, reverse=True):
            if pin_budget_used_bytes + layer_bytes <= pin_budget_bytes:
                runtime_pin_names.add(name)
                pin_budget_used_bytes += layer_bytes
    if args.lazy_prepare:
        if args.free_weights and args.lazy_release_after_forward:
            raise ValueError("--lazy-release-after-forward requires --no-free-weights")
        if resident_budget_bytes > 0 and args.lazy_release_after_forward:
            selected_resident_names, resident_budget_estimated_used_bytes = select_budgeted_resident_names(
                named_layers,
                resident_budget_bytes,
                resident_select,
                pin_hints,
                args,
            )
            for module_name, module in named_layers:
                if module_name not in selected_resident_names:
                    continue
                module_device = getattr(module.engine, "device", device)
                set_lazy_module_common(module, args, module_device, release_after_forward=False)
                try:
                    module._prepare_inference_weight(streaming=False, free_weights=False, pin_policy=args.streaming_pin_policy)
                    layer_bytes = module_stream_weight_bytes(module)
                    accounted_bytes = layer_bytes or estimate_stream_weight_bytes(module, args)
                    if accounted_bytes and resident_budget_used_bytes + accounted_bytes <= resident_budget_bytes:
                        resident_names.add(module_name)
                        resident_budget_used_bytes += accounted_bytes
                    else:
                        module.release_prepared_weight()
                        resident_errors.append(
                            f"{module_name}:actual_state_exceeds_analytical_budget:{accounted_bytes}"
                        )
                except torch.cuda.OutOfMemoryError as exc:
                    resident_errors.append(f"{module_name}:oom:{exc}")
                    try:
                        module.release_prepared_weight()
                    except Exception:
                        pass
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:
                    resident_errors.append(f"{module_name}:{type(exc).__name__}:{exc}")
                    try:
                        module.release_prepared_weight()
                    except Exception:
                        pass
        window_cache_candidates = [(name, module) for name, module in named_layers if name not in resident_names]
        window_cache_names, window_cache_budget_used_bytes = select_budgeted_window_pin_cache_names(
            window_cache_candidates,
            cache_budget_bytes,
            pin_hints,
            args,
        )

        for module_name, module in named_layers:
            supports_inference = True
            module_device = getattr(module.engine, "device", device)
            if window_cache_names is not None:
                object.__setattr__(module, "_window_pin_cache_allowed", module_name in window_cache_names)
            if module_name in resident_names:
                # Already prepared above and intentionally kept resident on the GPU.
                continue
            set_lazy_module_common(
                module,
                args,
                module_device,
                release_after_forward=args.lazy_release_after_forward,
            )
            lazy_layers += 1

        g_bytes = 0
        g_gpu_bytes = 0
        g_cpu_bytes = 0
        pinned_bytes = 0
        for module in layers:
            ws = module.weight_sliced
            for attr in STREAM_WEIGHT_ATTRS:
                value = getattr(ws, attr, None)
                size = tensor_bytes(value)
                g_bytes += size
                by_device = tensor_device_bytes(value)
                g_gpu_bytes += by_device.get("cuda", 0)
                g_cpu_bytes += by_device.get("cpu", 0)
            for value in getattr(module, "_pinned_buffers", {}).values():
                pinned_bytes += tensor_pinned_bytes(value)
            for value in getattr(module, "_active_pinned_buffers", {}).values():
                pinned_bytes += tensor_pinned_bytes(value)
        return {
            "prepared_layers": len(resident_names),
            "lazy_layers": lazy_layers,
            "streaming_layers": lazy_layers if args.streaming else 0,
            "cross_restore_prefetch_layers": 0,
            "supports_inference": supports_inference,
            "g_storage_mb": g_bytes / (1024 ** 2),
            "g_gpu_mb": g_gpu_bytes / (1024 ** 2),
            "g_cpu_mb": g_cpu_bytes / (1024 ** 2),
            "g_cpu_pinned_mb": pinned_bytes / (1024 ** 2),
            "streaming_persistent_pin_budget_mb": pin_budget_mb,
            "streaming_persistent_pin_used_mb": 0.0,
            "streaming_persistent_pin_layers": 0,
            "streaming_window_pin_layers": 0,
            "streaming_window_pin_cache_mb": cache_budget_mb,
            "streaming_window_pin_cache_selected_layers": len(window_cache_names or []),
            "streaming_window_pin_cache_estimated_used_mb": window_cache_budget_used_bytes / (1024 ** 2),
            "streaming_pin_hints_status": pin_hints_status,
            "memory_resident_budget_mb": resident_budget_mb,
            "memory_resident_select": resident_select,
            "memory_resident_used_mb": resident_budget_used_bytes / (1024 ** 2),
            "memory_resident_estimated_used_mb": resident_budget_estimated_used_bytes / (1024 ** 2),
            "memory_resident_layers": len(resident_names),
            "memory_lazy_release_layers": lazy_layers,
            "memory_resident_error_count": len(resident_errors),
            "memory_resident_errors": resident_errors[:8],
        }

    largest_pin_candidates = []
    if (not args.lazy_prepare) and args.streaming and resident_budget_bytes > 0:
        resident_names, resident_budget_estimated_used_bytes = select_budgeted_resident_names(
            named_layers,
            resident_budget_bytes,
            resident_select,
            pin_hints,
            args,
        )
    if not args.lazy_prepare:
        window_cache_candidates = [(name, module) for name, module in named_layers if name not in resident_names]
        window_cache_names, window_cache_budget_used_bytes = select_budgeted_window_pin_cache_names(
            window_cache_candidates,
            cache_budget_bytes,
            pin_hints,
            args,
        )

    for module_index, (module_name, module) in enumerate(named_layers):
        module_device = getattr(module.engine, "device", device)
        if window_cache_names is not None:
            object.__setattr__(module, "_window_pin_cache_allowed", module_name in window_cache_names)
        object.__setattr__(
            module,
            "_empty_cache_after_release",
            bool(getattr(args, "memory_empty_cache_after_offload", False)),
        )
        object.__setattr__(
            module,
            "_empty_cache_after_release_interval",
            max(1, int(getattr(args, "memory_empty_cache_after_offload_interval", 1) or 1)),
        )
        if hasattr(module, "inference_mode"):
            supports_inference = True
            module.inference_mode = True
        if supports_inference and hasattr(module, "weight_sliced"):
            module.weight_sliced.inference = True
        elif hasattr(module, "weight_sliced"):
            module.weight_sliced.inference = False
        module.update_weight()
        engine = module.engine
        if getattr(engine, "write_variation", 0) == 0 and hasattr(module.weight_sliced, "compress_G"):
            module.weight_sliced.compress_G(engine)
        if module.bias is not None and module.bias.device != module_device:
            module.bias.data = module.bias.data.to(module_device)
        if hasattr(module, "input_slice_method") and module.input_slice_method.device != module_device:
            module.input_slice_method = module.input_slice_method.to(module_device)
        if hasattr(module, "weight_slice_method") and module.weight_slice_method.device != module_device:
            module.weight_slice_method = module.weight_slice_method.to(module_device)
        if supports_inference and hasattr(module.weight_sliced, "quantized_data"):
            module.weight_sliced.quantized_data = None
        if supports_inference and hasattr(module.weight_sliced, "sliced_data"):
            module.weight_sliced.sliced_data = None
        if supports_inference and args.free_weights:
            module.weight.data = torch.empty(0, device="cpu", dtype=module.weight.dtype)
        if supports_inference and args.streaming:
            pin_policy = args.streaming_pin_policy
            keep_gpu_resident = module_name in resident_names
            layer_stream_bytes = module_stream_weight_bytes(module)
            if keep_gpu_resident and layer_stream_bytes and resident_budget_used_bytes + layer_stream_bytes <= resident_budget_bytes:
                resident_budget_used_bytes += layer_stream_bytes
                object.__setattr__(module, "_streaming", False)
                object.__setattr__(module, "_next_streaming_layer", None)
            else:
                if keep_gpu_resident:
                    resident_names.discard(module_name)
                    keep_gpu_resident = False
                if args.streaming_pin_policy == "persistent" and pin_budget_bytes > 0:
                    if pin_select in {"largest", "runtime"}:
                        pin_policy = "window"
                    else:
                        layer_bytes = layer_stream_bytes
                        if layer_bytes and pin_budget_used_bytes + layer_bytes <= pin_budget_bytes:
                            pin_policy = "persistent"
                            pin_budget_used_bytes += layer_bytes
                        else:
                            pin_policy = "window"
                offload_module_to_cpu(module, pin_policy=pin_policy)
                object.__setattr__(module, "_streaming", True)
            if args.streaming_pin_policy == "persistent" and pin_budget_bytes > 0 and pin_select == "largest" and not keep_gpu_resident:
                largest_pin_candidates.append((layer_stream_bytes, module_index, module))
            if bool(getattr(args, "memory_empty_cache_after_offload", False)) and device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    if args.streaming and args.streaming_pin_policy == "persistent" and pin_budget_bytes > 0 and pin_select == "largest":
        pin_budget_used_bytes = 0
        for layer_bytes, _module_index, module in sorted(largest_pin_candidates, reverse=True):
            if layer_bytes and pin_budget_used_bytes + layer_bytes <= pin_budget_bytes:
                offload_module_to_cpu(module, pin_policy="persistent")
                pin_budget_used_bytes += layer_bytes
    elif args.streaming and args.streaming_pin_policy == "persistent" and pin_budget_bytes > 0 and pin_select == "runtime":
        for module_name, module in named_layers:
            if module_name in runtime_pin_names and module_name not in resident_names and bool(getattr(module, "_streaming", False)):
                offload_module_to_cpu(module, pin_policy="persistent")

    streaming_layers = 0
    if supports_inference and args.streaming:
        streaming_sequence = [module for module in layers if bool(getattr(module, "_streaming", False))]
        streaming_layers = len(streaming_sequence)
        if streaming_layers and bool(getattr(args, "streaming_prefetch", True)):
            prefetch_distance = max(1, int(getattr(args, "streaming_prefetch_distance", 1) or 1))
            for i in range(streaming_layers):
                target = i + prefetch_distance
                if target < streaming_layers:
                    object.__setattr__(streaming_sequence[i], "_next_streaming_layer", streaming_sequence[target])
                else:
                    object.__setattr__(streaming_sequence[i], "_next_streaming_layer", None)
            cycle_prefetch = bool(getattr(args, "streaming_prefetch_cycle", False))
            if cycle_prefetch:
                for i in range(max(0, streaming_layers - prefetch_distance), streaming_layers):
                    object.__setattr__(
                        streaming_sequence[i],
                        "_next_streaming_layer",
                        streaming_sequence[(i + prefetch_distance) % streaming_layers],
                    )
        else:
            for module in layers:
                object.__setattr__(module, "_next_streaming_layer", None)
    else:
        for module in layers:
            object.__setattr__(module, "_next_streaming_layer", None)

    cross_restore_layers = 0
    if (
        supports_inference
        and bool(getattr(args, "triton_cross_linear_restore_prefetch", False))
        and not bool(getattr(args, "streaming", False))
    ):
        for i, module in enumerate(layers):
            target = layers[i + 1] if i + 1 < len(layers) else None
            object.__setattr__(module, "_next_restore_prefetch_layer", target)
            if target is not None:
                cross_restore_layers += 1
    else:
        for module in layers:
            object.__setattr__(module, "_next_restore_prefetch_layer", None)

    g_bytes = 0
    g_gpu_bytes = 0
    g_cpu_bytes = 0
    pinned_bytes = 0
    persistent_pin_layers = 0
    window_pin_layers = 0
    for module in layers:
        ws = module.weight_sliced
        policy = getattr(module, "_streaming_pin_policy", None)
        is_streaming_layer = bool(getattr(module, "_streaming", False))
        if args.streaming and is_streaming_layer and policy == "window":
            window_pin_layers += 1
        elif args.streaming and is_streaming_layer and policy == "persistent":
            persistent_pin_layers += 1
        for attr in STREAM_WEIGHT_ATTRS:
            value = getattr(ws, attr, None)
            size = tensor_bytes(value)
            g_bytes += size
            by_device = tensor_device_bytes(value)
            g_gpu_bytes += by_device.get("cuda", 0)
            g_cpu_bytes += by_device.get("cpu", 0)
        for value in getattr(module, "_pinned_buffers", {}).values():
            pinned_bytes += tensor_pinned_bytes(value)
        for value in getattr(module, "_active_pinned_buffers", {}).values():
            pinned_bytes += tensor_pinned_bytes(value)
    return {
        "prepared_layers": len(layers),
        "lazy_layers": lazy_layers,
        "streaming_layers": streaming_layers,
        "cross_restore_prefetch_layers": cross_restore_layers,
        "supports_inference": supports_inference,
        "g_storage_mb": g_bytes / (1024 ** 2),
        "g_gpu_mb": g_gpu_bytes / (1024 ** 2),
        "g_cpu_mb": g_cpu_bytes / (1024 ** 2),
        "g_cpu_pinned_mb": pinned_bytes / (1024 ** 2),
        "streaming_persistent_pin_budget_mb": pin_budget_mb,
        "streaming_persistent_pin_select": pin_select,
        "streaming_window_pin_cache_mb": cache_budget_mb,
        "streaming_window_pin_cache_selected_layers": len(window_cache_names or []),
        "streaming_window_pin_cache_estimated_used_mb": window_cache_budget_used_bytes / (1024 ** 2),
        "streaming_pin_hints_status": pin_hints_status,
        "streaming_persistent_pin_used_mb": pin_budget_used_bytes / (1024 ** 2),
        "streaming_persistent_pin_layers": persistent_pin_layers,
        "streaming_window_pin_layers": window_pin_layers,
        "memory_resident_budget_mb": resident_budget_mb,
        "memory_resident_select": resident_select,
        "memory_resident_used_mb": resident_budget_used_bytes / (1024 ** 2),
        "memory_resident_estimated_used_mb": resident_budget_estimated_used_bytes / (1024 ** 2),
        "memory_resident_layers": len(resident_names),
        "memory_lazy_release_layers": lazy_layers,
        "memory_resident_error_count": len(resident_errors),
        "memory_resident_errors": resident_errors[:8],
    }


def main():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["hf", "original", "v2", "v3"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-loader", choices=["auto", "causal", "multimodal"], default="auto")
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-on-device", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-exclude-linearmem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-profiler-capture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--workload", choices=["prefill", "classification", "generation"], default="prefill")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--classification-task", choices=["token_proxy", "arc_easy_smoke", "multi_choice_json"], default="token_proxy")
    parser.add_argument("--classification-examples-json", default="")
    parser.add_argument("--classification-choice-labels", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--classification-token-ids", type=int, nargs="*", default=None)
    parser.add_argument("--classification-candidate-count", type=int, default=4)
    parser.add_argument("--classification-scoring", choices=["label_token", "choice_text_ll"], default="label_token")
    parser.add_argument("--classification-score-batch-size", type=int, default=8)
    parser.add_argument("--save-logits", default="")
    parser.add_argument("--execution-mode", choices=["speed", "balanced", "memory"], default=None)
    parser.add_argument("--s1-stage", choices=["off", "block", "budgeted"], default="budgeted")
    parser.add_argument("--s2-stage", choices=["off", "intra", "full"], default="full")
    parser.add_argument("--cuda-peak-budget-mb", type=float, default=0.0)
    parser.add_argument("--state-resident-budget-mb", type=float, default=0.0)
    parser.add_argument("--output-block-cols", type=int, default=0)
    parser.add_argument("--memory-prepare-policy", choices=["lazy_release", "lazy_keep", "eager_streaming"], default="lazy_release")
    parser.add_argument("--memory-runtime-diagnostic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-budget-mb", type=float, default=0.0)
    parser.add_argument("--memory-resident-budget-mb", type=float, default=0.0)
    parser.add_argument("--memory-resident-select", choices=["sequential", "largest", "runtime"], default="largest")
    parser.add_argument("--require-all-linears", action="store_true")
    parser.add_argument("--simulate-lm-head", action="store_true")
    parser.add_argument("--only-lm-head", action="store_true")
    parser.add_argument("--always-include-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--linear-include-regex", action="append", default=[])
    parser.add_argument("--linear-exclude-regex", action="append", default=[])
    parser.add_argument("--linear-name-limit", type=int, default=32)
    parser.add_argument("--lm-head-output-shards", type=int, default=1)
    parser.add_argument("--lm-head-shard-devices", default="")
    parser.add_argument("--lm-head-shard-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lm-head-input-select", choices=["all", "last"], default="all")
    parser.add_argument("--lm-head-output-select", choices=["all", "label_tokens"], default="all")
    parser.add_argument("--max-linears", type=int, default=None)
    parser.add_argument("--max-non-lm-head-linears", type=int, default=None)
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--mode2-input-mode", choices=["signed", "differential"], default="signed")
    parser.add_argument("--fast-inference", action="store_true")
    parser.add_argument("--fast-inference-backend", choices=["auto", "torch", "triton", "triton_gidx"], default="auto")
    parser.add_argument("--triton-input-precision", choices=["ieee", "tf32", "tf32x3"], default="ieee")
    parser.add_argument("--triton-block-r", type=int, default=32)
    parser.add_argument("--triton-block-l", type=int, default=16)
    parser.add_argument("--triton-block-k", type=int, default=64)
    parser.add_argument("--triton-output-chunk-limit", type=int, default=256)
    parser.add_argument("--triton-auto-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode0-input-tile-group", type=int, default=1)
    parser.add_argument("--triton-gidx-read-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-fused-restore-read-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-block", type=int, default=512)
    parser.add_argument("--triton-gidx-restore-block-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-small-block", type=int, default=128)
    parser.add_argument("--triton-gidx-restore-auto-in-features-threshold", type=int, default=4096)
    parser.add_argument("--triton-gidx-restore-num-warps", type=int, default=4)
    parser.add_argument("--triton-gidx-restore-strided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-m-slab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-approx-linear-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-restore-exp2-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-fast-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-fuse-input-slices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode0-strict-intermediate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--triton-mode0-strict-intermediate-backend",
        choices=["auto", "gidx", "off"],
        default="auto",
    )
    parser.add_argument("--triton-reuse-input-voltage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-reuse-weight-tile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-precompute-input-voltage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fast-adc-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-restored-input-slices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-activation-slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-reuse-activation-slice-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-probe-activation-slice-reuse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-probe-activation-density", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-activation-slice-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-activation-slice-cache-max-entries", type=int, default=8)
    parser.add_argument("--triton-binary-input-slice-dac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-direct-final-num-warps", type=int, default=4)
    parser.add_argument("--triton-direct-final-partial-m-group", type=int, default=0)
    parser.add_argument("--triton-direct-final-exact-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-output-finalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-direct-final-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-direct-output-zero-once", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-direct-final-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-direct-final-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-overlap-restore-direct", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-cross-linear-restore-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode1-gidx-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode1-input-tile-group", type=int, default=1)
    parser.add_argument("--triton-mode1-chunked-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-presubtract", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode2-diff-fuse-input-slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-block-r-cap", type=int, default=16)
    parser.add_argument("--triton-mode2-diff-block-l-cap", type=int, default=8)
    parser.add_argument("--mode1-grouped-tile-gemm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-output-chunk-write", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-mlp-gate-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fuse-common-input-projections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-sync-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-stage-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-counters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--module-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-prefetch-cycle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-prefetch-distance", type=int, default=1)
    parser.add_argument("--streaming-pin-policy", choices=["persistent", "window"], default="persistent")
    parser.add_argument("--streaming-persistent-pin-budget-mb", type=float, default=0.0)
    parser.add_argument("--streaming-persistent-pin-select", choices=["sequential", "largest", "runtime"], default="sequential")
    parser.add_argument("--streaming-window-pin-cache-mb", type=float, default=0.0)
    parser.add_argument("--streaming-pin-hints-json", default="")
    parser.add_argument("--memory-empty-cache-after-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-empty-cache-after-offload-interval", type=int, default=1)
    parser.add_argument("--free-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lazy-prepare", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lazy-release-after-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hgs", type=float, default=1e-5)
    parser.add_argument("--lgs", type=float, default=1e-8)
    parser.add_argument("--g-level", type=int, default=16)
    parser.add_argument("--write-variation", type=float, default=0.0)
    parser.add_argument("--read-variation", type=float, default=0.0)
    parser.add_argument("--read-variation-seed", type=int, default=None)
    parser.add_argument("--vnoise", type=float, default=0.0)
    parser.add_argument("--write-variation-mode", choices=["materialized", "virtual"], default="materialized")
    parser.add_argument("--conductance-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--linear-output-dtype", choices=["auto", "input", "float32", "float16", "bfloat16"], default="input")
    parser.add_argument("--vmm-lowp-format", choices=VMM_LOWP_CHOICES, default="auto")
    parser.add_argument("--mode0-semantic-policy", choices=["auto", "strict", "fast"], default="auto")
    parser.add_argument("--mode0-vmm-compute-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--rdac", type=int, default=256)
    parser.add_argument("--radc", type=int, default=4096)
    parser.add_argument("--vread", type=float, default=0.2)
    parser.add_argument("--rate-stuck-hgs", type=float, default=0.0)
    parser.add_argument("--rate-stuck-lgs", type=float, default=0.0)
    parser.add_argument("--input-slice", type=int, nargs="+", default=[1, 1, 1, 1, 1])
    parser.add_argument("--weight-slice", type=int, nargs="+", default=[1, 1, 1, 1, 1])
    parser.add_argument("--mode2-input-slice", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--mode2-weight-slice", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--input-paral-size", type=int, nargs=2, default=[1, 64])
    parser.add_argument("--weight-paral-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--input-quant-gran", type=int, nargs=2, default=[1, 64])
    parser.add_argument("--weight-quant-gran", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--inference-chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--collect-layer-buffer-accounting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-buffer-accounting-limit", type=int, default=12)
    args = parser.parse_args()
    args = apply_vmm_lowp_format_defaults(args, argv_has_option=lambda opt: _worker_argv_has_option(raw_argv, opt))
    if args.kind == "v3" and args.vmm_lowp_format in VMM_LOWP_EXPERIMENTAL_8BIT:
        raise RuntimeError(unsupported_8bit_vmm_reason(args.vmm_lowp_format))
    lowp_sets_vmm_dtype = args.vmm_lowp_format in VMM_LOWP_STABLE_DTYPES
    if args.kind == "v3" and args.execution_mode in {"speed", "balanced", "memory"}:
        if not _worker_argv_has_option(raw_argv, "--conductance-dtype") and not lowp_sets_vmm_dtype:
            args.conductance_dtype = "bfloat16"
        if not _worker_argv_has_option(raw_argv, "--compute-dtype") and not lowp_sets_vmm_dtype:
            args.compute_dtype = "bfloat16"
    args = apply_execution_mode_defaults(args)
    args = apply_worker_s1_stage(args)
    args = apply_worker_s2_stage(args)

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_memory_limit = configure_cuda_memory_fraction(args, device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    config = prepare_config(config, args)
    vocab_size = config_vocab_size(config)
    args.lm_head_output_token_ids = resolve_lm_head_output_token_ids(args)

    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    model = load_hf_model(args, config, dtype)
    model.config.use_cache = bool(args.workload == "generation")
    original_linear_names = list_linear_module_names(model)
    validate_full_model_request(args, original_linear_names)
    sync(device)
    load_ms = (time.perf_counter() - t0) * 1000.0
    load_peak = peak_mb(device)

    replaced = 0
    skipped = 0
    replaced_linear_names = []
    non_lm_head_replaced = 0
    lm_head_replaced = 0
    fused_gate_up_groups = 0
    fused_common_projection_groups = 0
    fused_common_projection_linears = 0
    output_blocked_linears = 0
    output_block_plans = []
    LinearMem = None
    prep_info = {
        "prepared_layers": 0,
        "lazy_layers": 0,
        "streaming_layers": 0,
        "cross_restore_prefetch_layers": 0,
        "supports_inference": False,
        "g_storage_mb": 0.0,
        "g_gpu_mb": 0.0,
        "g_cpu_mb": 0.0,
        "g_cpu_pinned_mb": 0.0,
    }
    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    if args.kind == "hf":
        if not args.load_on_device:
            model = model.to(device)
    else:
        from memintelli.NN_layers.linear import LinearMem

        engine = make_engine(args, device)
        args.planner_base_allocated_mb = estimate_non_linear_model_bytes(model) / (1024 ** 2)
        replace_state = {
            "replaced": 0,
            "replaced_names": [],
            "non_lm_head_replaced": 0,
            "lm_head_replaced": 0,
            "fused_gate_up_groups": 0,
            "fused_common_projection_groups": 0,
            "fused_common_projection_linears": 0,
        }
        replaced, skipped = replace_linear(model, LinearMem, args, engine, device, state=replace_state)
        replaced_linear_names = list(replace_state["replaced_names"])
        non_lm_head_replaced = int(replace_state["non_lm_head_replaced"])
        lm_head_replaced = int(replace_state["lm_head_replaced"])
        fused_gate_up_groups = int(replace_state.get("fused_gate_up_groups", 0))
        fused_common_projection_groups = int(replace_state.get("fused_common_projection_groups", 0))
        fused_common_projection_linears = int(replace_state.get("fused_common_projection_linears", 0))
        output_blocked_linears = int(replace_state.get("output_blocked_linears", 0))
        output_block_plans = list(replace_state.get("output_block_plans", []))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        prep_info = prepare_mem_model(model, LinearMem, args, device)
        if not args.load_on_device:
            if keep_linear_weights_on_cpu(args):
                model = move_non_mem_tensors_to_device(model, LinearMem, device)
            else:
                model = model.to(device)
    model.eval()
    sync(device)
    prepare_ms = (time.perf_counter() - t0) * 1000.0
    prepare_peak = peak_mb(device)
    resident_mb = alloc_mb(device)
    forward_model = model
    torch_compile_wrap_ms = 0.0
    torch_compile_exclude_linearmem_info = {"enabled": False, "reason": "disabled", "wrapped_modules": 0}
    if args.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        if bool(getattr(args, "torch_compile_exclude_linearmem", False)):
            torch_compile_exclude_linearmem_info = disable_linearmem_for_torch_compile(model, LinearMem)
        sync(device)
        t_compile = time.perf_counter()
        forward_model = torch.compile(
            model,
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode,
            fullgraph=False,
            dynamic=bool(args.torch_compile_dynamic),
        )
        sync(device)
        torch_compile_wrap_ms = (time.perf_counter() - t_compile) * 1000.0
    remaining_linear_names = list_linear_module_names(model)
    linear_manifest = build_linear_manifest(
        original_linear_names,
        remaining_linear_names,
        replaced,
        lm_head_replaced,
        args,
    )
    if args.require_all_linears and args.kind != "hf" and not linear_manifest["full_model_deployment_valid"]:
        raise RuntimeError("full-model deployment failed: " + json.dumps(linear_manifest, sort_keys=True))

    classification_context = {
        "classification_task": args.classification_task,
        "classification_examples_json": args.classification_examples_json,
        "classification_example_ids": [],
        "classification_choice_labels": list(args.classification_choice_labels),
        "classification_choice_texts": [],
        "classification_choice_token_ids": [],
        "classification_gold_labels": [],
        "classification_prompt_count": 0,
        "classification_prompt_preview": [],
    }
    if args.workload == "classification" and args.classification_task in {"arc_easy_smoke", "multi_choice_json"}:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        input_ids, attention_mask, classification_context = build_multi_choice_classification_batch(args, tokenizer, device)
    else:
        input_ids = torch.randint(0, vocab_size, (args.batch, args.seq), device=device)
        attention_mask = torch.ones_like(input_ids)

    def classification_token_ids():
        if args.classification_token_ids:
            ids = [int(token_id) for token_id in args.classification_token_ids]
        elif classification_context.get("classification_choice_token_ids"):
            ids = [int(token_id) for token_id in classification_context["classification_choice_token_ids"]]
        else:
            ids = list(range(max(1, int(args.classification_candidate_count))))
        ids = sorted({token_id for token_id in ids if 0 <= token_id < vocab_size})
        if len(ids) < 2:
            ids = list(range(min(2, vocab_size)))
        return ids

    def classification_label_token_metrics(logits):
        token_ids = classification_token_ids()
        final_logits = logits[:, -1, :]
        if args.lm_head_output_select == "label_tokens":
            output_token_ids = [int(item) for item in args.lm_head_output_token_ids]
            output_index = {token_id: idx for idx, token_id in enumerate(output_token_ids)}
            missing = [token_id for token_id in token_ids if token_id not in output_index]
            if missing:
                raise RuntimeError(
                    "reduced lm_head token ids do not cover classification token ids: "
                    f"missing={missing}, output={output_token_ids}, requested={token_ids}"
                )
            column_index = torch.tensor([output_index[token_id] for token_id in token_ids], device=final_logits.device)
            label_logits = final_logits.index_select(-1, column_index)
        else:
            label_logits = final_logits[:, token_ids]
        label_probs = torch.softmax(label_logits.float(), dim=-1)
        order = torch.argsort(label_logits, dim=-1, descending=True)
        pred_indices = order[:, 0]
        pred_ids = [token_ids[int(idx)] for idx in pred_indices.detach().cpu()]
        choice_labels = classification_context.get("classification_choice_labels") or [str(idx) for idx in range(len(token_ids))]
        pred_labels = [
            choice_labels[int(idx)] if int(idx) < len(choice_labels) else str(int(idx))
            for idx in pred_indices.detach().cpu()
        ]
        gold_labels = list(classification_context.get("classification_gold_labels") or [])
        accuracy = None
        if gold_labels and len(gold_labels) == len(pred_labels):
            accuracy = sum(int(p == g) for p, g in zip(pred_labels, gold_labels)) / max(1, len(gold_labels))
        if len(token_ids) > 1:
            margin = label_logits.gather(1, order[:, :1]) - label_logits.gather(1, order[:, 1:2])
            margin_mean = float(torch.mean(margin.float()))
        else:
            margin_mean = None
        entropy = -torch.sum(label_probs * torch.log(torch.clamp(label_probs, min=1e-30)), dim=-1)
        return {
            "classification_proxy": True,
            "classification_scoring": "label_token",
            "classification_task": classification_context.get("classification_task", args.classification_task),
            "classification_examples_json": classification_context.get("classification_examples_json", ""),
            "classification_example_ids": classification_context.get("classification_example_ids", []),
            "classification_choice_labels": choice_labels,
            "classification_choice_texts": classification_context.get("classification_choice_texts", []),
            "classification_token_ids": token_ids,
            "classification_pred_token_ids": pred_ids,
            "classification_pred_labels": pred_labels,
            "classification_gold_labels": gold_labels,
            "classification_accuracy_proxy": accuracy,
            "classification_label_logits": label_logits.detach().float().cpu().tolist(),
            "classification_margin_mean": margin_mean,
            "classification_entropy_mean": float(torch.mean(entropy)),
            "classification_top_score_mean": float(torch.mean(torch.max(label_logits.float(), dim=-1).values)),
            "classification_prompt_count": classification_context.get("classification_prompt_count", 0),
            "classification_prompt_preview": classification_context.get("classification_prompt_preview", []),
        }

    def classification_choice_text_ll_metrics():
        labels = list(classification_context.get("classification_choice_labels") or [])
        prompts = list(classification_context.get("classification_prompt_preview") or [])
        all_prompts = []
        choice_texts = list(classification_context.get("classification_choice_texts") or [])
        # prompt_preview is truncated, so rebuild prompts from input context when needed.
        examples = load_classification_examples(args)
        selected = [examples[(args.seed + idx) % len(examples)] for idx in range(max(1, int(args.batch)))] if examples else []
        for example in selected:
            all_prompts.append(prompt_from_example(example, labels))
        if not all_prompts:
            all_prompts = prompts
        if not choice_texts and selected:
            choice_texts = [choice_texts_from_example(example, labels) for example in selected]
        if not all_prompts or not choice_texts:
            return classification_label_token_metrics(logits)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        candidate_records = []
        for row_idx, (prompt, choices) in enumerate(zip(all_prompts, choice_texts)):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            for choice_idx, choice in enumerate(choices[: len(labels)]):
                choice_ids = tokenizer.encode(" " + str(choice).strip(), add_special_tokens=False)
                if not choice_ids:
                    choice_ids = tokenizer.encode(str(choice).strip(), add_special_tokens=False)
                if not choice_ids:
                    choice_ids = [pad_token_id]
                ids = prompt_ids + choice_ids
                if len(ids) > int(args.seq):
                    prompt_keep = max(1, int(args.seq) - len(choice_ids))
                    prompt_ids_trimmed = prompt_ids[-prompt_keep:]
                    ids = prompt_ids_trimmed + choice_ids
                    prompt_len = len(prompt_ids_trimmed)
                else:
                    prompt_len = len(prompt_ids)
                candidate_records.append(
                    {
                        "row_idx": row_idx,
                        "choice_idx": choice_idx,
                        "ids": ids,
                        "prompt_len": prompt_len,
                        "choice_len": len(choice_ids),
                    }
                )

        scores = torch.full((len(all_prompts), len(labels)), -1.0e30, dtype=torch.float32, device="cpu")
        max_len = max(len(item["ids"]) for item in candidate_records)
        batch_size = max(1, int(getattr(args, "classification_score_batch_size", 8) or 8))
        for start in range(0, len(candidate_records), batch_size):
            chunk = candidate_records[start : start + batch_size]
            encoded_rows = []
            attention_rows = []
            for item in chunk:
                pad_len = max_len - len(item["ids"])
                encoded_rows.append([pad_token_id] * pad_len + item["ids"])
                attention_rows.append([0] * pad_len + [1] * len(item["ids"]))
            cand_input = torch.tensor(encoded_rows, dtype=torch.long, device=device)
            cand_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
            cand_out = forward_model(input_ids=cand_input, attention_mask=cand_mask, use_cache=False)
            cand_log_probs = torch.log_softmax(cand_out.logits.float(), dim=-1)
            for local_idx, item in enumerate(chunk):
                pad_len = max_len - len(item["ids"])
                token_start = pad_len + item["prompt_len"]
                token_end = pad_len + len(item["ids"])
                total = 0.0
                count = 0
                for pos in range(token_start, token_end):
                    target = cand_input[local_idx, pos]
                    total += float(cand_log_probs[local_idx, pos - 1, target].detach().cpu())
                    count += 1
                norm_score = total / max(1, count)
                scores[item["row_idx"], item["choice_idx"]] = norm_score

        order = torch.argsort(scores, dim=-1, descending=True)
        pred_indices = order[:, 0]
        pred_labels = [
            labels[int(idx)] if int(idx) < len(labels) else str(int(idx))
            for idx in pred_indices.detach().cpu()
        ]
        pred_ids = [int(idx) for idx in pred_indices.detach().cpu()]
        gold_labels = list(classification_context.get("classification_gold_labels") or [])
        accuracy = None
        if gold_labels and len(gold_labels) == len(pred_labels):
            accuracy = sum(int(p == g) for p, g in zip(pred_labels, gold_labels)) / max(1, len(gold_labels))
        if scores.shape[1] > 1:
            margin = scores.gather(1, order[:, :1]) - scores.gather(1, order[:, 1:2])
            margin_mean = float(torch.mean(margin.float()))
        else:
            margin_mean = None
        probs = torch.softmax(scores.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-30)), dim=-1)
        return {
            "classification_proxy": True,
            "classification_scoring": "choice_text_ll",
            "classification_task": classification_context.get("classification_task", args.classification_task),
            "classification_examples_json": classification_context.get("classification_examples_json", ""),
            "classification_example_ids": classification_context.get("classification_example_ids", []),
            "classification_choice_labels": labels,
            "classification_choice_texts": choice_texts,
            "classification_token_ids": [],
            "classification_pred_token_ids": pred_ids,
            "classification_pred_labels": pred_labels,
            "classification_gold_labels": gold_labels,
            "classification_accuracy_proxy": accuracy,
            "classification_label_logits": scores.detach().float().cpu().tolist(),
            "classification_margin_mean": margin_mean,
            "classification_entropy_mean": float(torch.mean(entropy)),
            "classification_top_score_mean": float(torch.mean(torch.max(scores.float(), dim=-1).values)),
            "classification_prompt_count": len(all_prompts),
            "classification_prompt_preview": all_prompts[: min(2, len(all_prompts))],
        }

    def classification_metrics(logits):
        if args.workload != "classification":
            return {}
        if getattr(args, "classification_scoring", "label_token") == "choice_text_ll":
            return classification_choice_text_ll_metrics()
        return classification_label_token_metrics(logits)

    def next_token_ce_metrics(logits):
        metrics = {
            "next_token_ce_available": False,
            "next_token_ce": None,
            "next_token_ppl": None,
            "next_token_count": 0,
            "next_token_reason": "",
        }
        if args.lm_head_output_select != "all":
            metrics["next_token_reason"] = "lm_head_output_select_not_all"
            return metrics
        if logits is None or logits.dim() != 3 or input_ids is None or input_ids.dim() != 2:
            metrics["next_token_reason"] = "missing_logits_or_input_ids"
            return metrics
        usable = min(int(logits.shape[1]), int(input_ids.shape[1])) - 1
        if usable <= 0:
            metrics["next_token_reason"] = "sequence_too_short"
            return metrics
        shift_logits = logits[:, :usable, :].float().contiguous()
        shift_labels = input_ids[:, 1 : usable + 1].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1 : usable + 1].contiguous().bool()
        else:
            shift_mask = torch.ones_like(shift_labels, dtype=torch.bool)
        valid_labels = shift_labels >= 0
        valid_labels.logical_and_(shift_labels < shift_logits.shape[-1])
        shift_mask.logical_and_(valid_labels)
        token_count = int(shift_mask.sum().item())
        if token_count <= 0:
            metrics["next_token_reason"] = "no_valid_tokens"
            return metrics
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.reshape(-1)
        flat_mask = shift_mask.reshape(-1)
        losses = torch.nn.functional.cross_entropy(
            flat_logits[flat_mask],
            flat_labels[flat_mask],
            reduction="none",
        )
        ce = float(losses.mean().item())
        metrics.update({
            "next_token_ce_available": True,
            "next_token_ce": ce,
            "next_token_ppl": float(math.exp(min(ce, 80.0))),
            "next_token_count": token_count,
            "next_token_reason": "",
        })
        return metrics

    def run_prefill_or_classification():
        out = forward_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits, 0.0, 0.0

    generation_state = {
        "token_ids": [],
        "token_ids_by_step": [],
    }

    def run_generation():
        sync(device)
        t_prefill = time.perf_counter()
        out = forward_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        sync(device)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0
        logits = out.logits
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RuntimeError("Generation workload requires model output past_key_values")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        decode_attention_mask = attention_mask
        decode_ms = 0.0
        generated_by_step = []
        for _ in range(max(0, args.decode_steps)):
            generated_by_step.append([int(item) for item in next_token.detach().view(-1).cpu().tolist()])
            decode_attention_mask = torch.cat(
                [decode_attention_mask, torch.ones((args.batch, 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            sync(device)
            t_decode = time.perf_counter()
            out = forward_model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            sync(device)
            decode_ms += (time.perf_counter() - t_decode) * 1000.0
            logits = out.logits
            past = getattr(out, "past_key_values", None)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if generated_by_step:
            generation_state["token_ids_by_step"] = generated_by_step
            generation_state["token_ids"] = [
                [step[batch_idx] for step in generated_by_step]
                for batch_idx in range(len(generated_by_step[0]))
            ]
        else:
            generation_state["token_ids_by_step"] = []
            generation_state["token_ids"] = []
        return logits, prefill_ms, decode_ms

    def run_workload_once():
        if args.workload == "generation":
            return run_generation()
        return run_prefill_or_classification()

    cross_restore_trace_info = {
        "enabled": False,
        "trace_length": 0,
        "unique_layers": 0,
        "links": 0,
    }
    with torch.no_grad():
        if (
            args.kind != "hf"
            and bool(getattr(args, "triton_cross_linear_restore_prefetch", False))
            and LinearMem is not None
            and hasattr(LinearMem, "begin_execution_trace")
            and hasattr(LinearMem, "end_execution_trace")
        ):
            LinearMem.begin_execution_trace()
            try:
                _ = run_workload_once()
            finally:
                trace = LinearMem.end_execution_trace()
            cross_restore_trace_info = configure_cross_restore_prefetch_from_execution_trace(
                model,
                LinearMem,
                trace,
            )
            for _ in range(max(0, int(args.warmup) - 1)):
                _ = run_workload_once()
        else:
            for _ in range(args.warmup):
                _ = run_workload_once()
        sync(device)
        reset_runtime_mechanism_counters(model, LinearMem)
        reset_engine_fastpath_counters(model, LinearMem)

        times = []
        prefill_times = []
        decode_times = []
        profile_summaries = []
        profile_layer_counts = []
        peaks = []
        logits = None
        module_timing_summaries = []
        semantic_probe_rows = []
        layer_peak_trace_rows = []
        for _ in range(max(1, args.repeat)):
            if args.kind != "hf" and args.profile:
                reset_engine_profiles(model, LinearMem)
            reset_peak(device)
            sync(device)
            t0 = time.perf_counter()
            if args.cuda_profiler_capture:
                cuda_profiler_start(device)
            with LayerPeakTraceContext(
                model,
                LinearMem,
                enabled=os.environ.get("MEMINTELLI_LAYER_PEAK_TRACE", "0") == "1",
            ) as layer_peak_trace:
                with SemanticOutputProbe(
                    model,
                    LinearMem,
                    enabled=os.environ.get("MEMINTELLI_SEMANTIC_PROBE", "0") == "1",
                ) as semantic_probe:
                    with ModuleTimingContext(model, LinearMem, enabled=args.module_timing) as module_timing:
                        logits, prefill_ms, decode_ms = run_workload_once()
            semantic_probe_rows.extend(semantic_probe.rows)
            layer_peak_trace_rows.extend(layer_peak_trace.rows)
            sync(device)
            if args.cuda_profiler_capture:
                cuda_profiler_stop(device)
            iter_ms = (time.perf_counter() - t0) * 1000.0
            times.append(iter_ms)
            prefill_times.append(prefill_ms if args.workload == "generation" else times[-1])
            decode_times.append(decode_ms if args.workload == "generation" else 0.0)
            peaks.append(peak_mb(device))
            if args.module_timing:
                module_timing_summaries.append(module_timing.summary(total_ms=iter_ms))
            if args.kind != "hf" and args.profile:
                profile_layers, profile_summary = collect_engine_profile(model, LinearMem)
                profile_layer_counts.append(profile_layers)
                profile_summaries.append(profile_summary)

    if logits is None or not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite or missing logits")
    if args.save_logits:
        logits_path = os.path.abspath(args.save_logits)
        os.makedirs(os.path.dirname(logits_path), exist_ok=True)
        torch.save(logits.detach().to(device="cpu", dtype=torch.float32), logits_path)

    total_ms_mean = sum(times) / len(times)
    prefill_ms_mean = sum(prefill_times) / len(prefill_times)
    decode_ms_mean = sum(decode_times) / len(decode_times)
    merged_profile = merge_profile_summaries(profile_summaries) if profile_summaries else {}
    profiled_total_ms = sum(item.get("total_ms", 0.0) for item in merged_profile.values())
    module_timing_summary = module_timing_summaries[-1] if module_timing_summaries else {"enabled": False}
    simulated_tokens = args.batch * (args.seq + (args.decode_steps if args.workload == "generation" else 0))
    decode_tokens = args.batch * args.decode_steps if args.workload == "generation" else 0
    cls_metrics = classification_metrics(logits)
    ce_metrics = next_token_ce_metrics(logits)
    layer_buffer_accounting = (
        collect_layer_buffer_accounting(model, LinearMem, args, prep_info)
        if args.collect_layer_buffer_accounting
        else {"enabled": False}
    )
    runtime_mechanisms = collect_runtime_mechanism_counters(model, LinearMem, args)

    out = {
        "status": "ok",
        "error": "",
        "label": args.label,
        "kind": args.kind,
        "model_path": args.model_path,
        "model_type": getattr(config, "model_type", "unknown"),
        "model_loader": args.model_loader,
        "language_model_only": bool(args.language_model_only),
        "load_on_device": bool(args.load_on_device),
        "batch": args.batch,
        "seq": args.seq,
        "workload": args.workload,
        "decode_steps": args.decode_steps if args.workload == "generation" else 0,
        "generation_token_ids": generation_state["token_ids"] if args.workload == "generation" else [],
        "generation_token_ids_by_step": generation_state["token_ids_by_step"] if args.workload == "generation" else [],
        "classification_task": cls_metrics.get("classification_task", args.classification_task),
        "classification_scoring": cls_metrics.get("classification_scoring", args.classification_scoring),
        "classification_examples_json": cls_metrics.get("classification_examples_json", args.classification_examples_json),
        "classification_token_ids": cls_metrics.get("classification_token_ids", []),
        "classification_choice_labels": cls_metrics.get("classification_choice_labels", list(args.classification_choice_labels)),
        "classification_example_ids": cls_metrics.get("classification_example_ids", []),
        "classification_gold_labels": cls_metrics.get("classification_gold_labels", []),
        "next_token_ce_available": ce_metrics.get("next_token_ce_available", False),
        "next_token_ce": ce_metrics.get("next_token_ce"),
        "next_token_ppl": ce_metrics.get("next_token_ppl"),
        "next_token_count": ce_metrics.get("next_token_count", 0),
        "next_token_reason": ce_metrics.get("next_token_reason", ""),
        "dtype": str(dtype),
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": args.torch_compile_mode if args.torch_compile else "none",
        "torch_compile_backend": args.torch_compile_backend if args.torch_compile else "none",
        "torch_compile_dynamic": bool(args.torch_compile_dynamic) if args.torch_compile else False,
        "torch_compile_exclude_linearmem": bool(args.torch_compile_exclude_linearmem) if args.torch_compile else False,
        "torch_compile_exclude_linearmem_info": torch_compile_exclude_linearmem_info,
        "torch_compile_wrap_ms": torch_compile_wrap_ms,
        "execution_mode": args.execution_mode or "manual",
        "memory_prepare_policy": args.memory_prepare_policy,
        "memory_runtime_diagnostic": bool(args.memory_runtime_diagnostic),
        "runtime_stage_timing": bool(args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
        "runtime_counters": bool(args.runtime_counters or args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
        "memory_budget_mb": float(getattr(args, "memory_budget_mb", 0.0) or 0.0),
        "memory_resident_budget_mb": float(args.memory_resident_budget_mb or 0.0),
        "memory_resident_select": args.memory_resident_select,
        "memory_resident_used_mb": prep_info.get("memory_resident_used_mb"),
        "memory_resident_layers": prep_info.get("memory_resident_layers"),
        "memory_lazy_release_layers": prep_info.get("memory_lazy_release_layers"),
        "memory_resident_error_count": prep_info.get("memory_resident_error_count"),
        "memory_resident_errors": prep_info.get("memory_resident_errors", []),
        "require_all_linears": bool(args.require_all_linears),
        "simulate_lm_head": args.simulate_lm_head,
        "only_lm_head": args.only_lm_head,
        "always_include_lm_head": args.always_include_lm_head,
        "linear_include_regex": list(args.linear_include_regex),
        "linear_exclude_regex": list(args.linear_exclude_regex),
        "lm_head_output_shards": args.lm_head_output_shards,
        "lm_head_shard_devices": args.lm_head_shard_devices,
        "lm_head_shard_parallel": bool(args.lm_head_shard_parallel),
        "lm_head_input_select": args.lm_head_input_select,
        "lm_head_output_select": args.lm_head_output_select,
        "lm_head_output_token_ids": list(args.lm_head_output_token_ids),
        "fuse_mlp_gate_up": bool(getattr(args, "fuse_mlp_gate_up", False)),
        "fused_gate_up_groups": int(fused_gate_up_groups),
        "fuse_common_input_projections": bool(getattr(args, "fuse_common_input_projections", False)),
        "fused_common_projection_groups": int(fused_common_projection_groups),
        "fused_common_projection_linears": int(fused_common_projection_linears),
        "output_blocked_linears": int(output_blocked_linears),
        "output_block_plans": output_block_plans,
        "mode": args.mode if args.kind == "v3" else 0,
        "mode2_input_mode": args.mode2_input_mode if args.kind == "v3" and args.mode == 2 else "n/a",
        "fast_inference": bool(args.fast_inference) if args.kind == "v3" else False,
        "fast_inference_backend": args.fast_inference_backend if args.kind == "v3" else "n/a",
        "g_level": args.g_level if args.kind in {"v2", "v3"} else None,
        "write_variation": args.write_variation if args.kind in {"v2", "v3"} else 0.0,
        "read_variation": args.read_variation if args.kind in {"v2", "v3"} else 0.0,
        "read_variation_seed": args.read_variation_seed if args.kind in {"v2", "v3"} else None,
        "vnoise": args.vnoise if args.kind in {"v2", "v3"} else 0.0,
        "write_variation_mode": args.write_variation_mode if args.kind == "v3" else "n/a",
        "vmm_lowp_format": getattr(args, "vmm_lowp_format", "auto") if args.kind == "v3" else "n/a",
        "vmm_lowp_status": getattr(args, "vmm_lowp_status", "n/a") if args.kind == "v3" else "n/a",
        "conductance_dtype": str(getattr(engine, "conductance_dtype", args.conductance_dtype)) if args.kind in {"v2", "v3"} else "n/a",
        "requested_compute_dtype": args.compute_dtype if args.kind in {"v2", "v3"} else "n/a",
        "compute_dtype": str(getattr(engine, "compute_dtype", getattr(engine, "vmm_compute_dtype", args.compute_dtype))) if args.kind in {"v2", "v3"} else "n/a",
        "adc_compute_dtype": str(getattr(engine, "adc_compute_dtype", "n/a")) if args.kind in {"v2", "v3"} else "n/a",
        "linear_output_dtype": str(getattr(engine, "linear_output_dtype", "n/a")) if args.kind == "v3" else "n/a",
        "mode0_semantic_policy": str(getattr(engine, "mode0_semantic_policy", "n/a")) if args.kind == "v3" else "n/a",
        "mode0_vmm_compute_dtype": str(getattr(engine, "mode0_vmm_compute_dtype", "n/a")) if args.kind == "v3" else "n/a",
        "rdac": args.rdac if args.kind in {"v2", "v3"} else None,
        "radc": args.radc if args.kind in {"v2", "v3"} else None,
        "vread": args.vread if args.kind in {"v2", "v3"} else None,
        "input_slice": list(args.input_slice) if args.kind in {"v2", "v3"} else [],
        "weight_slice": list(args.weight_slice) if args.kind in {"v2", "v3"} else [],
        "mode2_input_slice": list(args.mode2_input_slice) if args.kind == "v3" else [],
        "mode2_weight_slice": list(args.mode2_weight_slice) if args.kind == "v3" else [],
        "input_paral_size": list(args.input_paral_size) if args.kind in {"v2", "v3"} else [],
        "weight_paral_size": list(args.weight_paral_size) if args.kind in {"v2", "v3"} else [],
        "input_quant_gran": list(args.input_quant_gran) if args.kind in {"v2", "v3"} else [],
        "weight_quant_gran": list(args.weight_quant_gran) if args.kind in {"v2", "v3"} else [],
        "inference_chunk_size": int(args.inference_chunk_size) if args.kind in {"v2", "v3"} else None,
        "triton_input_precision": args.triton_input_precision if args.kind == "v3" else "n/a",
        "triton_block_r": args.triton_block_r if args.kind == "v3" else None,
        "triton_block_l": args.triton_block_l if args.kind == "v3" else None,
        "triton_block_k": args.triton_block_k if args.kind == "v3" else None,
        "triton_output_chunk_limit": args.triton_output_chunk_limit if args.kind == "v3" else None,
        "triton_auto_config": bool(args.triton_auto_config) if args.kind == "v3" else False,
        "triton_mode0_input_tile_group": int(args.triton_mode0_input_tile_group) if args.kind == "v3" else 1,
        "triton_gidx_read_noise": bool(args.triton_gidx_read_noise) if args.kind == "v3" else False,
        "triton_gidx_fused_restore_read_noise": bool(args.triton_gidx_fused_restore_read_noise) if args.kind == "v3" else False,
        "triton_gidx_restore_block": int(args.triton_gidx_restore_block) if args.kind == "v3" else 0,
        "triton_gidx_restore_block_auto": bool(args.triton_gidx_restore_block_auto) if args.kind == "v3" else False,
        "triton_gidx_restore_small_block": int(args.triton_gidx_restore_small_block) if args.kind == "v3" else 0,
        "triton_gidx_restore_auto_in_features_threshold": (
            int(args.triton_gidx_restore_auto_in_features_threshold) if args.kind == "v3" else 0
        ),
        "triton_gidx_restore_num_warps": int(args.triton_gidx_restore_num_warps) if args.kind == "v3" else 0,
        "triton_gidx_restore_strided": bool(args.triton_gidx_restore_strided) if args.kind == "v3" else False,
        "triton_gidx_restore_m_slab": bool(args.triton_gidx_restore_m_slab) if args.kind == "v3" else False,
        "triton_gidx_restore_approx_linear_noise": (
            bool(args.triton_gidx_restore_approx_linear_noise) if args.kind == "v3" else False
        ),
        "triton_gidx_restore_exp2_noise": (
            bool(args.triton_gidx_restore_exp2_noise) if args.kind == "v3" else False
        ),
        "triton_gidx_restore_fast_noise": (
            bool(args.triton_gidx_restore_fast_noise) if args.kind == "v3" else False
        ),
        "triton_gidx_fuse_input_slices": bool(args.triton_gidx_fuse_input_slices) if args.kind == "v3" else False,
        "triton_mode0_strict_intermediate": (
            bool(getattr(args, "triton_mode0_strict_intermediate", False)) if args.kind == "v3" else False
        ),
        "triton_mode0_strict_intermediate_backend": (
            str(getattr(args, "triton_mode0_strict_intermediate_backend", "off")) if args.kind == "v3" else "n/a"
        ),
        "triton_reuse_input_voltage": bool(args.triton_reuse_input_voltage) if args.kind == "v3" else False,
        "triton_reuse_weight_tile": bool(args.triton_reuse_weight_tile) if args.kind == "v3" else False,
        "triton_precompute_input_voltage": bool(args.triton_precompute_input_voltage) if args.kind == "v3" else False,
        "triton_fast_adc_scale": bool(args.triton_fast_adc_scale) if args.kind == "v3" else False,
        "triton_fuse_restored_input_slices": bool(args.triton_fuse_restored_input_slices) if args.kind == "v3" else False,
        "triton_fuse_activation_slices": bool(args.triton_fuse_activation_slices) if args.kind == "v3" else False,
        "triton_reuse_activation_slice_buffer": bool(args.triton_reuse_activation_slice_buffer) if args.kind == "v3" else False,
        "triton_probe_activation_slice_reuse": bool(args.triton_probe_activation_slice_reuse) if args.kind == "v3" else False,
        "triton_probe_activation_density": bool(args.triton_probe_activation_density) if args.kind == "v3" else False,
        "triton_activation_slice_cache": bool(args.triton_activation_slice_cache) if args.kind == "v3" else False,
        "triton_activation_slice_cache_max_entries": (
            int(args.triton_activation_slice_cache_max_entries) if args.kind == "v3" else 0
        ),
        "triton_binary_input_slice_dac": bool(args.triton_binary_input_slice_dac) if args.kind == "v3" else False,
        "triton_direct_final_num_warps": int(args.triton_direct_final_num_warps) if args.kind == "v3" else 0,
        "triton_direct_final_partial_m_group": (
            int(args.triton_direct_final_partial_m_group) if args.kind == "v3" else 0
        ),
        "triton_direct_final_exact_reduce": (
            bool(getattr(args, "triton_direct_final_exact_reduce", False)) if args.kind == "v3" else False
        ),
        "triton_fuse_output_finalize": bool(args.triton_fuse_output_finalize) if args.kind == "v3" else False,
        "triton_direct_final_output": bool(args.triton_direct_final_output) if args.kind == "v3" else False,
        "triton_direct_output_zero_once": bool(args.triton_direct_output_zero_once) if args.kind == "v3" else False,
        "triton_gidx_direct_final_output": bool(args.triton_gidx_direct_final_output) if args.kind == "v3" else False,
        "triton_gidx_direct_final_deterministic": (
            bool(args.triton_gidx_direct_final_deterministic) if args.kind == "v3" else False
        ),
        "triton_overlap_restore_direct": bool(args.triton_overlap_restore_direct) if args.kind == "v3" else False,
        "triton_cross_linear_restore_prefetch": (
            bool(args.triton_cross_linear_restore_prefetch) if args.kind == "v3" else False
        ),
        "triton_mode1_gidx_direct_final": bool(args.triton_mode1_gidx_direct_final) if args.kind == "v3" else False,
        "triton_mode1_input_tile_group": int(args.triton_mode1_input_tile_group) if args.kind == "v3" else 1,
        "triton_mode1_chunked_direct_final": bool(args.triton_mode1_chunked_direct_final) if args.kind == "v3" else False,
        "triton_mode2_diff_direct_final": bool(args.triton_mode2_diff_direct_final) if args.kind == "v3" else False,
        "triton_mode2_diff_presubtract": bool(args.triton_mode2_diff_presubtract) if args.kind == "v3" else False,
        "triton_mode2_diff_fuse_input_slices": bool(args.triton_mode2_diff_fuse_input_slices) if args.kind == "v3" else False,
        "triton_mode2_diff_block_r_cap": int(args.triton_mode2_diff_block_r_cap) if args.kind == "v3" else 0,
        "triton_mode2_diff_block_l_cap": int(args.triton_mode2_diff_block_l_cap) if args.kind == "v3" else 0,
        "mode1_grouped_tile_gemm": bool(args.mode1_grouped_tile_gemm) if args.kind == "v3" else False,
        "direct_output_chunk_write": bool(args.direct_output_chunk_write) if args.kind == "v3" else False,
        "profile": bool(args.profile) if args.kind != "hf" else False,
        "profile_sync_cuda": bool(args.profile_sync_cuda) if args.kind != "hf" else False,
        "streaming": bool(args.streaming) if args.kind != "hf" else False,
        "streaming_prefetch": bool(args.streaming_prefetch) if args.kind != "hf" else False,
        "streaming_prefetch_cycle": bool(args.streaming_prefetch_cycle) if args.kind != "hf" else False,
        "streaming_prefetch_distance": int(getattr(args, "streaming_prefetch_distance", 1) or 1) if args.kind != "hf" else 0,
        "streaming_pin_policy": args.streaming_pin_policy if args.kind != "hf" else "n/a",
        "streaming_persistent_pin_budget_mb": (
            float(args.streaming_persistent_pin_budget_mb or 0.0) if args.kind != "hf" else 0.0
        ),
        "streaming_persistent_pin_select": args.streaming_persistent_pin_select if args.kind != "hf" else "n/a",
        "streaming_window_pin_cache_mb": (
            float(args.streaming_window_pin_cache_mb or 0.0) if args.kind != "hf" else 0.0
        ),
        "streaming_window_pin_cache_selected_layers": prep_info.get("streaming_window_pin_cache_selected_layers"),
        "streaming_window_pin_cache_estimated_used_mb": prep_info.get("streaming_window_pin_cache_estimated_used_mb"),
        "streaming_pin_hints_json": args.streaming_pin_hints_json if args.kind != "hf" else "",
        "streaming_pin_hints_status": prep_info.get("streaming_pin_hints_status"),
        "streaming_persistent_pin_used_mb": prep_info.get("streaming_persistent_pin_used_mb"),
        "streaming_persistent_pin_layers": prep_info.get("streaming_persistent_pin_layers"),
        "streaming_window_pin_layers": prep_info.get("streaming_window_pin_layers"),
        "memory_empty_cache_after_offload": bool(args.memory_empty_cache_after_offload) if args.kind != "hf" else False,
        "memory_empty_cache_after_offload_interval": (
            max(1, int(getattr(args, "memory_empty_cache_after_offload_interval", 1) or 1))
            if args.kind != "hf"
            else 0
        ),
        "lazy_prepare": bool(args.lazy_prepare) if args.kind != "hf" else False,
        "lazy_release_after_forward": bool(args.lazy_release_after_forward) if args.kind != "hf" else False,
        "replaced_linears": replaced,
        "replaced_non_lm_head_linears": non_lm_head_replaced,
        "replaced_lm_head_linears": lm_head_replaced,
        "replaced_linear_names": replaced_linear_names,
        "skipped_linears": skipped,
        "linear_manifest": linear_manifest,
        "full_model_deployment_valid": bool(linear_manifest["full_model_deployment_valid"]),
        "original_linear_count": linear_manifest["original_linear_count"],
        "remaining_linear_count": linear_manifest["remaining_linear_count"],
        "replacement_coverage": linear_manifest["replacement_coverage"],
        "lm_head_simulated": linear_manifest["lm_head_simulated"],
        "prepared_layers": prep_info["prepared_layers"],
        "lazy_layers": prep_info["lazy_layers"],
        "streaming_layers": prep_info["streaming_layers"],
        "cross_restore_prefetch_layers": prep_info.get("cross_restore_prefetch_layers", 0),
        "cross_restore_trace_enabled": bool(cross_restore_trace_info.get("enabled", False)),
        "cross_restore_trace_length": int(cross_restore_trace_info.get("trace_length", 0) or 0),
        "cross_restore_trace_unique_layers": int(cross_restore_trace_info.get("unique_layers", 0) or 0),
        "cross_restore_trace_links": int(cross_restore_trace_info.get("links", 0) or 0),
        "supports_inference": prep_info["supports_inference"],
        "g_storage_mb": prep_info["g_storage_mb"],
        "g_gpu_mb": prep_info["g_gpu_mb"],
        "g_cpu_mb": prep_info["g_cpu_mb"],
        "g_cpu_pinned_mb": max(
            float(prep_info["g_cpu_pinned_mb"] or 0.0),
            float(runtime_mechanisms.get("pinned_buffer_peak_mb", 0.0) or 0.0),
        ),
        "runtime_current_pinned_mb": runtime_mechanisms.get("current_pinned_buffer_mb"),
        "runtime_pinned_buffer_peak_mb": runtime_mechanisms.get("pinned_buffer_peak_mb"),
        "runtime_prefetch_started_count": runtime_mechanisms.get("prefetch_started_count"),
        "runtime_prefetch_wait_count": runtime_mechanisms.get("prefetch_wait_count"),
        "runtime_prefetch_bytes_mb": runtime_mechanisms.get("prefetch_mb"),
        "runtime_prefetch_oom_count": runtime_mechanisms.get("prefetch_oom_count"),
        "runtime_sync_load_count": runtime_mechanisms.get("sync_load_count"),
        "runtime_sync_load_bytes_mb": runtime_mechanisms.get("sync_load_mb"),
        "runtime_offload_to_cpu_count": runtime_mechanisms.get("offload_to_cpu_count"),
        "runtime_release_gpu_tensor_count": runtime_mechanisms.get("release_gpu_tensor_count"),
        "runtime_prefetch_link_count": runtime_mechanisms.get("prefetch_link_count"),
        "runtime_mechanism_counters": runtime_mechanisms,
        "load_ms": load_ms,
        "prepare_ms": prepare_ms,
        "forward_ms_mean": total_ms_mean,
        "forward_ms_min": min(times),
        "forward_ms_samples": list(times),
        "prefill_ms_samples": list(prefill_times),
        "decode_ms_samples": list(decode_times),
        "prefill_ms_mean": prefill_ms_mean,
        "decode_ms_mean": decode_ms_mean if args.workload == "generation" else None,
        "decode_ms_per_token": decode_ms_mean / max(1, args.decode_steps) if args.workload == "generation" else None,
        "tokens_per_s": simulated_tokens * 1000.0 / total_ms_mean,
        "decode_tokens_per_s": decode_tokens * 1000.0 / decode_ms_mean if decode_ms_mean > 0 else None,
        "profiled_layers": max(profile_layer_counts) if profile_layer_counts else 0,
        "profile_summary": merged_profile,
        "profile_top_events": top_profile_events(merged_profile),
        "profile_total_ms": profiled_total_ms,
        "profile_total_pct_of_forward": 100.0 * profiled_total_ms / max(total_ms_mean * max(1, args.repeat), 1e-9),
        "module_timing": module_timing_summary,
        "semantic_output_probe": {
            "enabled": bool(semantic_probe_rows),
            "rows": semantic_probe_rows,
        },
        "layer_peak_trace": {
            "enabled": bool(layer_peak_trace_rows),
            "rows": layer_peak_trace_rows,
        },
        "load_peak_mb": load_peak,
        "prepare_peak_mb": prepare_peak,
        "forward_peak_mb": max(v for v in peaks if v is not None) if device.type == "cuda" else None,
        "forward_peak_mb_samples": list(peaks),
        "resident_after_prepare_mb": resident_mb,
        "logits_shape": list(logits.shape),
        "logits_mean_abs": torch.mean(torch.abs(logits)).item(),
        "input_ids_sha256": hashlib.sha256(
            input_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
        ).hexdigest(),
        "input_ids_preview": input_ids.detach().reshape(-1)[:16].to(device="cpu").tolist(),
        "device": str(device),
        "cuda_memory_fraction": float(args.cuda_memory_fraction or 0.0),
        "cuda_memory_limit": cuda_memory_limit,
        "layer_buffer_accounting": layer_buffer_accounting,
    }
    out.update(cls_metrics)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
"""


VMM_LOWP_CHOICES = ("auto", "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2", "int8")
VMM_LOWP_STABLE_DTYPES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}
VMM_LOWP_EXPERIMENTAL_8BIT = {"fp8_e4m3", "fp8_e5m2", "int8"}


def apply_vmm_lowp_format_defaults(args, *, argv_has_option):
    fmt = getattr(args, "vmm_lowp_format", "auto")
    if fmt in (None, "auto"):
        args.vmm_lowp_format = "auto"
        args.vmm_lowp_status = "auto_maps_to_compute_dtype"
        return args
    if fmt in VMM_LOWP_STABLE_DTYPES:
        dtype = VMM_LOWP_STABLE_DTYPES[fmt]
        if not argv_has_option("--conductance-dtype"):
            args.conductance_dtype = dtype
        if not argv_has_option("--compute-dtype"):
            args.compute_dtype = dtype
        if not argv_has_option("--mode0-vmm-compute-dtype"):
            args.mode0_vmm_compute_dtype = "auto"
        args.vmm_lowp_status = f"mapped_to_{dtype}"
        return args
    if fmt in VMM_LOWP_EXPERIMENTAL_8BIT:
        args.vmm_lowp_status = "unsupported_needs_scaled_8bit_kernel"
        return args
    raise ValueError(f"Unsupported --vmm-lowp-format: {fmt}")


def unsupported_8bit_vmm_reason(fmt):
    return (
        f"--vmm-lowp-format {fmt} is an experimental interface only. "
        "Current Memintelli Triton VMM kernels support FP32/FP16/BF16 DOT_DTYPE only; "
        "FP8/INT8 needs a scaled current-preserving kernel that de-scales before ADC. "
        "Use examples/21_probe_vmm_lowp_support.py for capability probing, or choose "
        "--vmm-lowp-format bf16/fp16/fp32 for runnable benchmarks."
    )


def _trim_text(text: str | bytes | None, limit: int = 8000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extra_value(extra: list[str], flag: str, default: str | None = None) -> str | None:
    try:
        idx = extra.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(extra):
        return default
    return extra[idx + 1]


def _monitor_gpu(args: argparse.Namespace) -> str | None:
    if args.monitor_gpu:
        return args.monitor_gpu
    if args.cuda_visible_devices:
        return args.cuda_visible_devices.split(",")[0].strip()
    if args.device.startswith("cuda:"):
        return args.device.split(":", 1)[1]
    if args.device == "cuda":
        return "0"
    return None


class NvidiaSmiMonitor:
    def __init__(self, gpu_index: str | None, interval: float, enabled: bool):
        self.gpu_indices = [item.strip() for item in str(gpu_index).split(",") if item.strip()] if gpu_index is not None else []
        self.gpu_index = ",".join(self.gpu_indices) if self.gpu_indices else None
        self.interval = interval
        self.enabled = enabled and bool(self.gpu_indices)
        self.samples: list[dict[str, float]] = []
        self.static_info = []
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.enabled:
            self._load_static_info()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2))

    def _loop(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=5)
                if proc.returncode == 0:
                    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
                    for idx, line in enumerate(lines):
                        vals = [item.strip() for item in line.split(",")]
                        if len(vals) < 5:
                            continue
                        gpu_index = self.gpu_indices[idx] if idx < len(self.gpu_indices) else str(idx)
                        self.samples.append({
                            "gpu_index": gpu_index,
                            "gpu_util_pct": float(vals[0]),
                            "mem_util_pct": float(vals[1]),
                            "mem_used_mb": float(vals[2]),
                            "mem_total_mb": float(vals[3]),
                            "power_w": float(vals[4]),
                        })
                elif not self.error:
                    self.error = _trim_text(proc.stderr, 400)
            except Exception as exc:
                if not self.error:
                    self.error = str(exc)
            self._stop.wait(self.interval)

    def _load_static_info(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=5)
            if proc.returncode != 0:
                if not self.error:
                    self.error = _trim_text(proc.stderr, 400)
                return
            for line in [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]:
                vals = [item.strip() for item in line.split(",", 3)]
                if len(vals) < 4:
                    continue
                try:
                    mem_total = float(vals[3])
                except ValueError:
                    mem_total = None
                self.static_info.append({
                    "gpu_index": vals[0],
                    "gpu_name": vals[1],
                    "gpu_uuid": vals[2],
                    "mem_total_mb": mem_total,
                })
        except Exception as exc:
            if not self.error:
                self.error = str(exc)

    def summary(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        if not self.samples:
            out = {"enabled": True, "gpu_index": self.gpu_index, "samples": 0, "error": self.error}
            if self.static_info:
                out["gpu_static"] = self.static_info
                if len(self.static_info) == 1:
                    out.update(self.static_info[0])
            return out
        out = {"enabled": True, "gpu_index": self.gpu_index, "samples": len(self.samples), "error": self.error}
        if self.static_info:
            out["gpu_static"] = self.static_info
            if len(self.static_info) == 1:
                out.update(self.static_info[0])
        for key in ("gpu_util_pct", "mem_util_pct", "mem_used_mb", "power_w"):
            vals = [sample[key] for sample in self.samples]
            out[f"{key}_mean"] = sum(vals) / len(vals)
            out[f"{key}_max"] = max(vals)
        per_gpu = {}
        for gpu_index in self.gpu_indices:
            gpu_samples = [sample for sample in self.samples if sample.get("gpu_index") == gpu_index]
            if not gpu_samples:
                continue
            gpu_out = {"samples": len(gpu_samples)}
            for key in ("gpu_util_pct", "mem_util_pct", "mem_used_mb", "power_w"):
                vals = [sample[key] for sample in gpu_samples]
                gpu_out[f"{key}_mean"] = sum(vals) / len(vals)
                gpu_out[f"{key}_max"] = max(vals)
            per_gpu[gpu_index] = gpu_out
        if per_gpu:
            out["per_gpu"] = per_gpu
        return out


class ProcessMemoryMonitor:
    def __init__(self, proc, interval: float):
        self.proc = proc
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2))

    @staticmethod
    def _read_proc_status(pid: int) -> dict[str, float]:
        out = {}
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmSwap:", "VmPin:")):
                        key, rest = line.split(":", 1)
                        parts = rest.strip().split()
                        if parts:
                            out[key.rstrip(":")] = float(parts[0]) / 1024.0
        except FileNotFoundError:
            pass
        return out

    @staticmethod
    def _read_meminfo() -> dict[str, float]:
        out = {}
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
                        key, rest = line.split(":", 1)
                        parts = rest.strip().split()
                        if parts:
                            out[key.rstrip(":")] = float(parts[0]) / 1024.0
        except FileNotFoundError:
            pass
        return out

    def _loop(self):
        while not self._stop.is_set():
            try:
                sample = {"pid": float(self.proc.pid)}
                sample.update(self._read_proc_status(self.proc.pid))
                sample.update(self._read_meminfo())
                if len(sample) > 1:
                    self.samples.append(sample)
                if self.proc.poll() is not None:
                    break
            except Exception as exc:
                if not self.error:
                    self.error = str(exc)
            self._stop.wait(self.interval)

    def summary(self) -> dict:
        out = {"enabled": True, "pid": self.proc.pid, "samples": len(self.samples), "error": self.error}
        for key in ("VmRSS", "VmHWM", "VmSize", "VmSwap", "VmPin", "MemAvailable", "SwapFree"):
            vals = [sample[key] for sample in self.samples if key in sample]
            if vals:
                out[f"{key}_mb_mean"] = sum(vals) / len(vals)
                out[f"{key}_mb_max"] = max(vals)
                out[f"{key}_mb_min"] = min(vals)
        return out


def apply_launcher_execution_mode_defaults(args):
    """Map mode names to state placement only; S2 owns compute policy."""
    memory_prepare_policy = getattr(args, "memory_prepare_policy", "lazy_release")
    memory_budget_mb = float(getattr(args, "memory_budget_mb", 0.0) or 0.0)
    resident_budget_mb = float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0)
    if memory_budget_mb > 0.0:
        if resident_budget_mb > 0.0 and abs(resident_budget_mb - memory_budget_mb) > 1e-6:
            raise ValueError("--memory-budget-mb and --memory-resident-budget-mb must match when both are set.")
        args.memory_resident_budget_mb = memory_budget_mb
    args.memory_budget_mb = float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0)
    if not hasattr(args, "state_resident_budget_mb"):
        args.state_resident_budget_mb = resident_budget_mb
    if args.execution_mode == "speed":
        args.state_resident_budget_mb = -1.0
        args.streaming = False
        args.lazy_prepare = False
        args.lazy_release_after_forward = False
    elif args.execution_mode == "balanced":
        args.state_resident_budget_mb = float(
            getattr(args, "state_resident_budget_mb", 0.0) or resident_budget_mb
        )
        args.streaming = True
        if memory_prepare_policy == "lazy_release":
            args.memory_prepare_policy = "eager_streaming"
            memory_prepare_policy = args.memory_prepare_policy
        args.lazy_prepare = False
        args.lazy_release_after_forward = False
        args.free_weights = True
    elif args.execution_mode == "memory":
        args.state_resident_budget_mb = 0.0
        args.streaming = True
        if memory_prepare_policy == "eager_streaming":
            args.lazy_prepare = False
            args.lazy_release_after_forward = False
            args.free_weights = True
        elif memory_prepare_policy == "lazy_keep":
            args.lazy_prepare = True
            args.lazy_release_after_forward = False
            args.free_weights = False
        else:
            args.lazy_prepare = True
            args.lazy_release_after_forward = True
            args.free_weights = False
        if args.memory_runtime_diagnostic:
            args.lazy_prepare = False
    return args


def plan_linear_output_blocks(args, child, manual_shard_count=0):
    stage = str(getattr(args, "s1_stage", "off") or "off")
    block_addressable = bool(getattr(args, "s1_block_addressable", stage != "off"))
    if not block_addressable:
        return argparse.Namespace(
            output_block_cols=int(child.out_features),
            shard_count=1,
            estimated_peak_mb=0.0,
            workspace_budget_mb=0.0,
            base_allocated_mb=0.0,
            resident_state_mb=0.0,
            safety_margin_mb=0.0,
            manual_override=False,
        )

    from memintelli.NN_layers.state_planner import plan_output_block

    planner_enabled = str(getattr(args, "state_planner", "off")) == "analytical"
    manual_cols = int(getattr(args, "output_block_cols", 0) or 0) if block_addressable else 0
    if block_addressable and not manual_cols and int(manual_shard_count or 0) > 1:
        manual_cols = math.ceil(int(child.out_features) / int(manual_shard_count))
    cuda_budget_mb = (
        float(getattr(args, "cuda_peak_budget_mb", 0.0) or 0.0)
        if block_addressable and planner_enabled
        else 0.0
    )
    resident_mb = max(0.0, float(getattr(args, "state_resident_budget_mb", 0.0) or 0.0))
    weight_tiles = tuple(int(v) for v in getattr(args, "weight_paral_size", (64, 64)))
    input_slices = getattr(args, "input_slice", (1,))
    weight_slices = getattr(args, "weight_slice", (1,))
    read_variation = float(getattr(args, "read_variation", 0.0) or 0.0)
    vmm_dtype = str(getattr(args, "mode0_vmm_compute_dtype", "auto") or "auto")
    if vmm_dtype == "auto":
        vmm_dtype = str(getattr(args, "compute_dtype", "float32") or "float32")
    grouped_noisy_vmm = read_variation > 0.0 and vmm_dtype in {"float16", "bfloat16"}
    return plan_output_block(
        tokens=max(1, int(getattr(args, "batch", 1))) * max(1, int(getattr(args, "seq", 1))),
        in_features=int(child.in_features),
        out_features=int(child.out_features),
        input_slices=max(1, len(input_slices)),
        weight_slices=max(1, len(weight_slices)),
        array_rows=weight_tiles[0],
        array_cols=weight_tiles[1],
        cuda_peak_budget_mb=cuda_budget_mb,
        base_allocated_mb=max(0.0, float(getattr(args, "planner_base_allocated_mb", 0.0) or 0.0)),
        resident_state_mb=resident_mb,
        safety_margin_mb=max(0.0, float(getattr(args, "planner_safety_margin_mb", 256.0) or 0.0)),
        manual_output_block_cols=manual_cols,
        read_variation=read_variation,
        seeded_read_noise=getattr(args, "read_variation_seed", None) is not None,
        grouped_noisy_vmm=grouped_noisy_vmm,
    )


def apply_s1_stage(args):
    """Select one state-management level without changing compute policy."""
    stage = getattr(args, "s1_stage", "budgeted")
    if stage not in {"off", "block", "budgeted"}:
        raise ValueError(f"unsupported S1 stage: {stage}")
    args.s1_stage = stage
    args.s1_block_addressable = stage in {"block", "budgeted"}
    if stage == "off":
        args.state_planner = "off"
        args.state_resident_budget_mb = -1.0
    elif stage == "block":
        args.state_planner = "off"
        args.state_resident_budget_mb = 0.0
    else:
        args.state_planner = "analytical"
        args.state_resident_budget_mb = float(
            getattr(args, "state_resident_budget_mb", 0.0) or 0.0
        )
    if float(getattr(args, "state_resident_budget_mb", -1.0) or 0.0) >= 0.0:
        args.memory_resident_budget_mb = float(args.state_resident_budget_mb)
    return args


def apply_s2_stage(args):
    """Select intra-Linear and inter-Linear semantics-preserving compaction."""
    stage = getattr(args, "s2_stage", None)
    if stage is None:
        stage = getattr(args, "s2_ablation_stage", "full")
    if stage == "grouped":
        stage = "intra"
    if stage not in {"off", "intra", "full"}:
        raise ValueError(f"unsupported S2 stage: {stage}")
    args.s2_stage = stage
    args.s2_ablation_stage = stage

    fast_inference = stage != "off"
    args.triton_fuse_restored_input_slices = fast_inference
    args.triton_direct_final_output = fast_inference
    args.triton_gidx_direct_final_output = False
    args.triton_direct_final_exact_reduce = fast_inference
    args.triton_overlap_restore_direct = False
    args.triton_precompute_input_voltage = False
    args.triton_fast_adc_scale = False
    args.triton_direct_output_zero_once = False
    args.fuse_mlp_gate_up = stage == "full"
    args.fuse_common_input_projections = stage == "full"
    args.triton_activation_slice_cache = False
    if getattr(args, "fast_inference_backend", "auto") == "auto":
        args.fast_inference_backend = "triton_gidx" if fast_inference else "torch"
    return args, fast_inference


def apply_s2_ablation_stage(args):
    """Backward-compatible alias for pre-reconstruction experiment scripts."""
    return apply_s2_stage(args)


def failure_row(
    args,
    label,
    kind,
    status,
    error,
    extra=None,
    monitor=None,
    host_monitor=None,
    returncode=None,
    stdout=None,
    stderr=None,
    cmd=None,
):
    extra = extra or []
    mode = int(_extra_value(extra, "--mode", "0")) if kind == "v3" else 0
    input_mode = _extra_value(extra, "--mode2-input-mode", "signed")
    backend = _extra_value(extra, "--fast-inference-backend", args.fast_inference_backend) if kind == "v3" else "n/a"
    return {
        "status": status,
        "error": _trim_text(error),
        "returncode": returncode,
        "stdout_tail": _trim_text(stdout),
        "stderr_tail": _trim_text(stderr),
        "cmd_tail": list(cmd[-24:]) if cmd else [],
        "label": label,
        "kind": kind,
        "model_path": args.model_path,
        "batch": args.batch,
        "seq": args.seq,
        "workload": args.workload,
        "decode_steps": args.decode_steps if args.workload == "generation" else 0,
        "generation_token_ids": [],
        "generation_token_ids_by_step": [],
        "generation_token_agreement": None,
        "generation_sequence_match": None,
        "classification_task": args.classification_task,
        "classification_examples_json": args.classification_examples_json,
        "classification_choice_labels": list(args.classification_choice_labels),
        "classification_example_ids": [],
        "classification_gold_labels": [],
        "classification_token_ids": list(args.classification_token_ids or []),
        "classification_proxy": False,
        "classification_pred_token_ids": [],
        "classification_pred_labels": [],
        "classification_accuracy_proxy": None,
        "classification_margin_mean": None,
        "classification_entropy_mean": None,
        "classification_top_score_mean": None,
        "simulate_lm_head": args.simulate_lm_head,
        "only_lm_head": args.only_lm_head,
        "always_include_lm_head": args.always_include_lm_head,
        "linear_include_regex": list(args.linear_include_regex),
        "linear_exclude_regex": list(args.linear_exclude_regex),
        "load_on_device": args.load_on_device,
        "lm_head_output_shards": args.lm_head_output_shards,
        "lm_head_shard_devices": args.lm_head_shard_devices,
        "lm_head_shard_parallel": args.lm_head_shard_parallel,
        "lm_head_input_select": getattr(args, "lm_head_input_select", "all"),
        "lm_head_output_select": getattr(args, "lm_head_output_select", "all"),
        "lm_head_output_token_ids": list(getattr(args, "lm_head_output_token_ids", []) or []),
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": args.torch_compile_mode if args.torch_compile else "none",
        "torch_compile_backend": args.torch_compile_backend if args.torch_compile else "none",
        "torch_compile_dynamic": bool(args.torch_compile_dynamic) if args.torch_compile else False,
        "torch_compile_exclude_linearmem": (
            bool(getattr(args, "torch_compile_exclude_linearmem", False)) if args.torch_compile else False
        ),
        "torch_compile_exclude_linearmem_info": {
            "enabled": False,
            "reason": "failed_before_worker_report",
            "wrapped_modules": 0,
        },
        "torch_compile_wrap_ms": None,
        "model_loader": args.model_loader,
        "language_model_only": args.language_model_only,
        "fuse_common_input_projections": bool(getattr(args, "fuse_common_input_projections", False)) if kind == "v3" else False,
        "execution_mode": args.execution_mode or "manual",
        "memory_prepare_policy": args.memory_prepare_policy,
        "memory_runtime_diagnostic": bool(args.memory_runtime_diagnostic),
        "runtime_stage_timing": bool(args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
        "runtime_counters": bool(args.runtime_counters or args.runtime_stage_timing or args.memory_runtime_diagnostic or args.profile),
        "memory_budget_mb": float(getattr(args, "memory_budget_mb", 0.0) or 0.0),
        "memory_resident_budget_mb": float(getattr(args, "memory_resident_budget_mb", 0.0) or 0.0),
        "memory_resident_select": getattr(args, "memory_resident_select", "largest"),
        "streaming": bool(args.streaming) if kind != "hf" else False,
        "streaming_prefetch": bool(getattr(args, "streaming_prefetch", True)) if kind != "hf" else False,
        "streaming_prefetch_cycle": bool(getattr(args, "streaming_prefetch_cycle", False)) if kind != "hf" else False,
        "streaming_pin_policy": getattr(args, "streaming_pin_policy", "persistent") if kind != "hf" else "n/a",
        "streaming_persistent_pin_budget_mb": (
            float(getattr(args, "streaming_persistent_pin_budget_mb", 0.0) or 0.0) if kind != "hf" else 0.0
        ),
        "streaming_persistent_pin_select": (
            getattr(args, "streaming_persistent_pin_select", "sequential") if kind != "hf" else "n/a"
        ),
        "streaming_window_pin_cache_mb": (
            float(getattr(args, "streaming_window_pin_cache_mb", 0.0) or 0.0) if kind != "hf" else 0.0
        ),
        "streaming_pin_hints_json": getattr(args, "streaming_pin_hints_json", "") if kind != "hf" else "",
        "memory_empty_cache_after_offload": bool(getattr(args, "memory_empty_cache_after_offload", False)) if kind != "hf" else False,
        "memory_empty_cache_after_offload_interval": (
            max(1, int(getattr(args, "memory_empty_cache_after_offload_interval", 1) or 1))
            if kind != "hf"
            else 0
        ),
        "cuda_memory_fraction": float(args.cuda_memory_fraction or 0.0),
        "require_all_linears": bool(args.require_all_linears),
        "lazy_prepare": args.lazy_prepare,
        "lazy_release_after_forward": args.lazy_release_after_forward,
        "mode": mode,
        "mode2_input_mode": input_mode if kind == "v3" and mode == 2 else "n/a",
        "fast_inference": "--fast-inference" in extra if kind == "v3" else False,
        "fast_inference_backend": backend,
        "g_level": args.g_level if kind == "v3" else None,
        "write_variation": args.write_variation if kind == "v3" else 0.0,
        "read_variation": args.read_variation if kind == "v3" else 0.0,
        "read_variation_seed": args.read_variation_seed if kind == "v3" else None,
        "vnoise": args.vnoise if kind == "v3" else 0.0,
        "write_variation_mode": args.write_variation_mode if kind == "v3" else "n/a",
        "vmm_lowp_format": getattr(args, "vmm_lowp_format", "auto") if kind == "v3" else "n/a",
        "vmm_lowp_status": getattr(args, "vmm_lowp_status", "n/a") if kind == "v3" else "n/a",
        "conductance_dtype": args.conductance_dtype if kind == "v3" else "n/a",
        "requested_compute_dtype": args.compute_dtype if kind == "v3" else "n/a",
        "compute_dtype": args.compute_dtype if kind == "v3" else "n/a",
        "adc_compute_dtype": "torch.float32" if kind == "v3" else "n/a",
        "linear_output_dtype": args.linear_output_dtype if kind == "v3" else "n/a",
        "mode0_semantic_policy": args.mode0_semantic_policy if kind == "v3" else "n/a",
        "mode0_vmm_compute_dtype": args.mode0_vmm_compute_dtype if kind == "v3" else "n/a",
        "rdac": args.rdac if kind == "v3" else None,
        "radc": args.radc if kind == "v3" else None,
        "vread": args.vread if kind == "v3" else None,
        "input_slice": list(args.input_slice) if kind == "v3" else [],
        "weight_slice": list(args.weight_slice) if kind == "v3" else [],
        "mode2_input_slice": list(args.mode2_input_slice) if kind == "v3" else [],
        "mode2_weight_slice": list(args.mode2_weight_slice) if kind == "v3" else [],
        "input_paral_size": list(args.input_paral_size) if kind == "v3" else [],
        "weight_paral_size": list(args.weight_paral_size) if kind == "v3" else [],
        "input_quant_gran": list(args.input_quant_gran) if kind == "v3" else [],
        "weight_quant_gran": list(args.weight_quant_gran) if kind == "v3" else [],
        "triton_input_precision": args.triton_input_precision if kind == "v3" else "n/a",
        "triton_block_r": args.triton_block_r if kind == "v3" else None,
        "triton_block_l": args.triton_block_l if kind == "v3" else None,
        "triton_block_k": args.triton_block_k if kind == "v3" else None,
        "triton_output_chunk_limit": args.triton_output_chunk_limit if kind == "v3" else None,
        "triton_auto_config": bool(getattr(args, "triton_auto_config", False)) if kind == "v3" else False,
        "triton_mode0_input_tile_group": int(getattr(args, "triton_mode0_input_tile_group", 1)) if kind == "v3" else 1,
        "triton_gidx_read_noise": bool(args.triton_gidx_read_noise) if kind == "v3" else False,
        "triton_gidx_fused_restore_read_noise": bool(args.triton_gidx_fused_restore_read_noise) if kind == "v3" else False,
        "triton_gidx_restore_block": int(getattr(args, "triton_gidx_restore_block", 512)) if kind == "v3" else 0,
        "triton_gidx_restore_block_auto": bool(getattr(args, "triton_gidx_restore_block_auto", True)) if kind == "v3" else False,
        "triton_gidx_restore_small_block": int(getattr(args, "triton_gidx_restore_small_block", 128)) if kind == "v3" else 0,
        "triton_gidx_restore_auto_in_features_threshold": (
            int(getattr(args, "triton_gidx_restore_auto_in_features_threshold", 4096)) if kind == "v3" else 0
        ),
        "triton_gidx_restore_num_warps": int(getattr(args, "triton_gidx_restore_num_warps", 4)) if kind == "v3" else 0,
        "triton_gidx_restore_strided": bool(getattr(args, "triton_gidx_restore_strided", True)) if kind == "v3" else False,
        "triton_gidx_restore_m_slab": bool(getattr(args, "triton_gidx_restore_m_slab", True)) if kind == "v3" else False,
        "triton_gidx_restore_approx_linear_noise": (
            bool(getattr(args, "triton_gidx_restore_approx_linear_noise", False)) if kind == "v3" else False
        ),
        "triton_gidx_restore_exp2_noise": (
            bool(getattr(args, "triton_gidx_restore_exp2_noise", False)) if kind == "v3" else False
        ),
        "triton_gidx_restore_fast_noise": (
            bool(getattr(args, "triton_gidx_restore_fast_noise", False)) if kind == "v3" else False
        ),
        "triton_gidx_fuse_input_slices": bool(args.triton_gidx_fuse_input_slices) if kind == "v3" else False,
        "triton_mode0_strict_intermediate": (
            bool(getattr(args, "triton_mode0_strict_intermediate", False)) if kind == "v3" else False
        ),
        "triton_mode0_strict_intermediate_backend": (
            str(getattr(args, "triton_mode0_strict_intermediate_backend", "off")) if kind == "v3" else "n/a"
        ),
        "triton_reuse_input_voltage": bool(args.triton_reuse_input_voltage) if kind == "v3" else False,
        "triton_reuse_weight_tile": bool(getattr(args, "triton_reuse_weight_tile", False)) if kind == "v3" else False,
        "triton_precompute_input_voltage": bool(getattr(args, "triton_precompute_input_voltage", False)) if kind == "v3" else False,
        "triton_fast_adc_scale": bool(getattr(args, "triton_fast_adc_scale", False)) if kind == "v3" else False,
        "triton_fuse_restored_input_slices": bool(args.triton_fuse_restored_input_slices) if kind == "v3" else False,
        "triton_fuse_activation_slices": bool(args.triton_fuse_activation_slices) if kind == "v3" else False,
        "triton_reuse_activation_slice_buffer": bool(args.triton_reuse_activation_slice_buffer) if kind == "v3" else False,
        "triton_probe_activation_density": bool(getattr(args, "triton_probe_activation_density", False)) if kind == "v3" else False,
        "triton_binary_input_slice_dac": bool(args.triton_binary_input_slice_dac) if kind == "v3" else False,
        "triton_direct_final_num_warps": int(getattr(args, "triton_direct_final_num_warps", 4)) if kind == "v3" else 0,
        "triton_direct_final_partial_m_group": (
            int(getattr(args, "triton_direct_final_partial_m_group", 0)) if kind == "v3" else 0
        ),
        "triton_direct_final_exact_reduce": (
            bool(getattr(args, "triton_direct_final_exact_reduce", False)) if kind == "v3" else False
        ),
        "triton_fuse_output_finalize": bool(args.triton_fuse_output_finalize) if kind == "v3" else False,
        "triton_direct_final_output": bool(args.triton_direct_final_output) if kind == "v3" else False,
        "triton_direct_output_zero_once": bool(args.triton_direct_output_zero_once) if kind == "v3" else False,
        "triton_gidx_direct_final_output": bool(args.triton_gidx_direct_final_output) if kind == "v3" else False,
        "triton_gidx_direct_final_deterministic": (
            bool(getattr(args, "triton_gidx_direct_final_deterministic", False)) if kind == "v3" else False
        ),
        "triton_overlap_restore_direct": bool(args.triton_overlap_restore_direct) if kind == "v3" else False,
        "triton_cross_linear_restore_prefetch": (
            bool(getattr(args, "triton_cross_linear_restore_prefetch", False)) if kind == "v3" else False
        ),
        "triton_mode1_gidx_direct_final": bool(args.triton_mode1_gidx_direct_final) if kind == "v3" else False,
        "triton_mode1_input_tile_group": int(args.triton_mode1_input_tile_group) if kind == "v3" else 1,
        "triton_mode1_chunked_direct_final": bool(args.triton_mode1_chunked_direct_final) if kind == "v3" else False,
        "triton_mode2_diff_direct_final": bool(args.triton_mode2_diff_direct_final) if kind == "v3" else False,
        "triton_mode2_diff_presubtract": bool(args.triton_mode2_diff_presubtract) if kind == "v3" else False,
        "triton_mode2_diff_fuse_input_slices": bool(args.triton_mode2_diff_fuse_input_slices) if kind == "v3" else False,
        "triton_mode2_diff_block_r_cap": int(args.triton_mode2_diff_block_r_cap) if kind == "v3" else 0,
        "triton_mode2_diff_block_l_cap": int(args.triton_mode2_diff_block_l_cap) if kind == "v3" else 0,
        "mode1_grouped_tile_gemm": bool(args.mode1_grouped_tile_gemm) if kind == "v3" else False,
        "direct_output_chunk_write": bool(args.direct_output_chunk_write) if kind == "v3" else False,
        "profile": bool(args.profile) if kind != "hf" else False,
        "forward_ms_mean": None,
        "prefill_ms_mean": None,
        "decode_ms_mean": None,
        "decode_ms_per_token": None,
        "tokens_per_s": None,
        "decode_tokens_per_s": None,
        "forward_peak_mb": None,
        "resident_after_prepare_mb": None,
        "g_storage_mb": None,
        "g_gpu_mb": None,
        "g_cpu_mb": None,
        "g_cpu_pinned_mb": None,
        "replaced_non_lm_head_linears": None,
        "replaced_lm_head_linears": None,
        "full_model_deployment_valid": False,
        "original_linear_count": None,
        "remaining_linear_count": None,
        "replacement_coverage": None,
        "lm_head_simulated": None,
        "profiled_layers": 0,
        "profile_summary": {},
        "profile_top_events": [],
        "profile_total_ms": None,
        "profile_total_pct_of_forward": None,
        "logits_path": "",
        "agreement_ref_label": "",
        "agreement_snr_db": None,
        "agreement_mae": None,
        "agreement_max_abs_err": None,
        "agreement_top1": None,
        "monitor": monitor or {},
        "host_monitor": host_monitor or {},
    }


def run_worker(args, repo: Path | None, label: str, kind: str, extra: list[str]) -> dict:
    env = os.environ.copy()
    if repo is not None:
        env["PYTHONPATH"] = str(repo)
    if args.ld_library_path:
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = args.ld_library_path + (":" + old_ld if old_ld else "")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    logits_dir = Path(args.logits_dir) if args.logits_dir else Path(tempfile.mkdtemp(prefix="memintelli_logits_"))
    logits_dir.mkdir(parents=True, exist_ok=True)
    logits_path = logits_dir / f"{label}.pt"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(WORKER_CODE)
        worker_path = f.name

    mode = int(_extra_value(extra, "--mode", "0")) if kind == "v3" else 0
    triton_block_r = int(args.triton_block_r)
    triton_block_l = int(args.triton_block_l)
    triton_block_k = int(args.triton_block_k)
    triton_output_chunk_limit = int(args.triton_output_chunk_limit)
    worker_conductance_dtype = args.conductance_dtype
    worker_compute_dtype = args.compute_dtype
    lowp_sets_vmm_dtype = getattr(args, "vmm_lowp_format", "auto") in VMM_LOWP_STABLE_DTYPES
    if kind == "v3" and args.execution_mode in {"speed", "balanced", "memory"}:
        if not bool(getattr(args, "_conductance_dtype_user_set", False)) and not lowp_sets_vmm_dtype:
            worker_conductance_dtype = "bfloat16"
        if not bool(getattr(args, "_compute_dtype_user_set", False)) and not lowp_sets_vmm_dtype:
            worker_compute_dtype = "bfloat16"
    if kind == "v3" and getattr(args, "vmm_lowp_format", "auto") in VMM_LOWP_EXPERIMENTAL_8BIT:
        try:
            os.remove(worker_path)
        except OSError:
            pass
        return failure_row(
            args,
            label,
            kind,
            "unsupported",
            unsupported_8bit_vmm_reason(args.vmm_lowp_format),
            extra,
            monitor={},
            host_monitor={},
            returncode="unsupported_8bit_vmm",
            cmd=[],
        )
    if kind == "v3" and bool(getattr(args, "mode_aware_triton_defaults", False)):
        if mode == 0:
            if args.execution_mode == "speed":
                if not bool(getattr(args, "_triton_block_r_user_set", False)):
                    triton_block_r = int(getattr(args, "mode0_triton_block_r", triton_block_r))
                if not bool(getattr(args, "_triton_block_l_user_set", False)):
                    triton_block_l = int(getattr(args, "mode0_triton_block_l", triton_block_l))
                if not bool(getattr(args, "_triton_output_chunk_limit_user_set", False)):
                    triton_output_chunk_limit = int(getattr(args, "mode0_triton_output_chunk_limit", triton_output_chunk_limit))
        elif mode == 1:
            if not bool(getattr(args, "_triton_block_r_user_set", False)):
                triton_block_r = int(getattr(args, "mode1_triton_block_r", triton_block_r))
            if not bool(getattr(args, "_triton_output_chunk_limit_user_set", False)):
                triton_output_chunk_limit = int(getattr(args, "mode1_triton_output_chunk_limit", triton_output_chunk_limit))
        elif mode == 2:
            if not bool(getattr(args, "_triton_block_l_user_set", False)):
                triton_block_l = int(getattr(args, "mode2_triton_block_l", triton_block_l))
            if not bool(getattr(args, "_triton_block_k_user_set", False)):
                triton_block_k = int(getattr(args, "mode2_triton_block_k", triton_block_k))

    cmd = [
        args.python,
        worker_path,
        "--kind", kind,
        "--label", label,
        "--model-path", args.model_path,
        "--model-loader", args.model_loader,
        "--device", args.device,
        "--dtype", args.dtype,
        "--batch", str(args.batch),
        "--seq", str(args.seq),
        "--seed", str(args.seed),
        "--repeat", str(args.repeat),
        "--warmup", str(args.warmup),
        "--torch-compile" if args.torch_compile else "--no-torch-compile",
        "--torch-compile-mode", args.torch_compile_mode,
        "--torch-compile-backend", args.torch_compile_backend,
        "--torch-compile-dynamic" if args.torch_compile_dynamic else "--no-torch-compile-dynamic",
        "--torch-compile-exclude-linearmem" if args.torch_compile_exclude_linearmem else "--no-torch-compile-exclude-linearmem",
        "--cuda-profiler-capture" if args.cuda_profiler_capture else "--no-cuda-profiler-capture",
        "--workload", args.workload,
        "--decode-steps", str(args.decode_steps),
        "--classification-task", args.classification_task,
        "--classification-choice-labels", *map(str, args.classification_choice_labels),
        "--classification-candidate-count", str(args.classification_candidate_count),
        "--classification-scoring", args.classification_scoring,
        "--classification-score-batch-size", str(args.classification_score_batch_size),
        "--save-logits", str(logits_path),
        "--memory-prepare-policy", args.memory_prepare_policy,
        "--memory-budget-mb", str(args.memory_budget_mb),
        "--memory-resident-budget-mb", str(args.memory_resident_budget_mb),
        "--memory-resident-select", args.memory_resident_select,
        "--inference-chunk-size", str(args.inference_chunk_size),
        "--cuda-memory-fraction", str(args.cuda_memory_fraction),
        "--layer-buffer-accounting-limit", str(args.layer_buffer_accounting_limit),
        "--hgs", str(args.hgs),
        "--lgs", str(args.lgs),
        "--g-level", str(args.g_level),
        "--write-variation", str(args.write_variation),
        "--read-variation", str(args.read_variation),
        "--vnoise", str(args.vnoise),
        "--write-variation-mode", args.write_variation_mode,
        "--conductance-dtype", worker_conductance_dtype,
        "--compute-dtype", worker_compute_dtype,
        "--linear-output-dtype", args.linear_output_dtype,
        "--vmm-lowp-format", args.vmm_lowp_format,
        "--mode0-semantic-policy", args.mode0_semantic_policy,
        "--mode0-vmm-compute-dtype", args.mode0_vmm_compute_dtype,
        "--rdac", str(args.rdac),
        "--radc", str(args.radc),
        "--vread", str(args.vread),
        "--rate-stuck-hgs", str(args.rate_stuck_hgs),
        "--rate-stuck-lgs", str(args.rate_stuck_lgs),
        "--input-slice", *map(str, args.input_slice),
        "--weight-slice", *map(str, args.weight_slice),
        "--mode2-input-slice", *map(str, args.mode2_input_slice),
        "--mode2-weight-slice", *map(str, args.mode2_weight_slice),
        "--input-paral-size", *map(str, args.input_paral_size),
        "--weight-paral-size", *map(str, args.weight_paral_size),
        "--input-quant-gran", *map(str, args.input_quant_gran),
        "--weight-quant-gran", *map(str, args.weight_quant_gran),
        "--fast-inference-backend", args.fast_inference_backend,
        "--triton-input-precision", args.triton_input_precision,
        "--triton-block-r", str(triton_block_r),
        "--triton-block-l", str(triton_block_l),
        "--triton-block-k", str(triton_block_k),
        "--triton-output-chunk-limit", str(triton_output_chunk_limit),
        *extra,
    ]
    cmd.append("--triton-auto-config" if args.triton_auto_config else "--no-triton-auto-config")
    cmd.extend(["--triton-mode0-input-tile-group", str(args.triton_mode0_input_tile_group)])
    cmd.append("--triton-gidx-read-noise" if args.triton_gidx_read_noise else "--no-triton-gidx-read-noise")
    cmd.append("--triton-gidx-fused-restore-read-noise" if args.triton_gidx_fused_restore_read_noise else "--no-triton-gidx-fused-restore-read-noise")
    cmd.extend(["--triton-gidx-restore-block", str(args.triton_gidx_restore_block)])
    cmd.append("--triton-gidx-restore-block-auto" if args.triton_gidx_restore_block_auto else "--no-triton-gidx-restore-block-auto")
    cmd.extend(["--triton-gidx-restore-small-block", str(args.triton_gidx_restore_small_block)])
    cmd.extend([
        "--triton-gidx-restore-auto-in-features-threshold",
        str(args.triton_gidx_restore_auto_in_features_threshold),
    ])
    cmd.extend(["--triton-gidx-restore-num-warps", str(args.triton_gidx_restore_num_warps)])
    cmd.append("--triton-gidx-restore-strided" if args.triton_gidx_restore_strided else "--no-triton-gidx-restore-strided")
    cmd.append("--triton-gidx-restore-m-slab" if args.triton_gidx_restore_m_slab else "--no-triton-gidx-restore-m-slab")
    cmd.append(
        "--triton-gidx-restore-approx-linear-noise"
        if args.triton_gidx_restore_approx_linear_noise
        else "--no-triton-gidx-restore-approx-linear-noise"
    )
    cmd.append(
        "--triton-gidx-restore-exp2-noise"
        if args.triton_gidx_restore_exp2_noise
        else "--no-triton-gidx-restore-exp2-noise"
    )
    cmd.append(
        "--triton-gidx-restore-fast-noise"
        if args.triton_gidx_restore_fast_noise
        else "--no-triton-gidx-restore-fast-noise"
    )
    cmd.append("--triton-gidx-fuse-input-slices" if args.triton_gidx_fuse_input_slices else "--no-triton-gidx-fuse-input-slices")
    cmd.append(
        "--triton-mode0-strict-intermediate"
        if bool(getattr(args, "triton_mode0_strict_intermediate", False))
        else "--no-triton-mode0-strict-intermediate"
    )
    cmd.extend([
        "--triton-mode0-strict-intermediate-backend",
        str(getattr(args, "triton_mode0_strict_intermediate_backend", "auto")),
    ])
    cmd.append("--triton-reuse-input-voltage" if args.triton_reuse_input_voltage else "--no-triton-reuse-input-voltage")
    cmd.append("--triton-reuse-weight-tile" if args.triton_reuse_weight_tile else "--no-triton-reuse-weight-tile")
    cmd.append("--triton-precompute-input-voltage" if args.triton_precompute_input_voltage else "--no-triton-precompute-input-voltage")
    cmd.append("--triton-fast-adc-scale" if args.triton_fast_adc_scale else "--no-triton-fast-adc-scale")
    cmd.append("--triton-fuse-restored-input-slices" if args.triton_fuse_restored_input_slices else "--no-triton-fuse-restored-input-slices")
    cmd.append("--triton-fuse-activation-slices" if args.triton_fuse_activation_slices else "--no-triton-fuse-activation-slices")
    cmd.append("--triton-reuse-activation-slice-buffer" if args.triton_reuse_activation_slice_buffer else "--no-triton-reuse-activation-slice-buffer")
    cmd.append("--triton-probe-activation-slice-reuse" if args.triton_probe_activation_slice_reuse else "--no-triton-probe-activation-slice-reuse")
    cmd.append("--triton-probe-activation-density" if args.triton_probe_activation_density else "--no-triton-probe-activation-density")
    cmd.append("--triton-activation-slice-cache" if args.triton_activation_slice_cache else "--no-triton-activation-slice-cache")
    cmd.extend(["--triton-activation-slice-cache-max-entries", str(args.triton_activation_slice_cache_max_entries)])
    cmd.append("--triton-binary-input-slice-dac" if args.triton_binary_input_slice_dac else "--no-triton-binary-input-slice-dac")
    cmd.extend(["--triton-direct-final-num-warps", str(args.triton_direct_final_num_warps)])
    cmd.extend(["--triton-direct-final-partial-m-group", str(args.triton_direct_final_partial_m_group)])
    cmd.append(
        "--triton-direct-final-exact-reduce"
        if bool(getattr(args, "triton_direct_final_exact_reduce", False))
        else "--no-triton-direct-final-exact-reduce"
    )
    cmd.append("--triton-fuse-output-finalize" if args.triton_fuse_output_finalize else "--no-triton-fuse-output-finalize")
    cmd.append("--triton-direct-final-output" if args.triton_direct_final_output else "--no-triton-direct-final-output")
    cmd.append("--triton-direct-output-zero-once" if args.triton_direct_output_zero_once else "--no-triton-direct-output-zero-once")
    cmd.append("--triton-gidx-direct-final-output" if args.triton_gidx_direct_final_output else "--no-triton-gidx-direct-final-output")
    cmd.append(
        "--triton-gidx-direct-final-deterministic"
        if args.triton_gidx_direct_final_deterministic
        else "--no-triton-gidx-direct-final-deterministic"
    )
    cmd.append("--triton-overlap-restore-direct" if args.triton_overlap_restore_direct else "--no-triton-overlap-restore-direct")
    cmd.append(
        "--triton-cross-linear-restore-prefetch"
        if args.triton_cross_linear_restore_prefetch
        else "--no-triton-cross-linear-restore-prefetch"
    )
    cmd.append("--triton-mode1-gidx-direct-final" if args.triton_mode1_gidx_direct_final else "--no-triton-mode1-gidx-direct-final")
    cmd.extend(["--triton-mode1-input-tile-group", str(args.triton_mode1_input_tile_group)])
    cmd.append("--triton-mode1-chunked-direct-final" if args.triton_mode1_chunked_direct_final else "--no-triton-mode1-chunked-direct-final")
    cmd.append("--triton-mode2-diff-direct-final" if args.triton_mode2_diff_direct_final else "--no-triton-mode2-diff-direct-final")
    cmd.append("--triton-mode2-diff-presubtract" if args.triton_mode2_diff_presubtract else "--no-triton-mode2-diff-presubtract")
    cmd.append("--triton-mode2-diff-fuse-input-slices" if args.triton_mode2_diff_fuse_input_slices else "--no-triton-mode2-diff-fuse-input-slices")
    cmd.extend(["--triton-mode2-diff-block-r-cap", str(args.triton_mode2_diff_block_r_cap)])
    cmd.extend(["--triton-mode2-diff-block-l-cap", str(args.triton_mode2_diff_block_l_cap)])
    cmd.append("--mode1-grouped-tile-gemm" if args.mode1_grouped_tile_gemm else "--no-mode1-grouped-tile-gemm")
    cmd.append("--direct-output-chunk-write" if args.direct_output_chunk_write else "--no-direct-output-chunk-write")
    cmd.append("--fuse-mlp-gate-up" if args.fuse_mlp_gate_up else "--no-fuse-mlp-gate-up")
    cmd.append(
        "--fuse-common-input-projections"
        if args.fuse_common_input_projections
        else "--no-fuse-common-input-projections"
    )
    if args.read_variation_seed is not None:
        cmd.extend(["--read-variation-seed", str(args.read_variation_seed)])
    if args.classification_examples_json:
        cmd.extend(["--classification-examples-json", args.classification_examples_json])
    if args.execution_mode is not None:
        cmd.extend(["--execution-mode", args.execution_mode])
    cmd.extend(["--s1-stage", args.s1_stage])
    cmd.extend(["--s2-stage", args.s2_stage])
    cmd.extend(["--cuda-peak-budget-mb", str(args.cuda_peak_budget_mb)])
    cmd.extend(["--state-resident-budget-mb", str(args.state_resident_budget_mb)])
    cmd.extend(["--output-block-cols", str(args.output_block_cols)])
    if args.memory_runtime_diagnostic:
        cmd.append("--memory-runtime-diagnostic")
    cmd.append("--runtime-stage-timing" if args.runtime_stage_timing else "--no-runtime-stage-timing")
    cmd.append("--runtime-counters" if args.runtime_counters else "--no-runtime-counters")
    if args.classification_token_ids:
        cmd.extend(["--classification-token-ids", *map(str, args.classification_token_ids)])
    if args.require_all_linears:
        cmd.append("--require-all-linears")
    if args.profile:
        cmd.append("--profile")
    if not args.profile_sync_cuda:
        cmd.append("--no-profile-sync-cuda")
    if getattr(args, "module_timing", False):
        cmd.append("--module-timing")
    if args.collect_layer_buffer_accounting:
        cmd.append("--collect-layer-buffer-accounting")
    if args.language_model_only:
        cmd.append("--language-model-only")
    if args.load_on_device:
        cmd.append("--load-on-device")
    if args.simulate_lm_head:
        cmd.append("--simulate-lm-head")
    if args.only_lm_head:
        cmd.append("--only-lm-head")
    if args.always_include_lm_head:
        cmd.append("--always-include-lm-head")
    for pattern in args.linear_include_regex:
        cmd.extend(["--linear-include-regex", pattern])
    for pattern in args.linear_exclude_regex:
        cmd.extend(["--linear-exclude-regex", pattern])
    cmd.extend(["--linear-name-limit", str(args.linear_name_limit)])
    cmd.extend(["--lm-head-output-shards", str(args.lm_head_output_shards)])
    if args.lm_head_shard_devices:
        cmd.extend(["--lm-head-shard-devices", args.lm_head_shard_devices])
    if args.lm_head_shard_parallel:
        cmd.append("--lm-head-shard-parallel")
    cmd.extend(["--lm-head-input-select", args.lm_head_input_select])
    cmd.extend(["--lm-head-output-select", args.lm_head_output_select])
    if args.streaming:
        cmd.append("--streaming")
    if not args.streaming_prefetch:
        cmd.append("--no-streaming-prefetch")
    if args.streaming_prefetch_cycle:
        cmd.append("--streaming-prefetch-cycle")
    cmd.extend(["--streaming-prefetch-distance", str(args.streaming_prefetch_distance)])
    cmd.extend(["--streaming-pin-policy", args.streaming_pin_policy])
    if float(getattr(args, "streaming_persistent_pin_budget_mb", 0.0) or 0.0) > 0.0:
        cmd.extend(["--streaming-persistent-pin-budget-mb", str(args.streaming_persistent_pin_budget_mb)])
    cmd.extend(["--streaming-persistent-pin-select", args.streaming_persistent_pin_select])
    if float(getattr(args, "streaming_window_pin_cache_mb", 0.0) or 0.0) > 0.0:
        cmd.extend(["--streaming-window-pin-cache-mb", str(args.streaming_window_pin_cache_mb)])
    if args.streaming_pin_hints_json:
        cmd.extend(["--streaming-pin-hints-json", args.streaming_pin_hints_json])
    if args.memory_empty_cache_after_offload:
        cmd.append("--memory-empty-cache-after-offload")
    cmd.extend(["--memory-empty-cache-after-offload-interval", str(args.memory_empty_cache_after_offload_interval)])
    if not args.free_weights:
        cmd.append("--no-free-weights")
    if args.lazy_prepare:
        cmd.append("--lazy-prepare")
    if not args.lazy_release_after_forward:
        cmd.append("--no-lazy-release-after-forward")
    if args.max_linears is not None:
        cmd.extend(["--max-linears", str(args.max_linears)])
    if args.max_non_lm_head_linears is not None:
        cmd.extend(["--max-non-lm-head-linears", str(args.max_non_lm_head_linears)])

    child = None
    stdout = ""
    stderr = ""
    host_mon = {}
    try:
        with NvidiaSmiMonitor(_monitor_gpu(args), args.monitor_interval, args.monitor_nvidia_smi) as monitor:
            child = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with ProcessMemoryMonitor(child, args.monitor_interval) as host_monitor:
                try:
                    stdout, stderr = child.communicate(timeout=args.worker_timeout)
                except subprocess.TimeoutExpired as exc:
                    child.kill()
                    stdout, stderr = child.communicate()
                    mon = monitor.summary()
                    host_mon = host_monitor.summary()
                    raise subprocess.TimeoutExpired(exc.cmd, exc.timeout, output=stdout, stderr=stderr)
                host_mon = host_monitor.summary()
            mon = monitor.summary()
    except subprocess.TimeoutExpired as exc:
        mon = monitor.summary() if "monitor" in locals() else {}
        if args.fail_fast:
            raise
        return failure_row(
            args,
            label,
            kind,
            "timeout",
            f"Timed out after {args.worker_timeout}s\nSTDOUT:\n{_trim_text(exc.stdout)}\nSTDERR:\n{_trim_text(exc.stderr)}",
            extra,
            mon,
            host_mon,
            returncode="timeout",
            stdout=exc.stdout,
            stderr=exc.stderr,
            cmd=cmd,
        )
    finally:
        try:
            os.remove(worker_path)
        except OSError:
            pass

    class ProcResult:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    proc = ProcResult(child.returncode if child is not None else -999, stdout, stderr)

    if proc.returncode != 0:
        if args.fail_fast:
            raise RuntimeError(f"{label} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return failure_row(
            args,
            label,
            kind,
            "failed",
            f"Exit code {proc.returncode}\nSTDOUT:\n{_trim_text(proc.stdout)}\nSTDERR:\n{_trim_text(proc.stderr)}",
            extra,
            mon,
            host_mon,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            cmd=cmd,
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return failure_row(
            args,
            label,
            kind,
            "failed",
            f"No output\nSTDERR:\n{proc.stderr}",
            extra,
            mon,
            host_mon,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            cmd=cmd,
        )
    try:
        row = json.loads(lines[-1])
    except json.JSONDecodeError:
        return failure_row(
            args,
            label,
            kind,
            "failed",
            f"Could not parse JSON\nSTDOUT:\n{_trim_text(proc.stdout)}\nSTDERR:\n{_trim_text(proc.stderr)}",
            extra,
            mon,
            host_mon,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            cmd=cmd,
        )
    row["monitor"] = mon
    row["host_monitor"] = host_mon
    row["logits_path"] = str(logits_path) if logits_path.exists() else ""
    return row


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.{digits}f}"


def compute_output_agreement(rows: list[dict], enabled: bool = True) -> None:
    if not enabled:
        return
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("logits_path")]
    if not ok_rows:
        return
    ref = next((row for row in ok_rows if row.get("kind") == "hf"), ok_rows[0])
    ref_path = ref.get("logits_path")
    if not ref_path or not Path(ref_path).exists():
        return
    import torch

    def load_logits(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    ref_logits = load_logits(ref_path).float()
    ref_top1 = torch.argmax(ref_logits, dim=-1)
    ref_energy = torch.mean(ref_logits * ref_logits)
    ref_generation_tokens = ref.get("generation_token_ids") or []
    for row in ok_rows:
        row["agreement_ref_label"] = ref.get("label", "")
        path = row.get("logits_path")
        if not path or not Path(path).exists():
            continue
        try:
            logits = load_logits(path).float()
        except Exception as exc:
            row["agreement_error"] = type(exc).__name__
            continue
        if tuple(logits.shape) != tuple(ref_logits.shape):
            row["agreement_error"] = f"shape mismatch: {tuple(logits.shape)} vs {tuple(ref_logits.shape)}"
            continue
        diff = logits - ref_logits
        noise = torch.mean(diff * diff)
        row["agreement_snr_db"] = float("inf") if float(noise) == 0.0 else float(10.0 * torch.log10(ref_energy / torch.clamp(noise, min=1e-30)))
        row["agreement_mae"] = float(torch.mean(torch.abs(diff)))
        row["agreement_max_abs_err"] = float(torch.max(torch.abs(diff)))
        row["agreement_top1"] = float(torch.mean((torch.argmax(logits, dim=-1) == ref_top1).float()))
        generation_tokens = row.get("generation_token_ids") or []
        if ref_generation_tokens and generation_tokens:
            total = 0
            same = 0
            for pred_seq, ref_seq in zip(generation_tokens, ref_generation_tokens):
                for pred_tok, ref_tok in zip(pred_seq, ref_seq):
                    total += 1
                    same += int(pred_tok == ref_tok)
            row["generation_token_agreement"] = same / total if total else None
            row["generation_sequence_match"] = generation_tokens == ref_generation_tokens


def print_table(rows: list[dict]) -> None:
    headers = [
        "status",
        "label",
        "kind",
        "seq",
        "work",
        "exec",
        "mode",
        "input",
        "fast",
        "backend",
        "prof",
        "lazy",
        "full",
        "cov",
        "lmhead",
        "headpar",
        "onlyhead",
        "fwd ms",
        "pre ms",
        "dec ms/tok",
        "tok/s",
        "prof ms",
        "top event",
        "SNR",
        "top1",
        "cls task",
        "cls acc",
        "cls margin",
        "cls H",
        "fwd MB",
        "smi MB",
        "gpu%",
        "mem%",
        "power",
        "resident MB",
        "G MB",
        "G gpu",
        "G pin",
        "rss max",
        "avail min",
        "layers",
    ]
    table = []
    for row in rows:
        mon = row.get("monitor", {})
        host = row.get("host_monitor", {})
        table.append([
            row.get("status", "ok"),
            row["label"],
            row["kind"],
            str(row.get("seq", "N/A")),
            row.get("workload", "prefill"),
            row.get("execution_mode", "manual"),
            str(row.get("mode", 0)),
            row.get("mode2_input_mode", "n/a"),
            str(row.get("fast_inference", False)),
            row.get("fast_inference_backend", "n/a"),
            str(row.get("profile", False)),
            str(row.get("lazy_prepare", False)),
            str(row.get("full_model_deployment_valid", "N/A")),
            fmt(row.get("replacement_coverage"), 3) if row.get("replacement_coverage") is not None else "N/A",
            str(row.get("simulate_lm_head", False)),
            str(row.get("lm_head_shard_parallel", False)),
            str(row.get("only_lm_head", False)),
            fmt(row.get("forward_ms_mean")),
            fmt(row.get("prefill_ms_mean")),
            fmt(row.get("decode_ms_per_token")),
            fmt(row.get("tokens_per_s"), 1),
            fmt(row.get("profile_total_ms")),
            (
                row.get("profile_top_events", [{}])[0].get("label", "N/A")
                if row.get("profile_top_events") else "N/A"
            ),
            fmt(row.get("agreement_snr_db")),
            fmt(row.get("agreement_top1"), 3),
            row.get("classification_task", ""),
            fmt(row.get("classification_accuracy_proxy"), 3),
            fmt(row.get("classification_margin_mean")),
            fmt(row.get("classification_entropy_mean")),
            fmt(row.get("forward_peak_mb"), 1),
            fmt(mon.get("mem_used_mb_max"), 1),
            fmt(mon.get("gpu_util_pct_mean"), 1),
            fmt(mon.get("mem_util_pct_mean"), 1),
            fmt(mon.get("power_w_mean"), 1),
            fmt(row.get("resident_after_prepare_mb"), 1),
            fmt(row.get("g_storage_mb"), 1),
            fmt(row.get("g_gpu_mb"), 1),
            fmt(row.get("g_cpu_pinned_mb"), 1),
            fmt(host.get("VmRSS_mb_max"), 1),
            fmt(host.get("MemAvailable_mb_min"), 1),
            str(row.get("prepared_layers", "N/A")),
        ])
    widths = [len(h) for h in headers]
    for row in table:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    def line(values):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    print(line(headers))
    print(line("-" * w for w in widths))
    for row in table:
        print(line(row))


def _argv_has_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in argv)


def parse_args():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-loader", choices=["auto", "causal", "multimodal"], default="auto")
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--original-root", default=None)
    parser.add_argument("--v2-root", default=None)
    parser.add_argument("--v3-root", default=str(ROOT))
    parser.add_argument("--ld-library-path", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-on-device", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile-exclude-linearmem", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cuda-profiler-capture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--workload", choices=["prefill", "classification", "generation"], default="prefill")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--classification-task", choices=["token_proxy", "arc_easy_smoke", "multi_choice_json"], default="token_proxy")
    parser.add_argument("--classification-examples-json", default="")
    parser.add_argument("--classification-choice-labels", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--classification-token-ids", type=int, nargs="*", default=None)
    parser.add_argument("--classification-candidate-count", type=int, default=4)
    parser.add_argument("--classification-scoring", choices=["label_token", "choice_text_ll"], default="label_token")
    parser.add_argument("--classification-score-batch-size", type=int, default=8)
    parser.add_argument("--compute-agreement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logits-dir", default=None)
    parser.add_argument("--execution-mode", choices=["speed", "balanced", "memory"], default=None)
    parser.add_argument("--s1-stage", choices=["off", "block", "budgeted"], default="budgeted")
    parser.add_argument("--s2-stage", choices=["off", "intra", "full"], default="full")
    parser.add_argument("--s2-ablation-stage", choices=["off", "grouped", "intra", "full"], default=None)
    parser.add_argument("--cuda-peak-budget-mb", type=float, default=0.0)
    parser.add_argument("--state-resident-budget-mb", type=float, default=0.0)
    parser.add_argument("--output-block-cols", type=int, default=0)
    parser.add_argument("--memory-prepare-policy", choices=["lazy_release", "lazy_keep", "eager_streaming"], default="lazy_release")
    parser.add_argument("--memory-runtime-diagnostic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-budget-mb", type=float, default=0.0)
    parser.add_argument("--memory-resident-budget-mb", type=float, default=0.0)
    parser.add_argument("--memory-resident-select", choices=["sequential", "largest", "runtime"], default="largest")
    parser.add_argument("--require-all-linears", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-hf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-v2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-v3-mode0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-v3-mode1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-v3-mode2-signed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-v3-mode2-diff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--simulate-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--always-include-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--linear-include-regex", action="append", default=[])
    parser.add_argument("--linear-exclude-regex", action="append", default=[])
    parser.add_argument("--linear-name-limit", type=int, default=32)
    parser.add_argument("--lm-head-output-shards", type=int, default=1)
    parser.add_argument("--lm-head-shard-devices", default="")
    parser.add_argument("--lm-head-shard-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lm-head-input-select", choices=["all", "last"], default="all")
    parser.add_argument("--lm-head-output-select", choices=["all", "label_tokens"], default="all")
    parser.add_argument("--max-linears", type=int, default=None)
    parser.add_argument("--max-non-lm-head-linears", type=int, default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-prefetch-cycle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streaming-prefetch-distance", type=int, default=1)
    parser.add_argument("--streaming-pin-policy", choices=["persistent", "window"], default="persistent")
    parser.add_argument("--streaming-persistent-pin-budget-mb", type=float, default=0.0)
    parser.add_argument("--streaming-persistent-pin-select", choices=["sequential", "largest", "runtime"], default="sequential")
    parser.add_argument("--streaming-window-pin-cache-mb", type=float, default=0.0)
    parser.add_argument("--streaming-pin-hints-json", default="")
    parser.add_argument("--memory-empty-cache-after-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-empty-cache-after-offload-interval", type=int, default=1)
    parser.add_argument("--free-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lazy-prepare", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lazy-release-after-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-timeout", type=float, default=900)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--monitor-nvidia-smi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-gpu", default=None)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument("--hgs", type=float, default=1e-5)
    parser.add_argument("--lgs", type=float, default=1e-8)
    parser.add_argument("--g-level", type=int, default=16)
    parser.add_argument("--write-variation", type=float, default=0.0)
    parser.add_argument("--read-variation", type=float, default=0.0)
    parser.add_argument("--read-variation-seed", type=int, default=None)
    parser.add_argument("--vnoise", type=float, default=0.0)
    parser.add_argument("--write-variation-mode", choices=["materialized", "virtual"], default="materialized")
    parser.add_argument("--conductance-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--compute-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--linear-output-dtype", choices=["auto", "input", "float32", "float16", "bfloat16"], default="input")
    parser.add_argument("--vmm-lowp-format", choices=VMM_LOWP_CHOICES, default="auto")
    parser.add_argument("--mode0-semantic-policy", choices=["auto", "strict", "fast"], default="auto")
    parser.add_argument("--mode0-vmm-compute-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--rdac", type=int, default=256)
    parser.add_argument("--radc", type=int, default=4096)
    parser.add_argument("--vread", type=float, default=0.2)
    parser.add_argument("--rate-stuck-hgs", type=float, default=0.0)
    parser.add_argument("--rate-stuck-lgs", type=float, default=0.0)
    parser.add_argument("--input-slice", type=int, nargs="+", default=[1, 1, 1, 1, 1])
    parser.add_argument("--weight-slice", type=int, nargs="+", default=[1, 1, 1, 1, 1])
    parser.add_argument("--mode2-input-slice", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--mode2-weight-slice", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--fast-inference-backend", choices=["auto", "torch", "triton", "triton_gidx"], default="auto")
    parser.add_argument("--triton-input-precision", choices=["ieee", "tf32", "tf32x3"], default="ieee")
    parser.add_argument("--triton-block-r", type=int, default=32)
    parser.add_argument("--triton-block-l", type=int, default=16)
    parser.add_argument("--triton-block-k", type=int, default=64)
    parser.add_argument("--triton-output-chunk-limit", type=int, default=256)
    parser.add_argument("--triton-auto-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode0-input-tile-group", type=int, default=1)
    parser.add_argument("--mode-aware-triton-defaults", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode0-triton-block-r", type=int, default=64)
    parser.add_argument("--mode0-triton-block-l", type=int, default=32)
    parser.add_argument("--mode0-triton-output-chunk-limit", type=int, default=512)
    parser.add_argument("--mode1-triton-block-r", type=int, default=64)
    parser.add_argument("--mode1-triton-output-chunk-limit", type=int, default=256)
    parser.add_argument("--mode2-triton-block-l", type=int, default=8)
    parser.add_argument("--mode2-triton-block-k", type=int, default=64)
    parser.add_argument("--triton-gidx-read-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-fused-restore-read-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-block", type=int, default=512)
    parser.add_argument("--triton-gidx-restore-block-auto", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-small-block", type=int, default=128)
    parser.add_argument("--triton-gidx-restore-auto-in-features-threshold", type=int, default=4096)
    parser.add_argument("--triton-gidx-restore-num-warps", type=int, default=4)
    parser.add_argument("--triton-gidx-restore-strided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-m-slab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-approx-linear-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-restore-exp2-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-restore-fast-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-fuse-input-slices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode0-strict-intermediate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--triton-mode0-strict-intermediate-backend",
        choices=["auto", "gidx", "off"],
        default="auto",
    )
    parser.add_argument("--triton-reuse-input-voltage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-reuse-weight-tile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-precompute-input-voltage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fast-adc-scale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-restored-input-slices", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-activation-slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-reuse-activation-slice-buffer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-probe-activation-slice-reuse", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-probe-activation-density", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-activation-slice-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-activation-slice-cache-max-entries", type=int, default=8)
    parser.add_argument("--triton-binary-input-slice-dac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-direct-final-num-warps", type=int, default=4)
    parser.add_argument("--triton-direct-final-partial-m-group", type=int, default=0)
    parser.add_argument("--triton-direct-final-exact-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-fuse-output-finalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-direct-final-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-direct-output-zero-once", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-gidx-direct-final-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-gidx-direct-final-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-overlap-restore-direct", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-cross-linear-restore-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode1-gidx-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode1-input-tile-group", type=int, default=1)
    parser.add_argument("--triton-mode1-chunked-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-direct-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-presubtract", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--triton-mode2-diff-fuse-input-slices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triton-mode2-diff-block-r-cap", type=int, default=16)
    parser.add_argument("--triton-mode2-diff-block-l-cap", type=int, default=8)
    parser.add_argument("--mode1-grouped-tile-gemm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-output-chunk-write", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-mlp-gate-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fuse-common-input-projections", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-sync-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-stage-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--runtime-counters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--module-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--input-paral-size", type=int, nargs=2, default=[1, 64])
    parser.add_argument("--weight-paral-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--input-quant-gran", type=int, nargs=2, default=[1, 64])
    parser.add_argument("--weight-quant-gran", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--inference-chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.0)
    parser.add_argument("--collect-layer-buffer-accounting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-buffer-accounting-limit", type=int, default=12)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    args._triton_block_r_user_set = _argv_has_option(raw_argv, "--triton-block-r")
    args._triton_block_l_user_set = _argv_has_option(raw_argv, "--triton-block-l")
    args._triton_block_k_user_set = _argv_has_option(raw_argv, "--triton-block-k")
    args._triton_output_chunk_limit_user_set = _argv_has_option(raw_argv, "--triton-output-chunk-limit")
    args._inference_chunk_size_user_set = _argv_has_option(raw_argv, "--inference-chunk-size")
    args._conductance_dtype_user_set = _argv_has_option(raw_argv, "--conductance-dtype")
    args._compute_dtype_user_set = _argv_has_option(raw_argv, "--compute-dtype")
    args = apply_vmm_lowp_format_defaults(args, argv_has_option=lambda opt: _argv_has_option(raw_argv, opt))
    args._triton_fuse_restored_input_slices_user_set = (
        _argv_has_option(raw_argv, "--triton-fuse-restored-input-slices")
        or _argv_has_option(raw_argv, "--no-triton-fuse-restored-input-slices")
    )
    args._triton_overlap_restore_direct_user_set = (
        _argv_has_option(raw_argv, "--triton-overlap-restore-direct")
        or _argv_has_option(raw_argv, "--no-triton-overlap-restore-direct")
    )
    args._triton_precompute_input_voltage_user_set = (
        _argv_has_option(raw_argv, "--triton-precompute-input-voltage")
        or _argv_has_option(raw_argv, "--no-triton-precompute-input-voltage")
    )
    args._triton_reuse_input_voltage_user_set = (
        _argv_has_option(raw_argv, "--triton-reuse-input-voltage")
        or _argv_has_option(raw_argv, "--no-triton-reuse-input-voltage")
    )
    args._triton_fast_adc_scale_user_set = (
        _argv_has_option(raw_argv, "--triton-fast-adc-scale")
        or _argv_has_option(raw_argv, "--no-triton-fast-adc-scale")
    )
    args._triton_mode0_strict_intermediate_user_set = (
        _argv_has_option(raw_argv, "--triton-mode0-strict-intermediate")
        or _argv_has_option(raw_argv, "--no-triton-mode0-strict-intermediate")
    )
    args._triton_direct_output_zero_once_user_set = (
        _argv_has_option(raw_argv, "--triton-direct-output-zero-once")
        or _argv_has_option(raw_argv, "--no-triton-direct-output-zero-once")
    )
    args._triton_direct_final_exact_reduce_user_set = (
        _argv_has_option(raw_argv, "--triton-direct-final-exact-reduce")
        or _argv_has_option(raw_argv, "--no-triton-direct-final-exact-reduce")
    )
    args._triton_gidx_direct_final_output_user_set = (
        _argv_has_option(raw_argv, "--triton-gidx-direct-final-output")
        or _argv_has_option(raw_argv, "--no-triton-gidx-direct-final-output")
    )
    args._fuse_mlp_gate_up_user_set = (
        _argv_has_option(raw_argv, "--fuse-mlp-gate-up")
        or _argv_has_option(raw_argv, "--no-fuse-mlp-gate-up")
    )
    args._fuse_common_input_projections_user_set = (
        _argv_has_option(raw_argv, "--fuse-common-input-projections")
        or _argv_has_option(raw_argv, "--no-fuse-common-input-projections")
    )
    if args.s2_ablation_stage is not None and not _argv_has_option(raw_argv, "--s2-stage"):
        args.s2_stage = args.s2_ablation_stage
    args = apply_launcher_execution_mode_defaults(args)
    args = apply_s1_stage(args)
    args, args._s2_fast_inference = apply_s2_stage(args)
    return args


def main():
    args = parse_args()
    rows = []
    if args.include_hf:
        rows.append(run_worker(args, None, "hf_float", "hf", []))
    if args.include_original:
        if args.original_root is None:
            raise ValueError("--original-root is required with --include-original")
        rows.append(run_worker(args, Path(args.original_root).resolve(), "original_mem", "original", []))
    if args.include_v2:
        if args.v2_root is None:
            raise ValueError("--v2-root is required with --include-v2")
        rows.append(run_worker(args, Path(args.v2_root).resolve(), "v2_mem_mode0", "v2", []))
    v3_root = Path(args.v3_root).resolve()
    if args.include_v3_mode0:
        extra = ["--mode", "0"]
        if args._s2_fast_inference:
            extra.append("--fast-inference")
        rows.append(run_worker(
            args,
            v3_root,
            f"v3_mem_mode0_s1_{args.s1_stage}_s2_{args.s2_stage}",
            "v3",
            extra,
        ))
    if args.include_v3_mode1:
        rows.append(run_worker(args, v3_root, "v3_mem_mode1_fast", "v3", ["--mode", "1", "--fast-inference"]))
    if args.include_v3_mode2_signed:
        rows.append(run_worker(
            args,
            v3_root,
            "v3_mem_mode2_signed_fast",
            "v3",
            ["--mode", "2", "--mode2-input-mode", "signed", "--fast-inference"],
        ))
    if args.include_v3_mode2_diff:
        rows.append(run_worker(
            args,
            v3_root,
            "v3_mem_mode2_diff_fast",
            "v3",
            ["--mode", "2", "--mode2-input-mode", "differential", "--fast-inference"],
        ))

    compute_output_agreement(rows, enabled=args.compute_agreement)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    print_table(rows)


if __name__ == "__main__":
    main()
