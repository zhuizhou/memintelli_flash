# -*- coding:utf-8 -*-
# @File  : 11_qwen3_wikitext2_inference.py
# @Author: Zhou
# @Date  : 2026/2/13
"""Memintelli example 11: Qwen3 token-level PPL on WikiText-2."""

import math
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import sys
from pathlib import Path

from transformers.models import FalconConfig
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from memintelli.NN_models import qwen3_zoo
from memintelli.pimpy.memmat_tensor import DPETensor


def load_wikitext2_test_text():
    """Load and join WikiText-2 test split."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",cache_dir="/data/dataset/wikitext2")
    # dataset = dataset.select(range(5))
    texts = [line for line in dataset["text"] if line and line.strip()]
    return "\n\n".join(texts)


@torch.no_grad()
def evaluate_token_ppl(model, tokenizer, text, device, max_length=1024, stride=512):
    """Evaluate token-level perplexity with sliding window."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, seq_len, stride), desc="Evaluating token PPL"):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i

        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        # use_cache=False: prevents KV cache from accumulating in returned outputs.
        # For PPL evaluation we don't reuse KV across windows, so caching wastes memory.
        outputs = model(input_ids=input_ids_chunk, labels=target_ids, use_cache=False)
        nll_val = (outputs.loss * trg_len).item()
        del outputs  # immediately free logits + any intermediates
        total_nll += nll_val
        total_tokens += trg_len

        # Periodically release unused CUDA cache to prevent fragmentation growth.
        # The caching allocator holds freed blocks; empty_cache() returns them to the driver.
        if (i // stride) % 10 == 9:
            torch.cuda.empty_cache()

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


def main():
    model_name = "meta-llama/Llama-3.2-3B"
    # model_name = "Qwen/Qwen3-8B"
    mem_enabled = True
    max_length = 512
    stride = 512

    input_slice = (1, 1, 1, 1)
    weight_slice = (1, 1, 1, 1)
    bw_e = None

    # Enable TF32 for faster float32 matmul on Ampere+ GPUs (RTX 3xxx/4xxx, A100, etc.)
    # This gives ~2-3x speedup for non-RRAM ops (attention, norms) with negligible precision loss.
    torch.set_float32_matmul_precision('high')

    # Streaming mode controls GPU↔CPU offloading of G data:
    #   False:  ALL G on GPU (fastest, ~16GB for 4B)
    #   True:   ALL G on CPU, load per-layer (slowest, ~3GB for 4B)  
    #   "auto": Keep as many layers on GPU as memory allows, stream the rest (recommended)
    streaming = "auto"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mem_engine = DPETensor(
        HGS=1e-5,
        LGS=1e-8,
        write_variation=0.0,
        rate_stuck_HGS=0.00,
        rate_stuck_LGS=0.00,
        read_variation=0.05,
        vnoise=0.0,
        rdac=2**1,
        g_level=2**1,
        radc=2**6,
        device=device,  
        inference_chunk_size=32*1024*1024,
        # IMPORTANT: must match model device (e.g. cuda:0, cuda:1)
        # inference_chunk_size controls peak memory during inference matmul.
        # Default (None) = ~32M elements (~128MB). Lower for smaller GPUs:
        #   8GB GPU:  inference_chunk_size=4*1024*1024
        #   16GB GPU: inference_chunk_size=16*1024*1024
        #   24GB GPU: inference_chunk_size=64*1024*1024  (or None for auto)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NOTE: Do NOT chain .to(device) here!
    # LinearMem weights are created on CPU during replacement to avoid GPU OOM.
    # update_weight_and_prepare() handles moving each layer to GPU one at a time.
    model = qwen3_zoo(
        model_name=model_name,
        pretrained=True,
        mem_enabled=mem_enabled,
        engine=mem_engine,
        input_slice=input_slice,
        weight_slice=weight_slice,
        device=device,
        bw_e=bw_e,
        input_paral_size=(1, 64),
        weight_paral_size=(64, 1),
        input_quant_gran=(1, 64),
        weight_quant_gran=(64, 1),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        # skip_embedding_and_head=True (default): lm_head uses standard nn.Linear.
        # Saves ~50% inference time + ~1.5GB memory for 4B model.
        skip_embedding_and_head=False,
    )

    if mem_enabled:
        # Process weights → G one layer at a time (minimal GPU peak memory).
        # streaming=True: only ONE layer's G on GPU at any time.
        model.update_weight_and_prepare(streaming="auto_speed")

    # Move remaining params (embedding, norms, lm_head) to GPU.
    # After update_weight_and_prepare: LinearMem weights are freed (empty),
    # so .to(device) only moves ~1-2GB of non-LinearMem params.
    model = model.to(device)

    # NOTE on torch.compile:
    # torch.compile is NOT compatible with the RRAM simulation engine because:
    #   1. Python loops over slices cause graph breaks
    #   2. float() / .item() calls on tensors break the computation graph
    #   3. Dynamic input shapes (varying seq_len) cause excessive recompilation
    #   4. mode="reduce-overhead" (CUDAGraph) crashes with dynamic shapes
    # Instead, we use:
    #   - torch.set_float32_matmul_precision('high') for TF32 matmul speedup
    #   - Manual chunked matmul optimization in the engine
    #   - G compression (uint8) + streaming for memory efficiency

    text = load_wikitext2_test_text()
    ppl = evaluate_token_ppl(
        model=model,
        tokenizer=tokenizer,
        text=text,
        device=device,
        max_length=max_length,
        stride=stride
    )
    print(f"\nToken-level PPL of {model_name} on WikiText-2(test): {ppl:.4f}")


if __name__ == "__main__":
    main()
