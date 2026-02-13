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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from memintelli.NN_models import qwen3_zoo
from memintelli.pimpy.memmat_tensor import DPETensor


def load_wikitext2_test_text():
    """Load and join WikiText-2 test split."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",cache_dir="/dataset/wikitext2")
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

        outputs = model(input_ids=input_ids_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        total_nll += neg_log_likelihood.item()
        total_tokens += trg_len

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


def main():
    model_name = "Qwen/Qwen3-0.6B"
    mem_enabled = True
    max_length = 1024
    stride = 512

    input_slice = (1, 1, )
    weight_slice = (1, 1,  )
    bw_e = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mem_engine = DPETensor(
        HGS=1e-5,
        LGS=1e-8,
        write_variation=0.0,
        rate_stuck_HGS=0.00,
        rate_stuck_LGS=0.00,
        read_variation=0.05,
        vnoise=0.0,
        rdac=2**2,
        g_level=2**2,
        radc=2**12
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)

    if mem_enabled:
        model.update_weight()
        model.prepare_for_inference()

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
