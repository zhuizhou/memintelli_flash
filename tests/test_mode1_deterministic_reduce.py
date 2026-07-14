from __future__ import annotations

import inspect

import pytest
import torch

from memintelli.pimpy.triton_fast_accumulate import (
    triton_mode1_gdiff_direct_final,
    triton_mode1_gidx_direct_final,
)


def test_mode1_gdiff_api_exposes_deterministic_reduce():
    signature = inspect.signature(triton_mode1_gdiff_direct_final)
    assert "deterministic_reduce" in signature.parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_gdiff", [False, True])
def test_mode1_deterministic_reduce_repeats_exactly(use_gdiff: bool):
    torch.manual_seed(29)
    device = torch.device("cuda")
    x = torch.randn((32, 512), device=device, dtype=torch.float32)
    gdiff = torch.randint(-15, 16, (512, 384), device=device, dtype=torch.int8)
    w_scale = torch.rand((8, 6), device=device, dtype=torch.float32) + 0.1
    kwargs = {
        "x_max": x.abs().max(),
        "lgs": 1e-7,
        "q_g": (1e-5 - 1e-7) / 15,
        "read_sigma": 0.0,
        "adc_ref_unit": (1e-5 - 1e-7) * 0.2,
        "rdac": 16,
        "radc": 256,
        "vread": 0.2,
        "g_level": 16,
        "tile_in": 64,
        "tile_out": 64,
        "input_precision": "ieee",
        "dot_dtype_override": 2,
        "block_r": 32,
        "block_l": 16,
        "block_k": 64,
        "input_tile_group": 4,
        "deterministic_reduce": True,
    }

    if use_gdiff:
        operands = (gdiff, w_scale)
        kernel = triton_mode1_gdiff_direct_final
    else:
        gp = torch.clamp(gdiff.to(torch.int16), min=0).to(torch.uint8)
        gn = torch.clamp(-gdiff.to(torch.int16), min=0).to(torch.uint8)
        operands = (gp, gn, w_scale)
        kernel = triton_mode1_gidx_direct_final

    first = kernel(x, *operands, **kwargs)
    second = kernel(x, *operands, **kwargs)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
