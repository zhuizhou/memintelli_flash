from __future__ import annotations

import pytest
import torch

from memintelli.pimpy.triton_fast_accumulate import (
    triton_mode1_gdiff_direct_final,
    triton_mode1_gidx_direct_final,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_gdiff_kernel_matches_pair_direct_final():
    torch.manual_seed(17)
    device = torch.device("cuda")
    x = torch.randn((32, 128), device=device, dtype=torch.float32)
    gdiff = torch.randint(-15, 16, (128, 192), device=device, dtype=torch.int8)
    gp = torch.clamp(gdiff.to(torch.int16), min=0).to(torch.uint8)
    gn = torch.clamp(-gdiff.to(torch.int16), min=0).to(torch.uint8)
    w_scale = torch.rand((2, 3), device=device, dtype=torch.float32) + 0.1
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
        "input_tile_group": 2,
    }

    expected = triton_mode1_gidx_direct_final(x, gp, gn, w_scale, **kwargs)
    actual = triton_mode1_gdiff_direct_final(x, gdiff, w_scale, **kwargs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
