from __future__ import annotations

import torch
import pytest

from memintelli.pimpy.triton_fast_accumulate import (
    _mode1_kernel_operand,
    triton_mode1_gidx_direct_final,
)


def test_mode1_kernel_operand_preserves_last_dim_contiguous_view():
    base = torch.arange(8 * 32, dtype=torch.int16).reshape(8, 32)
    view = base[:, 3:19]

    assert not view.is_contiguous()
    assert view.stride(1) == 1

    operand = _mode1_kernel_operand(view)

    assert operand.data_ptr() == view.data_ptr()
    assert operand.stride() == view.stride()


def test_mode1_kernel_operand_copies_transposed_input():
    transposed = torch.arange(8 * 16, dtype=torch.int16).reshape(8, 16).t()
    assert transposed.stride(1) != 1

    operand = _mode1_kernel_operand(transposed)

    assert operand.is_contiguous()
    torch.testing.assert_close(operand, transposed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_strided_kernel_matches_contiguous_operands():
    torch.manual_seed(11)
    device = torch.device("cuda")
    x = torch.randn((32, 128), device=device, dtype=torch.float32)
    gp = torch.randint(0, 16, (128, 256), device=device, dtype=torch.uint8)[:, 32:224]
    gn = torch.randint(0, 16, (128, 256), device=device, dtype=torch.uint8)[:, 32:224]
    w_scale = (torch.rand((2, 5), device=device, dtype=torch.float32) + 0.1)[:, 1:4]
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

    strided = triton_mode1_gidx_direct_final(x, gp, gn, w_scale, **kwargs)
    contiguous = triton_mode1_gidx_direct_final(
        x,
        gp.contiguous(),
        gn.contiguous(),
        w_scale.contiguous(),
        **kwargs,
    )

    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)
