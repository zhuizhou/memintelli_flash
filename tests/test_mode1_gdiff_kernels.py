import inspect

import pytest
import torch

from memintelli.pimpy import triton_fast_accumulate


def test_mode1_noisy_gdiff_restore_api_is_exported():
    assert hasattr(
        triton_fast_accumulate,
        "triton_restore_mode1_gdiff_gidx_read_noise",
    )


def test_mode1_gdiff_direct_final_api_is_exported():
    assert hasattr(
        triton_fast_accumulate,
        "triton_mode1_gdiff_direct_final",
    )


def test_mode1_gdiff_restore_accepts_reusable_output_workspace():
    signature = inspect.signature(
        triton_fast_accumulate.triton_restore_mode1_gdiff_gidx_read_noise
    )

    assert "out" in signature.parameters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_gdiff_restore_zero_noise_matches_index_difference():
    restore = triton_fast_accumulate.triton_restore_mode1_gdiff_gidx_read_noise
    gp = torch.randint(0, 16, (130, 257), device="cuda", dtype=torch.uint8)
    gn = torch.randint(0, 16, (130, 257), device="cuda", dtype=torch.uint8)

    actual = restore(
        gp,
        gn,
        lgs=1e-6,
        q_g=2e-6,
        read_sigma=0.0,
        dtype=torch.float32,
        noise_seed=123,
        noise_offset_base=0,
    )
    expected = (gp.float() - gn.float()) * 2e-6

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_gdiff_restore_uses_independent_reproducible_branch_noise():
    restore = triton_fast_accumulate.triton_restore_mode1_gdiff_gidx_read_noise
    gp = torch.full((256, 256), 8, device="cuda", dtype=torch.uint8)
    gn = gp.clone()

    first = restore(
        gp,
        gn,
        lgs=1e-6,
        q_g=2e-6,
        read_sigma=0.05,
        dtype=torch.float32,
        noise_seed=123,
        noise_offset_base=17,
    )
    second = restore(
        gp,
        gn,
        lgs=1e-6,
        q_g=2e-6,
        read_sigma=0.05,
        dtype=torch.float32,
        noise_seed=123,
        noise_offset_base=17,
    )

    assert first.std().item() > 0.0
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_gdiff_direct_final_matches_fp32_tile_reference():
    direct = triton_fast_accumulate.triton_mode1_gdiff_direct_final
    torch.manual_seed(19)
    vin = torch.randn((32, 130), device="cuda", dtype=torch.float32) * 0.1
    gdiff = torch.randn((130, 193), device="cuda", dtype=torch.float32) * 1e-6
    tile_in, tile_out = 64, 64
    in_tiles = (gdiff.shape[0] + tile_in - 1) // tile_in
    out_tiles = (gdiff.shape[1] + tile_out - 1) // tile_out
    scale = torch.rand((in_tiles, out_tiles), device="cuda", dtype=torch.float32) + 0.5
    x_max = torch.tensor(2.0, device="cuda", dtype=torch.float32)
    adc_ref_unit = 1.98e-6
    radc = 256
    q_g = 2e-6
    vread = 0.2
    g_level = 16

    actual = direct(
        vin,
        gdiff,
        scale,
        x_max=x_max,
        adc_ref_unit=adc_ref_unit,
        radc=radc,
        vread=vread,
        q_g=q_g,
        g_level=g_level,
        tile_in=tile_in,
        tile_out=tile_out,
        input_precision="ieee",
        dot_dtype_override=0,
        block_r=32,
        block_l=16,
        block_k=64,
        input_tile_group=1,
    )

    expected = torch.zeros_like(actual)
    for r0 in range(0, gdiff.shape[0], tile_in):
        r1 = min(r0 + tile_in, gdiff.shape[0])
        current = vin[:, r0:r1] @ gdiff[r0:r1]
        adc_ref = adc_ref_unit * (r1 - r0)
        quantized = torch.round(current / adc_ref * (radc - 1)) / (radc - 1)
        row_tile = r0 // tile_in
        column_scale = torch.repeat_interleave(
            scale[row_tile],
            tile_out,
        )[: gdiff.shape[1]]
        expected += (
            quantized
            * adc_ref
            * column_scale.view(1, -1)
            * (x_max / (vread * q_g * (g_level - 1)))
        )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
