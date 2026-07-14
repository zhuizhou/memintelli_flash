import pytest
import torch

from memintelli.pimpy import triton_fast_accumulate


def test_mode1_noisy_gdiff_restore_api_is_exported():
    assert hasattr(
        triton_fast_accumulate,
        "triton_restore_mode1_gdiff_gidx_read_noise",
    )


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
