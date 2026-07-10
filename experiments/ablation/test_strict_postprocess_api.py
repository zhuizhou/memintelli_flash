import pytest
import torch

from memintelli.pimpy import triton_fast_accumulate


def test_strict_adc_scale_accumulate_api_is_exposed():
    assert callable(triton_fast_accumulate.triton_strict_adc_scale_accumulate)


def _strict_torch_postprocess(partial, scale, accumulated, *, adc_ref, radc):
    partial = partial.to(torch.float32)
    partial.div_(adc_ref)
    partial.mul_(radc - 1)
    partial.round_()
    partial.div_(radc - 1)
    partial.mul_(scale)
    if accumulated is None:
        return partial
    accumulated.add_(partial)
    return accumulated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_accumulator", [False, True])
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_strict_adc_scale_accumulate_is_bitwise_equal_to_torch(
    with_accumulator,
    non_contiguous,
):
    torch.manual_seed(17)
    device = torch.device("cuda")
    source = torch.randn(7, 5, 3, 11, device=device, dtype=torch.bfloat16)
    partial = source.select(2, 1) if non_contiguous else source[:, :, 1].contiguous()
    scale = torch.tensor(0.03125, device=device, dtype=torch.float32)
    accumulated = (
        torch.randn(partial.shape, device=device, dtype=torch.float32)
        if with_accumulator
        else None
    )
    expected_accumulator = accumulated.clone() if accumulated is not None else None

    expected = _strict_torch_postprocess(
        partial.clone(),
        scale,
        expected_accumulator,
        adc_ref=2.75,
        radc=256,
    )
    actual = triton_fast_accumulate.triton_strict_adc_scale_accumulate(
        partial,
        scale,
        accumulated,
        adc_ref=2.75,
        radc=256,
    )
    torch.cuda.synchronize(device)

    assert torch.equal(actual, expected)
