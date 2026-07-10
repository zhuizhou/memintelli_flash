import itertools

import torch

from memintelli.pimpy.triton_fast_accumulate import (
    triton_strict_adc_scale_accumulate,
)


def strict_torch_postprocess(partial, scale, accumulated, *, adc_ref, radc):
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(17)
    device = torch.device("cuda")
    source = torch.randn(7, 5, 3, 11, device=device, dtype=torch.bfloat16)
    scale = torch.tensor(0.03125, device=device, dtype=torch.float32)

    for non_contiguous, with_accumulator in itertools.product((False, True), repeat=2):
        partial = (
            source.select(2, 1)
            if non_contiguous
            else source[:, :, 1].contiguous()
        )
        accumulated = (
            torch.randn(partial.shape, device=device, dtype=torch.float32)
            if with_accumulator
            else None
        )
        expected = strict_torch_postprocess(
            partial.clone(),
            scale,
            accumulated.clone() if accumulated is not None else None,
            adc_ref=2.75,
            radc=256,
        )
        actual = triton_strict_adc_scale_accumulate(
            partial,
            scale,
            accumulated,
            adc_ref=2.75,
            radc=256,
        )
        torch.cuda.synchronize(device)
        if not torch.equal(actual, expected):
            max_abs = (actual - expected).abs().max().item()
            raise AssertionError(
                f"strict postprocess mismatch: non_contiguous={non_contiguous}, "
                f"with_accumulator={with_accumulator}, max_abs={max_abs}"
            )

    print("strict_postprocess_cuda_equal=true cases=4")


if __name__ == "__main__":
    main()
