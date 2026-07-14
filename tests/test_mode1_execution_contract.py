from __future__ import annotations

import pytest
import torch

from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def _engine(device: torch.device, *, backend: str, require_fastpath: bool):
    return DPETensorMultiMode(
        HGS=1e-5,
        LGS=1e-7,
        g_level=16,
        write_variation=0.0,
        read_variation=0.0,
        vnoise=0.0,
        rate_stuck_HGS=0.0,
        rate_stuck_LGS=0.0,
        rdac=16,
        radc=256,
        mode=1,
        mode1_paral_size=(64, 64),
        mode1_adc_per_tile=True,
        fast_inference=True,
        fast_inference_backend=backend,
        triton_mode1_gidx_direct_final=True,
        triton_mode1_chunked_direct_final=True,
        triton_mode1_input_tile_group=1,
        mode1_grouped_tile_gemm=False,
        mode1_require_fastpath=require_fastpath,
        conductance_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        device=device,
    )


def _sliced(
    engine: DPETensorMultiMode,
    data: torch.Tensor,
    *,
    is_weight: bool,
) -> SlicedDataMultiMode:
    sliced = SlicedDataMultiMode(
        torch.tensor([1], device=data.device),
        is_weight=is_weight,
        paral_size=(64, 64),
        quant_gran=(64, 64),
        device=data.device,
        inference=True,
        mode=1,
    )
    sliced.slice_data_imp(engine, data)
    return sliced


def test_mode1_require_fastpath_is_an_explicit_engine_option():
    engine = _engine(torch.device("cpu"), backend="torch", require_fastpath=True)
    assert engine.mode1_require_fastpath is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_direct_final_hits_without_fallback():
    device = torch.device("cuda")
    torch.manual_seed(7)
    x = torch.randn((32, 128), device=device, dtype=torch.bfloat16)
    weight = torch.randn((128, 192), device=device, dtype=torch.bfloat16)

    optimized = _engine(device, backend="triton_gidx", require_fastpath=True)
    reference = _engine(device, backend="torch", require_fastpath=False)
    x_sliced = _sliced(optimized, x, is_weight=False)
    weight_sliced = _sliced(optimized, weight, is_weight=True)

    expected = reference(x_sliced, weight_sliced)
    actual = optimized(x_sliced, weight_sliced)
    counters = optimized.get_fastpath_counters()

    assert counters["mode1_gidx_direct_final_success_count"] > 0
    assert counters["mode1_gidx_direct_final_fallback_count"] == 0
    delta = (actual.float() - expected.float()).reshape(-1)
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    ).item()
    assert torch.isfinite(actual).all()
    assert relative_l2.item() < 0.1
    assert cosine > 0.995
