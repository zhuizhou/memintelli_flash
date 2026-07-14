from __future__ import annotations

import pytest
import torch

from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def _engine(
    device: torch.device,
    *,
    backend: str,
    require_fastpath: bool,
    read_var: float = 0.0,
    gdiff_direct: bool = False,
):
    return DPETensorMultiMode(
        HGS=1e-5,
        LGS=1e-7,
        g_level=16,
        write_variation=0.0,
        read_variation=float(read_var),
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
        triton_mode1_gdiff_direct_final=bool(gdiff_direct),
        mode1_gdiff_schedule="auto",
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


def test_mode1_precomputed_voltage_matches_pair_kernel_order():
    engine = _engine(torch.device("cpu"), backend="torch", require_fastpath=False)
    x = torch.tensor(
        [[-1.0, -0.37, 0.0, 0.41, 0.93]],
        dtype=torch.float32,
    )
    x_max = x.abs().max().clamp_min(torch.finfo(torch.float32).tiny)

    actual = engine._prepare_mode1_signed_voltage(x, x_max)
    expected = (
        torch.round(x / x_max * (engine.rdac - 1))
        * (engine.vread / (engine.rdac - 1))
    ).to(engine.compute_dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mode1_precomputed_voltage_counter_is_registered():
    engine = _engine(torch.device("cpu"), backend="torch", require_fastpath=False)
    counters = engine.get_fastpath_counters()

    assert "mode1_precomputed_v_success_count" in counters


def test_mode1_gdiff_direct_final_options_and_counters_are_registered():
    engine = DPETensorMultiMode(
        mode=1,
        triton_mode1_gdiff_direct_final=True,
        mode1_gdiff_schedule="owner",
        device=torch.device("cpu"),
    )
    counters = engine.get_fastpath_counters()

    assert engine.triton_mode1_gdiff_direct_final is True
    assert engine.mode1_gdiff_schedule == "owner"
    assert "mode1_gdiff_restore_attempt_count" in counters
    assert "mode1_gdiff_restore_success_count" in counters
    assert "mode1_gdiff_restore_fallback_count" in counters
    assert "mode1_gdiff_direct_final_attempt_count" in counters
    assert "mode1_gdiff_direct_final_success_count" in counters
    assert "mode1_gdiff_direct_final_fallback_count" in counters
    assert "mode1_gdiff_owner_success_count" in counters
    assert "mode1_gdiff_grouped_success_count" in counters


def test_mode1_gdiff_dispatch_is_an_explicit_engine_method():
    engine = _engine(
        torch.device("cpu"),
        backend="torch",
        require_fastpath=False,
        gdiff_direct=True,
    )

    assert hasattr(engine, "_triton_mode1_gdiff_direct_final_output")
    assert hasattr(engine, "_mode1_gdiff_workspace_view")


def test_mode1_gdiff_workspace_uses_geometric_capacity_and_reuses_it():
    engine = _engine(
        torch.device("cpu"),
        backend="torch",
        require_fastpath=False,
        gdiff_direct=True,
    )

    first = engine._mode1_gdiff_workspace_view((3, 5), torch.bfloat16, torch.device("cpu"))
    second = engine._mode1_gdiff_workspace_view((2, 4), torch.bfloat16, torch.device("cpu"))

    assert tuple(first.shape) == (3, 5)
    assert tuple(second.shape) == (2, 4)
    assert engine._mode1_gdiff_workspace.numel() == 16
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def test_mode1_gdiff_auto_dispatch_uses_measured_shape_rule():
    engine = _engine(
        torch.device("cpu"),
        backend="torch",
        require_fastpath=False,
        gdiff_direct=True,
    )

    qkv = engine._mode1_gdiff_launch_plan(128, 4096, 4096, 64, 64)
    down = engine._mode1_gdiff_launch_plan(128, 12288, 4096, 64, 64)
    lm_head = engine._mode1_gdiff_launch_plan(128, 4096, 248320, 64, 64)

    assert (qkv["block_r"], qkv["block_l"], qkv["schedule"], qkv["input_tile_group"]) == (
        64,
        32,
        "owner",
        64,
    )
    assert (down["block_r"], down["block_l"], down["schedule"], down["input_tile_group"]) == (
        64,
        32,
        "grouped",
        4,
    )
    assert (lm_head["block_r"], lm_head["block_l"], lm_head["schedule"], lm_head["input_tile_group"]) == (
        64,
        32,
        "owner",
        64,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_precomputed_voltage_preserves_pair_direct_output():
    from memintelli.pimpy.triton_fast_accumulate import triton_mode1_gidx_direct_final

    device = torch.device("cuda")
    torch.manual_seed(11)
    x = torch.randn((128, 512), device=device, dtype=torch.bfloat16)
    weight = torch.randn((512, 1536), device=device, dtype=torch.bfloat16)
    engine = _engine(device, backend="triton_gidx", require_fastpath=False)
    x_sliced = _sliced(engine, x, is_weight=False)
    weight_sliced = _sliced(engine, weight, is_weight=True)
    x_2d = x_sliced.quantized_data
    x_max = x_2d.abs().max().clamp_min(torch.finfo(x_2d.dtype).tiny)
    gp_idx, gn_idx = weight_sliced.G_indices
    scale = weight_sliced.mode1_w_max.float()
    voltage = engine._prepare_mode1_signed_voltage(x_2d, x_max)
    kwargs = dict(
        x_max=x_max,
        lgs=engine.LGS,
        q_g=engine.Q_G,
        read_sigma=0.0,
        adc_ref_unit=(engine.HGS - engine.LGS) * engine.vread,
        rdac=engine.rdac,
        radc=engine.radc,
        vread=engine.vread,
        g_level=engine.g_level,
        tile_in=64,
        tile_out=64,
        input_precision="ieee",
        dot_dtype_override=2,
        block_r=64,
        block_l=16,
        block_k=64,
        input_tile_group=1,
    )

    inline = triton_mode1_gidx_direct_final(
        x_2d, gp_idx, gn_idx, scale, **kwargs
    )
    precomputed = triton_mode1_gidx_direct_final(
        x_2d,
        gp_idx,
        gn_idx,
        scale,
        precomputed_v=voltage,
        **kwargs,
    )

    torch.testing.assert_close(precomputed, inline, rtol=1e-5, atol=1e-5)


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
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_noisy_gdiff_direct_final_hits_without_pair_fallback():
    device = torch.device("cuda")
    torch.manual_seed(23)
    x = torch.randn((32, 128), device=device, dtype=torch.bfloat16)
    weight = torch.randn((128, 192), device=device, dtype=torch.bfloat16)
    optimized = _engine(
        device,
        backend="triton_gidx",
        require_fastpath=True,
        read_var=0.05,
        gdiff_direct=True,
    )
    x_sliced = _sliced(optimized, x, is_weight=False)
    weight_sliced = _sliced(optimized, weight, is_weight=True)

    actual = optimized(x_sliced, weight_sliced)
    counters = optimized.get_fastpath_counters()

    assert torch.isfinite(actual).all()
    assert counters["mode1_gdiff_restore_success_count"] > 0
    assert counters["mode1_gdiff_restore_fallback_count"] == 0
    assert counters["mode1_gdiff_direct_final_success_count"] > 0
    assert counters["mode1_gdiff_direct_final_fallback_count"] == 0
    assert counters["mode1_gidx_direct_final_success_count"] == 0
