from __future__ import annotations

import torch

from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def _engine(read_var: float, *, policy: str = "all") -> DPETensorMultiMode:
    return DPETensorMultiMode(
        write_variation=0.0,
        read_variation=read_var,
        vnoise=0.0,
        rate_stuck_HGS=0.0,
        rate_stuck_LGS=0.0,
        g_level=16,
        mode=1,
        mode1_paral_size=(64, 64),
        fast_inference=True,
        fast_inference_backend="triton_gidx",
        triton_mode1_gdiff_direct_final=True,
        mode1_gdiff_policy=policy,
        device=torch.device("cpu"),
    )


def _weight(engine: DPETensorMultiMode) -> SlicedDataMultiMode:
    torch.manual_seed(13)
    data = torch.randn((128, 192), dtype=torch.float32)
    weight = SlicedDataMultiMode(
        torch.tensor([1]),
        is_weight=True,
        paral_size=(64, 64),
        quant_gran=(64, 64),
        device=torch.device("cpu"),
        inference=True,
        mode=1,
    )
    weight.slice_data_imp(engine, data)
    return weight


def test_mode1_zero_variation_stores_one_signed_difference_index():
    weight = _weight(_engine(read_var=0.0))

    assert weight.mode1_gdiff_indices is not None
    assert weight.mode1_gdiff_indices.dtype == torch.int8
    assert weight.G_indices is None
    assert int(weight.mode1_gdiff_indices.min()) >= -15
    assert int(weight.mode1_gdiff_indices.max()) <= 15


def test_mode1_read_variation_keeps_independent_branch_indices():
    weight = _weight(_engine(read_var=0.05))

    assert weight.mode1_gdiff_indices is None
    assert isinstance(weight.G_indices, tuple)
    assert len(weight.G_indices) == 2


def test_mode1_wide_policy_keeps_regular_layers_on_pair_indices():
    weight = _weight(_engine(read_var=0.0, policy="wide"))

    assert weight.mode1_gdiff_indices is None
    assert isinstance(weight.G_indices, tuple)


def test_mode1_torch_reference_runs_from_signed_difference_indices():
    engine = _engine(read_var=0.0)
    engine.fast_inference_backend = "torch"
    weight = _weight(engine)
    activation = SlicedDataMultiMode(
        torch.tensor([1]),
        is_weight=False,
        paral_size=(64, 64),
        quant_gran=(64, 64),
        device=torch.device("cpu"),
        inference=True,
        mode=1,
    )
    activation.slice_data_imp(engine, torch.randn((4, 128), dtype=torch.float32))

    output = engine(activation, weight)

    assert output.shape == (4, 192)
    assert torch.isfinite(output).all()
