from types import SimpleNamespace

import torch

from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def make_engine(*, exact_reduce):
    engine = object.__new__(DPETensorMultiMode)
    engine.fast_inference_backend = "triton_gidx"
    engine.triton_fuse_restored_input_slices = True
    engine.mode = 0
    engine.radc_is_list = False
    engine.rdac = 2
    engine.vnoise = 0.0
    engine.triton_direct_final_output = True
    engine.triton_direct_final_exact_reduce = exact_reduce
    engine.fast_inference = True
    engine._has_read_noise = True
    engine.read_variation_seed = 1234
    engine._mode0_requires_fp32_analog_compute = lambda: False
    engine._mode0_vmm_uses_low_precision_override = lambda: True
    engine._mode0_fast_policy_requested = lambda: False
    engine._mode0_seeded_semantic_audit = lambda: True
    engine._mode0_analog_compute_dtype = lambda: torch.bfloat16
    engine._fastpath_count = lambda *args, **kwargs: None
    return engine


def make_inputs():
    x = SimpleNamespace(
        shape=(128, 2560),
        sliced_data=SimpleNamespace(is_cuda=True, dim=lambda: 5),
        max_data=SimpleNamespace(dim=lambda: 4),
    )
    return x, SimpleNamespace()


def test_seeded_noisy_g_allows_exact_restored_compaction():
    engine = make_engine(exact_reduce=True)
    x, mat = make_inputs()

    assert engine._can_use_triton_restored_input_slice_fusion(
        x,
        mat,
        allow_seeded_exact=True,
    ) is True


def test_seeded_noisy_g_still_blocks_non_exact_restored_compaction():
    engine = make_engine(exact_reduce=False)
    x, mat = make_inputs()

    assert engine._can_use_triton_restored_input_slice_fusion(x, mat) is False


def test_seeded_noisy_g_allows_the_exact_fast_inference_entrypoint():
    engine = make_engine(exact_reduce=True)

    assert engine._can_use_fast_inference(differential_input=False) is True


def test_seeded_noisy_g_keeps_exact_direct_final_enabled_by_default():
    engine = make_engine(exact_reduce=True)

    assert engine._requires_strict_grouped_noisy_vmm() is False


def test_framework_noisy_vmm_reference_is_explicitly_opt_in():
    engine = make_engine(exact_reduce=True)
    engine.mode0_framework_noisy_vmm_reference = True

    assert engine._requires_strict_grouped_noisy_vmm() is True


def test_seeded_strict_restore_repeats_after_noise_state_reset():
    engine = DPETensorMultiMode(
        read_variation=0.05,
        read_variation_seed=1234,
        write_variation=0.0,
        vnoise=0.0,
        mode=0,
        device=torch.device("cpu"),
    )
    conductance = torch.full((2, 3, 2, 4, 4), 1.0e-6)

    first = engine._apply_read_noise_tensor(conductance.clone())
    engine.reset_fastpath_counters()
    second = engine._apply_read_noise_tensor(conductance.clone())

    assert torch.equal(first, second)


def test_grouped_noisy_vmm_matches_the_strict_weight_slice_order():
    engine = object.__new__(DPETensorMultiMode)
    engine.mode = 0
    engine.adc_compute_dtype = torch.float32
    engine.radc = 256
    engine._profile_start = lambda *args, **kwargs: None
    engine._profile_stop = lambda *args, **kwargs: None
    engine._fastpath_count = lambda *args, **kwargs: None
    torch.manual_seed(9)
    vin = torch.randn(4, 2, 1, 8, dtype=torch.bfloat16)
    conductance = torch.randn(2, 3, 5, 8, 4, dtype=torch.bfloat16)
    scale = torch.randn(5, dtype=torch.float32)
    adc_ref = 3.25

    expected = None
    for weight_slice in range(conductance.shape[2]):
        partial = torch.einsum(
            "nmjk,mpkl->nmpjl",
            vin,
            conductance[:, :, weight_slice],
        ).to(torch.float32)
        partial.div_(adc_ref).mul_(engine.radc - 1).round_().div_(engine.radc - 1)
        partial.mul_(scale[weight_slice])
        if expected is None:
            expected = partial
        else:
            expected.add_(partial)

    actual = engine._strict_grouped_noisy_weight_slice_accumulate(
        vin,
        conductance,
        scale,
        adc_ref,
    )

    assert torch.equal(actual, expected)


def test_grouped_noisy_vmm_shape_dispatch_is_conservative():
    engine = object.__new__(DPETensorMultiMode)
    vin = SimpleNamespace(shape=(1, 40, 1, 64))

    assert engine._strict_grouped_noisy_shape_supported(
        vin,
        SimpleNamespace(shape=(40, 128, 5, 64, 64)),
    ) is True
    assert engine._strict_grouped_noisy_shape_supported(
        vin,
        SimpleNamespace(shape=(40, 64, 5, 64, 64)),
    ) is False
    assert engine._strict_grouped_noisy_shape_supported(
        vin,
        SimpleNamespace(shape=(40, 512, 3, 64, 64)),
    ) is True
    assert engine._strict_grouped_noisy_shape_supported(
        vin,
        SimpleNamespace(shape=(40, 640, 3, 64, 64)),
    ) is False
    assert engine._strict_grouped_noisy_shape_supported(
        SimpleNamespace(shape=(1, 64, 1, 64)),
        SimpleNamespace(shape=(64, 512, 5, 64, 64)),
    ) is True
