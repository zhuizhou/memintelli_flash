from types import SimpleNamespace

import torch

from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode
from memintelli.pimpy.triton_fast_accumulate import (
    _normalize_runtime_noise_address,
    _seeded_runtime_noise_offset,
)


def test_seed_alone_does_not_enable_strict_semantic_audit():
    engine = object.__new__(DPETensorMultiMode)
    engine.mode = 0
    engine.mode0_semantic_policy = "auto"
    engine.read_variation_seed = 1234

    assert engine._mode0_seeded_semantic_audit() is False


def test_explicit_strict_policy_enables_semantic_audit():
    engine = object.__new__(DPETensorMultiMode)
    engine.mode = 0
    engine.mode0_semantic_policy = "strict"
    engine.read_variation_seed = 1234

    assert engine._mode0_seeded_semantic_audit() is True


def test_seeded_read_variation_keeps_wide_triton_output_chunking():
    engine = object.__new__(DPETensorMultiMode)
    engine.mode = 0
    engine.fast_inference_backend = "triton_gidx"
    engine.read_variation_seed = 1234
    engine.inference_chunk_size = None
    engine.triton_auto_config = False
    engine.triton_output_chunk_limit = 256
    mat = SimpleNamespace(
        max_data=SimpleNamespace(shape=(40, 3880)),
        G_indices=SimpleNamespace(shape=(40, 3880, 5, 64, 64)),
    )

    chunks = list(engine._iter_output_chunks(mat))

    assert len(chunks) > 1
    assert max(c1 - c0 for c0, c1 in chunks) <= 256


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


def test_mode0_noise_epochs_are_disjoint_for_different_layer_sizes():
    engine = object.__new__(DPETensorMultiMode)
    engine._read_noise_forward_offset_counter = 0

    large = SimpleNamespace(G_indices=SimpleNamespace(dim=lambda: 5, numel=lambda: 100))
    small = SimpleNamespace(G_indices=SimpleNamespace(dim=lambda: 5, numel=lambda: 40))

    large_base = engine._next_mode0_read_noise_epoch_base(large)
    small_base = engine._next_mode0_read_noise_epoch_base(small)

    assert large_base == 0
    assert small_base == 100


def test_runtime_noise_address_preserves_large_offsets_and_seed_identity():
    large_offset = (1 << 40) + 17

    seed, offset = _normalize_runtime_noise_address(987654321, large_offset)

    assert seed == 987654321
    assert offset == large_offset


def test_paper_noise_seeds_map_to_disjoint_large_counter_windows():
    offsets = [_seeded_runtime_noise_offset(seed, 0) for seed in range(1001, 1011)]
    window_span = 6_000_000_000

    for index, left in enumerate(offsets):
        for right in offsets[index + 1 :]:
            assert abs(left - right) > window_span


def test_seeded_runtime_noise_offset_preserves_the_layer_local_offset():
    mask = (1 << 63) - 1
    seed_base = _seeded_runtime_noise_offset(1001, 0)
    layer_offset = (1 << 40) + 29

    actual = _seeded_runtime_noise_offset(1001, layer_offset)

    assert (actual - seed_base) & mask == layer_offset


def test_seeded_restore_prefetch_reaches_normal_eligibility_checks():
    engine = object.__new__(DPETensorMultiMode)
    engine.read_variation_seed = 1234
    eligibility_checks = []
    engine._fastpath_count = lambda *args, **kwargs: None
    engine._can_schedule_mode0_restore_input_prefetch = (
        lambda *args, **kwargs: eligibility_checks.append(True) or False
    )

    scheduled = engine.schedule_mode0_restore_input_prefetch(
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert scheduled is False
    assert eligibility_checks == [True]


def test_seeded_restore_prefetch_is_eligible_with_stable_noise_addresses(monkeypatch):
    engine = object.__new__(DPETensorMultiMode)
    engine.triton_overlap_restore_direct = True
    engine.fast_inference = True
    engine.triton_direct_final_output = True
    engine.fast_inference_backend = "triton_gidx"
    engine.mode = 0
    engine._has_read_noise = True
    engine._rv_all_same = True
    engine._rv_sigma = 0.05
    engine.read_variation_seed = 1234
    engine.triton_gidx_fused_restore_read_noise = True
    engine.triton_gidx_read_noise = False
    engine.profile = False
    engine._write_variation_is_virtual = lambda: False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    input_2d = SimpleNamespace(dim=lambda: 2, is_cuda=True)
    mat = SimpleNamespace(
        G_is_compressed=True,
        G_indices=SimpleNamespace(is_cuda=True),
    )

    assert engine._can_schedule_mode0_restore_input_prefetch(input_2d, mat) is True


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
