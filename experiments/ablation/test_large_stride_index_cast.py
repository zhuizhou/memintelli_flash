import importlib
from types import SimpleNamespace

import torch


def test_segmented_index_cast_matches_regular_cast(monkeypatch):
    module = importlib.import_module("memintelli.pimpy.memmat_tensor_multimode")
    monkeypatch.setattr(module, "_INDEX_CAST_INT32_LIMIT", 1)

    source = torch.arange(4 * 3 * 2, dtype=torch.uint8).reshape(4, 3, 2)
    result = module._cast_compressed_index_chunk(source, torch.bfloat16)

    assert result.dtype == torch.bfloat16
    assert torch.equal(result, source.to(torch.bfloat16))


def test_large_stride_span_requires_segmented_index_cast(monkeypatch):
    module = importlib.import_module("memintelli.pimpy.memmat_tensor_multimode")
    monkeypatch.setattr(module, "_INDEX_CAST_INT32_LIMIT", 100)

    source = torch.empty_strided((4, 2), (40, 1), dtype=torch.uint8)

    assert module._compressed_index_cast_needs_segmentation(source)


def test_contiguous_small_span_keeps_regular_cast():
    module = importlib.import_module("memintelli.pimpy.memmat_tensor_multimode")
    source = torch.zeros((4, 3, 2), dtype=torch.uint8)

    assert not module._compressed_index_cast_needs_segmentation(source)


def test_requested_5d_restore_uses_int64_strided_kernel_even_if_contiguous():
    module = importlib.import_module("memintelli.pimpy.triton_fast_accumulate")
    source = torch.zeros((2, 3, 4, 5, 6), dtype=torch.uint8)

    assert module._should_use_strided_gidx_restore(source, strided=True)


def test_framework_loop_applies_only_address_safety_cap_to_compressed_restore():
    module = importlib.import_module("memintelli.pimpy.memmat_tensor_multimode")
    engine = object.__new__(module.DPETensorMultiMode)
    engine.mode = 0
    engine.inference_chunk_size = 4096
    engine.fast_inference_backend = "torch"
    engine.triton_auto_config = False
    engine.triton_output_chunk_limit = 256
    mat = SimpleNamespace(
        max_data=torch.empty((1, 3000, 1, 1), device="meta"),
        e_bias=None,
        G_indices=torch.empty((40, 3000, 5, 64, 64), dtype=torch.uint8, device="meta"),
        G=None,
    )

    chunks = list(engine._iter_output_chunks(mat))
    widths = [end - start for start, end in chunks]

    assert max(widths) <= 2441
    assert max(widths) > 256
