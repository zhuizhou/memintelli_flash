from __future__ import annotations

import pytest
import torch

from examples.mode1_differential_benchmark import (
    compare_mode1_outputs,
    run_mode1_comparison,
)


def test_compare_mode1_outputs_reports_exact_match():
    output = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    metrics = compare_mode1_outputs(output, output.clone())

    assert metrics["finite"] is True
    assert metrics["torch_equal"] is True
    assert metrics["max_abs"] == 0.0
    assert metrics["mean_abs"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["equal_fraction"] == 1.0
    assert metrics["cosine_similarity"] == pytest.approx(1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape",
    [
        (128, 512, 512),
        (128, 512, 1536),
        (128, 1536, 512),
        (128, 512, 4097),
    ],
)
def test_mode1_bf16_matches_reference_at_zero_variation(shape):
    metrics = run_mode1_comparison(
        shape,
        read_var=0.0,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    assert metrics["finite"]
    assert metrics["relative_l2"] < 0.1
    assert metrics["cosine_similarity"] > 0.995
    counters = metrics["fastpath_counters"]
    assert counters["mode1_gidx_direct_final_success_count"] > 0
    assert counters["mode1_gidx_direct_final_fallback_count"] == 0
