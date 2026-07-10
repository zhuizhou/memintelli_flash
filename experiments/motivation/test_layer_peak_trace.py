import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "examples" / "20_real_llm_prefill_benchmark.py"


def load_worker_namespace():
    spec = importlib.util.spec_from_file_location("real_llm_prefill_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)
    return worker


def test_layer_peak_trace_records_incremental_cuda_peak_per_linear():
    worker = load_worker_namespace()
    model = nn.Sequential(nn.Linear(4, 7, bias=False))
    context_class = worker["LayerPeakTraceContext"]

    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "synchronize"),
        mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
        mock.patch.object(torch.cuda, "memory_allocated", side_effect=[100, 110]),
        mock.patch.object(torch.cuda, "max_memory_allocated", return_value=180),
    ):
        with context_class(model, nn.Linear, enabled=True) as trace:
            model(torch.randn(2, 4))

    assert trace.rows == [
        {
            "index": 0,
            "name": "0",
            "in_features": 4,
            "out_features": 7,
            "allocated_before_bytes": 100,
            "allocated_after_bytes": 110,
            "peak_bytes": 180,
            "incremental_peak_bytes": 80,
        }
    ]
