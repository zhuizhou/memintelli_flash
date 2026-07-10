import argparse
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


class FakeWeightSliced:
    def __init__(self):
        self.inference = False
        self.G_indices = None
        self.G = None
        self.max_data = None
        self.e_bias = None
        self.mode1_w_max = None


class FakeLinearMem(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, cols))
        self.register_parameter("bias", None)
        self.weight_slice_method = torch.tensor([1, 1, 1, 1, 1])
        self.input_slice_method = torch.tensor([1, 1, 1, 1, 1])
        self.weight_sliced = FakeWeightSliced()
        self.engine = types.SimpleNamespace(device=torch.device("cpu"), mode=0)
        self.prepare_count = 0
        self._pinned_buffers = {}
        self._active_pinned_buffers = {}

    def enable_lazy_inference(self, **kwargs):
        return self

    def _prepare_inference_weight(self, **kwargs):
        self.prepare_count += 1
        self.weight_sliced.G_indices = torch.zeros(
            self.weight.numel() * len(self.weight_slice_method),
            dtype=torch.uint8,
        )
        return self

    def release_prepared_weight(self):
        self.weight_sliced.G_indices = None


def test_analytical_residency_does_not_prepare_rejected_candidates():
    worker = load_worker_namespace()
    large = FakeLinearMem(100, 100)
    small = FakeLinearMem(10, 10)
    model = nn.ModuleList([large, small])
    args = argparse.Namespace(
        streaming_window_pin_cache_mb=0.0,
        streaming_persistent_pin_budget_mb=0.0,
        streaming_persistent_pin_select="sequential",
        streaming_pin_hints_json="",
        memory_resident_budget_mb=0.052,
        memory_resident_select="largest",
        lazy_prepare=True,
        free_weights=False,
        lazy_release_after_forward=True,
        streaming=False,
        streaming_pin_policy="persistent",
        memory_empty_cache_after_offload=False,
        memory_empty_cache_after_offload_interval=1,
        weight_slice=[1, 1, 1, 1, 1],
        mode=0,
        kind="v3",
    )

    info = worker["prepare_mem_model"](
        model,
        FakeLinearMem,
        args,
        torch.device("cpu"),
    )

    assert large.prepare_count == 1
    assert small.prepare_count == 0
    assert info["memory_resident_layers"] == 1
