import argparse
import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock

import torch.nn as nn
import torch


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "examples" / "20_real_llm_prefill_benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("real_llm_prefill_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def planner_args(stage, *, cuda_budget_mb=0.0, output_block_cols=0):
    return argparse.Namespace(
        kind="v3",
        mode=0,
        s1_stage=stage,
        s1_block_addressable=stage != "off",
        state_planner="analytical" if stage == "budgeted" else "off",
        cuda_peak_budget_mb=cuda_budget_mb,
        state_resident_budget_mb=0.0,
        output_block_cols=output_block_cols,
        batch=1,
        seq=128,
        input_slice=[1, 1, 1, 1, 1],
        weight_slice=[1, 1, 1, 1, 1],
        input_paral_size=[1, 64],
        weight_paral_size=[64, 64],
    )


def test_s1_off_never_blocks_a_linear():
    module = load_benchmark_module()
    layer = nn.Linear(2560, 248320, bias=False)

    plan = module.plan_linear_output_blocks(
        planner_args("off", cuda_budget_mb=4096.0),
        layer,
    )

    assert plan.output_block_cols == layer.out_features
    assert plan.shard_count == 1


def test_worker_s1_off_does_not_require_the_v3_state_planner_module():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    layer = nn.Linear(2560, 248320, bias=False)
    with mock.patch.dict(sys.modules, {"memintelli.NN_layers.state_planner": None}):
        plan = worker["plan_linear_output_blocks"](
            planner_args("off", cuda_budget_mb=4096.0),
            layer,
        )

    assert plan.output_block_cols == layer.out_features
    assert plan.shard_count == 1


def test_s1_block_uses_only_the_manual_debug_override():
    module = load_benchmark_module()
    layer = nn.Linear(2560, 248320, bias=False)

    plan = module.plan_linear_output_blocks(
        planner_args("block", output_block_cols=16001),
        layer,
    )

    assert plan.output_block_cols == 16064
    assert plan.shard_count > 1
    assert plan.manual_override is True


def test_s1_budgeted_derives_an_aligned_block_without_a_kernel_sweep():
    module = load_benchmark_module()
    layer = nn.Linear(2560, 248320, bias=False)
    args = planner_args("budgeted", cuda_budget_mb=4096.0)
    args.planner_base_allocated_mb = 512.0
    args.planner_safety_margin_mb = 256.0

    plan = module.plan_linear_output_blocks(
        args,
        layer,
    )

    assert plan.output_block_cols % 64 == 0
    assert plan.output_block_cols < layer.out_features
    assert plan.shard_count >= 2
    assert plan.manual_override is False
    assert plan.base_allocated_mb == 512.0
    assert plan.safety_margin_mb == 256.0
    assert plan.workspace_budget_mb == 3328.0


def test_worker_counts_only_non_linear_resident_model_bytes():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    model = nn.Module()
    model.embed = nn.Embedding(10, 4)
    model.norm = nn.LayerNorm(4)
    model.proj = nn.Linear(4, 7, bias=False)

    expected = (
        model.embed.weight.numel() * model.embed.weight.element_size()
        + model.norm.weight.numel() * model.norm.weight.element_size()
        + model.norm.bias.numel() * model.norm.bias.element_size()
    )

    assert worker["estimate_non_linear_model_bytes"](model) == expected


def test_common_input_coalescing_accepts_seeded_read_variation():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    args = argparse.Namespace(
        kind="v3",
        mode=0,
        write_variation=0.0,
        read_variation=0.05,
        read_variation_seed=1234,
        weight_quant_gran=[64, 64],
        weight_paral_size=[64, 64],
    )
    linears = [
        nn.Linear(128, 128, bias=False),
        nn.Linear(128, 64, bias=False),
        nn.Linear(128, 64, bias=False),
    ]

    assert worker["can_fuse_common_input_projection_group"](args, linears) is True


def test_common_input_group_keeps_independent_weight_mappings():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    class FakeEngine:
        device = torch.device("cpu")

    class FakeLinearMem(nn.Module):
        def __init__(self, engine, in_features, out_features, bias, device, dtype, **kwargs):
            super().__init__()
            self.engine = engine
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
            self.call_count = 0

        def forward(self, value):
            self.call_count += 1
            return torch.nn.functional.linear(value, self.weight, self.bias)

    args = argparse.Namespace(
        kind="v3",
        mode=0,
        input_slice=[1, 1],
        weight_slice=[1, 1],
        mode2_input_slice=[1, 1],
        mode2_weight_slice=[1, 1],
        input_paral_size=[1, 4],
        weight_paral_size=[4, 4],
        input_quant_gran=[1, 4],
        weight_quant_gran=[4, 4],
    )
    q = nn.Linear(4, 4, bias=False)
    k = nn.Linear(4, 2, bias=False)
    owner = worker["FusedProjectionGroupLinearMem"](
        FakeLinearMem,
        args,
        FakeEngine(),
        [("q_proj", q), ("k_proj", k)],
        torch.device("cpu"),
        True,
    )
    value = torch.randn(3, 4)

    q_out = owner.project(0, value)
    k_out = owner.project(1, value)

    assert len(owner.inners) == 2
    assert torch.equal(owner.inners[0].weight, q.weight)
    assert torch.equal(owner.inners[1].weight, k.weight)
    assert owner.inners[0].call_count == 1
    assert owner.inners[1].call_count == 1
    torch.testing.assert_close(q_out, q(value), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(k_out, k(value), rtol=0.0, atol=1e-6)


def test_common_input_group_preserves_s1_output_blocking():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    class FakeEngine:
        device = torch.device("cpu")

    class FakeLinearMem(nn.Module):
        def __init__(self, engine, in_features, out_features, bias, device, dtype, **kwargs):
            super().__init__()
            self.engine = engine
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            self.register_parameter("bias", None)

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight, self.bias)

    args = argparse.Namespace(
        kind="v3",
        mode=0,
        s1_stage="block",
        s1_block_addressable=True,
        state_planner="off",
        output_block_cols=4,
        cuda_peak_budget_mb=0.0,
        state_resident_budget_mb=0.0,
        batch=1,
        seq=8,
        input_slice=[1, 1],
        weight_slice=[1, 1],
        mode2_input_slice=[1, 1],
        mode2_weight_slice=[1, 1],
        input_paral_size=[1, 4],
        weight_paral_size=[4, 4],
        input_quant_gran=[1, 4],
        weight_quant_gran=[4, 4],
    )
    q = nn.Linear(4, 9, bias=False)
    k = nn.Linear(4, 5, bias=False)
    owner = worker["FusedProjectionGroupLinearMem"](
        FakeLinearMem,
        args,
        FakeEngine(),
        [("q_proj", q), ("k_proj", k)],
        torch.device("cpu"),
        True,
    )

    assert owner.inners[0].shard_count == 3
    assert owner.inners[1].shard_count == 2


def test_worker_replaces_a_non_lm_head_with_output_blocks():
    module = load_benchmark_module()
    worker = {"__name__": "worker_test"}
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = object
    transformers.AutoTokenizer = object
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        exec(module.WORKER_CODE, worker)

    class FakeEngine:
        device = torch.device("cpu")

    class FakeLinearMem(nn.Module):
        def __init__(
            self,
            engine,
            in_features,
            out_features,
            input_slice,
            weight_slice,
            bias,
            device,
            dtype,
            bw_e,
            input_paral_size,
            weight_paral_size,
            input_quant_gran,
            weight_quant_gran,
            skip_initial_mapping=False,
        ):
            super().__init__()
            self.engine = engine
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)

        def forward(self, value):
            return torch.nn.functional.linear(value, self.weight, self.bias)

    args = planner_args("block", output_block_cols=4)
    args.mode2_input_slice = args.input_slice
    args.mode2_weight_slice = args.weight_slice
    args.input_paral_size = [1, 4]
    args.weight_paral_size = [4, 4]
    args.input_quant_gran = [1, 4]
    args.weight_quant_gran = [4, 4]
    args.fuse_common_input_projections = False
    args.fuse_mlp_gate_up = False
    args.max_linears = None
    args.max_non_lm_head_linears = None
    args.linear_name_limit = 16
    args.only_lm_head = False
    args.simulate_lm_head = True
    args.always_include_lm_head = True
    args.linear_include_regex = []
    args.linear_exclude_regex = []
    args.lm_head_output_token_ids = []
    args.lm_head_output_shards = 1
    args.lm_head_input_select = "all"

    model = nn.Module()
    model.proj = nn.Linear(6, 9, bias=False)
    original_weight = model.proj.weight.detach().clone()
    replace_state = {
        "replaced": 0,
        "replaced_names": [],
        "non_lm_head_replaced": 0,
        "lm_head_replaced": 0,
        "fused_gate_up_groups": 0,
        "fused_common_projection_groups": 0,
        "fused_common_projection_linears": 0,
    }
    replaced, skipped = worker["replace_linear"](
        model,
        FakeLinearMem,
        args,
        FakeEngine(),
        torch.device("cpu"),
        state=replace_state,
    )

    assert replaced == 1
    assert skipped == 0
    assert model.proj.shard_count == 3
    assert model.proj.block_ranges == [(0, 4), (4, 8), (8, 9)]
    assert torch.equal(model.proj.blocks[0].weight, original_weight[:4])
    assert torch.equal(model.proj.blocks[2].weight, original_weight[8:])
    assert replace_state["output_blocked_linears"] == 1
