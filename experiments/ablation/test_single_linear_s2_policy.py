import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("run_single_linear_s2_check.py")


def load_runner_module():
    spec = importlib.util.spec_from_file_location("single_linear_s2_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_intra_linear_uses_exact_reduction_for_bf16_oracle():
    module = load_runner_module()

    engine = module.make_engine("intra", torch.device("cpu"), 0.0, 1234)

    assert engine.triton_direct_final_exact_reduce is True


def test_noisy_intra_linear_uses_high_performance_direct_final():
    module = load_runner_module()

    engine = module.make_engine("intra", torch.device("cpu"), 0.05, 1234)

    assert engine.triton_direct_final_exact_reduce is False
