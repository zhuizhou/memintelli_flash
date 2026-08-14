from types import SimpleNamespace

from memintelli.NN_layers.linear import _input_chunk_max_positions
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode


def test_explicit_unlimited_input_chunk_does_not_inherit_output_tile_chunk():
    engine = SimpleNamespace(
        inference_chunk_size=256,
        inference_input_chunk_size=0,
    )

    max_positions = _input_chunk_max_positions(
        engine,
        in_features=2560,
        total_positions=128,
    )

    assert max_positions == 128


def test_missing_input_chunk_setting_keeps_backward_compatible_budget():
    engine = SimpleNamespace(inference_chunk_size=256)

    max_positions = _input_chunk_max_positions(
        engine,
        in_features=2560,
        total_positions=128,
    )

    assert max_positions == 1


def test_mode0_output_chunk_limit_can_be_overridden_by_the_weight_state():
    engine = object.__new__(DPETensorMultiMode)
    engine.triton_output_chunk_limit = 256
    engine.triton_auto_config = False
    engine.mode = 0
    mat = SimpleNamespace(triton_output_chunk_limit_override=1024)

    assert engine._mode0_triton_chunk_limit(4096, mat=mat) == 1024
