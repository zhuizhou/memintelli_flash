import torch
import torch.nn as nn
import gc
import weakref

from memintelli.NN_layers.output_blocking import OutputBlockedLinearMem


def make_blocked_linear(full):
    split = 4
    first = nn.Linear(full.in_features, split, bias=False)
    second = nn.Linear(full.in_features, full.out_features - split, bias=False)
    with torch.no_grad():
        first.weight.copy_(full.weight[:split])
        second.weight.copy_(full.weight[split:])
    return OutputBlockedLinearMem(
        blocks=[first, second],
        block_ranges=[(0, split), (split, full.out_features)],
        in_features=full.in_features,
        out_features=full.out_features,
    )


def test_output_blocked_linear_matches_whole_linear_for_3d_input():
    torch.manual_seed(7)
    full = nn.Linear(6, 9, bias=False)
    blocked = make_blocked_linear(full)
    x = torch.randn(2, 3, 6)

    expected = full(x)
    actual = blocked(x)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_output_blocked_linear_exposes_each_inner_block():
    full = nn.Linear(6, 9, bias=False)
    blocked = make_blocked_linear(full)

    assert list(blocked.iter_blocks()) == list(blocked.blocks)
    assert blocked.block_ranges == [(0, 4), (4, 9)]
    assert blocked.shard_count == 2


def test_sequential_blocks_release_each_transient_output_before_the_next_block():
    state = {}

    class FirstBlock(nn.Module):
        def forward(self, value):
            output = value[..., :2].clone()
            state["first_output"] = weakref.ref(output)
            return output

    class SecondBlock(nn.Module):
        def forward(self, value):
            gc.collect()
            assert state["first_output"]() is None
            return value[..., 2:4].clone()

    blocked = OutputBlockedLinearMem(
        blocks=[FirstBlock(), SecondBlock()],
        block_ranges=[(0, 2), (2, 4)],
        in_features=4,
        out_features=4,
    )

    output = blocked(torch.randn(3, 4))

    assert output.shape == (3, 4)
