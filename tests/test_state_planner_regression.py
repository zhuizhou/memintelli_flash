from memintelli.NN_layers.state_planner import plan_output_block


def test_mode0_unlimited_plan_preserves_full_layer_behavior():
    plan = plan_output_block(
        mode=0,
        tokens=128,
        in_features=4096,
        out_features=12288,
        input_slices=5,
        weight_slices=5,
        array_rows=64,
        array_cols=64,
        cuda_peak_budget_mb=0,
        read_variation=0.05,
        execution_strategy="direct_final",
        output_chunk_tiles=256,
    )

    assert plan.output_block_cols == 12288
    assert plan.shard_count == 1
    assert plan.full_layer_peak_bytes == plan.estimated_peak_bytes
