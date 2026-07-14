# v2.2 Release Manifest

- Source baseline: local v2.1 DAC runtime at commit `f54e4df`, plus audited
  S1/S2 runtime fixes from the current research tree.
- Modes: optimized mode0 bit-serial and mode1 differential-pair inference.
- Mode1 reference: pair-index BF16 VMM with FP32 ADC and reduction.
- Mode1 optimized path: mode-aware state planning, precomputed signed DAC,
  independent noisy `G+`/`G-` restore into a budget-selected `Gdiff`
  workspace, and single-VMM direct-final execution.
- Read variation: dynamic per-forward noise. A noisy `Gdiff` workspace may be
  reused as storage, but its sampled values are regenerated every forward.
- Release target: GitHub branch `v2.2` and annotated tag `v2.2.0`. Existing
  `v2` and `v2.1` branches remain unchanged.
- Excluded: paper data, experiment results, profiler artifacts, caches, and
  machine-specific launch scripts.
- Performance gate: Qwen3.5-9B mode1 at read variation 0.05 must be at least
  20% faster than the matched local v2.1 run and faster than matched v3 mode0
  on the same Pro6000 GPU.

## Frozen execution rule

- Source code checkpoint before release documentation: `043dbf5`.
- Mode1 Gdiff dispatch uses `BLOCK_R=64`, `BLOCK_L=32`, and `BLOCK_K=64`.
- Shapes with at most 64 input tiles use owner-compute. Wider input matrices
  use grouped execution with four input tiles per group.
- The speed endpoint uses a full-layer Gdiff workspace. Budgeted output windows
  remain available through the S1 planner but are not the default speed path.
- Workspace capacity grows geometrically and is reused across Linear calls to
  avoid repeated multi-gigabyte allocator churn.

## Correctness policy

- `read_variation=0.0`: retain the pair-index BF16 path. Require finite output
  and cosine similarity greater than 0.999 against the framework reference.
- `read_variation=0.05`: require independent `G+`/`G-` noise, finite full-model
  outputs, successful optimized-path counters, and zero required fallbacks.
- DAC voltage and restored conductance use BF16 by default. ADC quantization,
  tile scaling, and final accumulation remain FP32.

## Release evidence

Matched Pro6000, batch 1, sequence 128, warmup 1, repeat 10, full Linear
replacement including lm_head, BF16, and read variation 0.05:

- Qwen3.5-4B: three-process mean `209.44 ms`, process standard deviation
  `1.29 ms`, CUDA peak approximately `12.73 GB`.
- Qwen3.5-9B: three-process mean `225.91 ms`, process standard deviation
  `10.77 ms`, CUDA peak approximately `20.94 GB`.
- Matched local v2.1 mode1 baselines: `650.75 ms` for 4B and `896.86 ms` for
  9B, corresponding to `3.11x` and `3.97x` speedups.
- Matched v3 mode0 points: `408.81 ms` for 4B and `510.83 ms` for 9B. The v2.2
  mode1 speed endpoint is `1.95x` and `2.26x` faster, respectively.
- Mode0 Qwen3.5-0.8B smoke: full-model replacement succeeded with 140
  direct-final hits, zero direct-final fallback, and approximately `5.04 GB`
  CUDA peak.
