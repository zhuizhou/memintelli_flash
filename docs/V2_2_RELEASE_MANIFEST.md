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
