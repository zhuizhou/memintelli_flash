# v2.1 Release Manifest

- Source baseline: DAC submission runtime from the audited working tree based on research commit `239af7b`.
- Primary validation target: mode1 differential-pair inference without input or weight slice expansion.
- Included: complete `memintelli` runtime package, mode0 DAC submission mechanisms, focused correctness tests, focused benchmark, and release documentation.
- Excluded: paper experiments and results, motivation/evaluation scripts, profiler artifacts, caches, and temporary diagnostics.
- Correctness policy: compare against the same-version mode1 PyTorch detailed path at `read_var=0.0`; BF16 uses reported error metrics and bounded tolerance.
- Variation policy: `read_var=0.05` uses the existing independent `G+` and `G-` noise path and is validated functionally rather than by cross-run element equality.

## Mode1 execution contract

- Mode1 represents each signed weight with a differential conductance pair and does not expand input or weight bit slices.
- Every input tile retains its own DAC quantization, differential VMM, ADC quantization, scale, and cross-tile FP32 reduction.
- `mode1_require_fastpath=True` turns an unsupported Triton shape into an error instead of silently falling back to the framework path.
- Row-strided `G+`, `G-`, signed-index, and scale views are consumed directly when their innermost dimension is contiguous, avoiding wide-output operand copies.
- At `read_var=0.0`, the optional signed `int8` differential index is exactly equivalent to the stored `G+` and `G-` indices. The default `wide` policy uses it only for output-chunked layers; `all` is available for the lowest index-storage footprint.
- At `read_var=0.0`, input-tile groups write independent FP32 partials and use a fixed reduction instead of output atomics. Repeated full-model forwards are therefore stable for fixed inputs and state.
- At nonzero read variation, signed-index compression is disabled and the independent noisy `G+` and `G-` branches are retained.

## Validation scope

- CUDA unit tests cover fast-path dispatch, differential semantics, strided operands, signed-index storage, pair/signed-index equivalence, and deterministic reduction.
- Full-model checks cover all Linear layers including `lm_head`, with Qwen 0.8B, 4B, and 9B at batch 1 and sequence length 128.
- Mode0 remains the synchronized DAC submission implementation; mode2 is kept behaviorally unchanged by the mode1-specific dispatch guards.
