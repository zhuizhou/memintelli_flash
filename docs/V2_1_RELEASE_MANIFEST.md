# v2.1 Release Manifest

- Source baseline: DAC submission runtime from the audited working tree based on research commit `239af7b`.
- Primary validation target: mode1 differential-pair inference without input or weight slice expansion.
- Included: complete `memintelli` runtime package, mode0 DAC submission mechanisms, focused correctness tests, focused benchmark, and release documentation.
- Excluded: paper experiments and results, motivation/evaluation scripts, profiler artifacts, caches, and temporary diagnostics.
- Correctness policy: compare against the same-version mode1 PyTorch detailed path at `read_var=0.0`; BF16 uses reported error metrics and bounded tolerance.
- Variation policy: `read_var=0.05` uses the existing independent `G+` and `G-` noise path and is validated functionally rather than by cross-run element equality.
