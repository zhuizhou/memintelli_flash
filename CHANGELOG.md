# Changelog

## v2.2.0

- Unified S1 state-budget planning for mode0 and mode1 LLM Linear execution.
- Added reusable signed-DAC voltage preparation for mode1.
- Added independent noisy `G+`/`G-` restoration into a reusable BF16 `Gdiff`
  workspace.
- Added mode1 single-VMM direct-final execution with per-input-tile FP32 ADC,
  tile-local scale, and FP32 output accumulation.
- Added measured owner/grouped shape dispatch and full-layer speed workspaces.
- Added mode1 correctness, fallback, planner, and full-model benchmark contracts.
- Preserved the optimized mode0 direct-final runtime in the same release.
