# Mode1 Differential Acceleration v2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the DAC submission runtime as the release baseline, verify and further accelerate mode1 differential-pair execution, measure end-to-end speedup, and publish a clean `v2.1` branch.

**Architecture:** Keep the existing mode1 PyTorch tile loop as the semantic reference and the existing pair-index Triton direct-final kernel as the first optimized stage. Add a zero-read-variation specialization that precomputes signed DAC voltage once and consumes one signed differential conductance index instead of loading separate `G+` and `G-` indices. Add an atomic-free reduction only when profiling proves atomics materially limit the remaining kernel.

**Tech Stack:** Python 3.10+, PyTorch CUDA, Triton, pytest, Hugging Face Transformers, Nsight Systems, Git.

## Global Constraints

- Primary mode is `mode1`; mode2 receives regression coverage only.
- Correctness is established first at `read_var=0.0`; BF16 rounding error is acceptable.
- ADC, tile scaling, and final accumulation remain FP32.
- Full-model benchmark uses batch=1, seq=128, warmup=1, repeat=10, and replaces all Linear layers including lm_head.
- Performance runs use an idle Pro6000 GPU and three independent processes for final points.
- The GitHub release excludes paper data, experiment result directories, S1 planner code, Nsight artifacts, caches, and temporary diagnostics.
- The release updates `origin/v2.1` with a normal commit unless remote history requires explicit reconciliation.

---

### Task 1: Create the isolated release workspace from the DAC runtime

**Files:**
- Create worktree: `D:/OneDrive/odysseia/work/10.Memintelli/memintelli_flash_v2_1_dac_release`
- Modify: runtime package files under `memintelli/`
- Create: `docs/V2_1_RELEASE_MANIFEST.md`

**Interfaces:**
- Consumes: DAC runtime from `memintelli_flash_v3_research_20260710`.
- Produces: a clean `v2.1` worktree containing only releasable runtime files.

- [ ] **Step 1: Fetch and create an isolated branch/worktree**

```powershell
git -C memintelli_flash_v2_1_release fetch origin
git -C memintelli_flash_v2_1_release worktree add ..\memintelli_flash_v2_1_dac_release -b release/v2.1-dac origin/v2.1
```

Expected: new worktree on `release/v2.1-dac`, with a clean status.

- [ ] **Step 2: Synchronize the runtime whitelist**

Copy only these DAC runtime paths into the release worktree:

```text
memintelli/NN_layers/linear.py
memintelli/pimpy/__init__.py
memintelli/pimpy/data_formats_multimode.py
memintelli/pimpy/memmat_tensor_multimode.py
memintelli/pimpy/triton_fast_accumulate.py
pyproject.toml
requirements.txt
```

Use `Copy-Item -LiteralPath` for the mechanical copy. Do not copy `docs/paper_*`, `experiments/`, result JSON, shell launchers, or caches.

- [ ] **Step 3: Write the release manifest**

Create `docs/V2_1_RELEASE_MANIFEST.md` with:

```markdown
# v2.1 Release Manifest

- Source baseline: DAC submission runtime, research commit `686fcfc` plus the audited runtime working-tree diff recorded in this release commit.
- Primary new validation target: mode1 differential-pair inference.
- Included: multimode runtime, Triton kernels, focused correctness tests, focused benchmark.
- Excluded: paper experiments/results, S1 planner, motivation/evaluation scripts, profiling artifacts.
- Correctness policy: read_var=0.0 reference comparison; BF16 allclose with reported error metrics.
```

- [ ] **Step 4: Verify the imported runtime compiles**

Run:

```powershell
python -m compileall memintelli
```

Expected: exit code 0.

- [ ] **Step 5: Commit the isolated DAC runtime baseline**

```powershell
git add memintelli pyproject.toml requirements.txt docs/V2_1_RELEASE_MANIFEST.md
git commit -m "chore: align v2.1 with DAC runtime"
```

---

### Task 2: Freeze the mode1 semantic reference and fast-path contract

**Files:**
- Create: `tests/test_mode1_execution_contract.py`
- Modify: `memintelli/pimpy/memmat_tensor_multimode.py`

**Interfaces:**
- Consumes: `DPETensorMultiMode`, `SlicedDataMultiMode`.
- Produces: explicit reference and optimized mode1 configurations plus mandatory fast-path counters.

- [ ] **Step 1: Write the failing contract test**

The test constructs a small CUDA mode1 engine twice. Reference uses `fast_inference_backend="torch"`; optimized uses `fast_inference_backend="triton_gidx"`, `triton_mode1_gidx_direct_final=True`. Assert the optimized run increments `mode1_gidx_direct_final_success_count` and has zero fallback count.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mode1_direct_final_hits_without_fallback():
    reference, optimized, x, weight = build_mode1_case(read_var=0.0)
    expected = run_case(reference, x, weight)
    actual = run_case(optimized, x, weight)
    counters = optimized.get_fastpath_counters()
    assert counters["mode1_gidx_direct_final_success_count"] > 0
    assert counters["mode1_gidx_direct_final_fallback_count"] == 0
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 2: Run the test and verify the current contract failure**

Run:

```bash
pytest -q tests/test_mode1_execution_contract.py::test_mode1_direct_final_hits_without_fallback
```

Expected: FAIL if configuration silently falls back or required test helpers do not yet exist.

- [ ] **Step 3: Add a strict fast-path requirement flag**

Add constructor argument and state:

```python
mode1_require_fastpath=False
self.mode1_require_fastpath = bool(mode1_require_fastpath)
```

At the end of `_dot_mode1_inference`, raise `RuntimeError("Mode1 fast path was required but no Triton path succeeded")` when the flag is true and all mode1 direct-final attempts failed. Reference mode leaves the flag false.

- [ ] **Step 4: Run the contract test**

Run the test from Step 2. Expected: PASS with a nonzero success counter.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mode1_execution_contract.py memintelli/pimpy/memmat_tensor_multimode.py
git commit -m "test: freeze mode1 execution contract"
```

---

### Task 3: Establish read_var=0 mode1 correctness across LLM shapes

**Files:**
- Create: `tests/test_mode1_differential_correctness.py`
- Create: `examples/21_mode1_differential_benchmark.py`

**Interfaces:**
- Produces: `compare_mode1_outputs(reference, optimized) -> dict[str, float | bool]` and a JSON benchmark record.

- [ ] **Step 1: Write parametrized failing correctness tests**

Use representative reduced test shapes that preserve tile boundaries:

```python
@pytest.mark.parametrize("shape", [
    (128, 512, 512),
    (128, 512, 1536),
    (128, 1536, 512),
    (128, 512, 4097),
])
def test_mode1_bf16_matches_reference_at_zero_variation(shape):
    metrics = run_mode1_comparison(shape, read_var=0.0, dtype=torch.bfloat16)
    assert metrics["finite"]
    assert metrics["allclose_rtol1e_2_atol1e_2"]
    assert metrics["cosine_similarity"] > 0.999
```

- [ ] **Step 2: Verify the tests fail before the harness exists**

Run:

```bash
pytest -q tests/test_mode1_differential_correctness.py
```

Expected: FAIL because `run_mode1_comparison` is missing.

- [ ] **Step 3: Implement the focused comparison harness**

The benchmark helper must report:

```python
{
    "max_abs": float,
    "mean_abs": float,
    "max_rel": float,
    "cosine_similarity": float,
    "torch_equal": bool,
    "allclose_rtol1e_2_atol1e_2": bool,
    "finite": bool,
    "fastpath_counters": dict,
}
```

Both paths reuse identical input, quantized weight, mode1 tile scales, DAC/ADC settings, and `read_var=0.0`.

- [ ] **Step 4: Run correctness tests on GPU**

Run the command from Step 2. Expected: all parametrized cases PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_mode1_differential_correctness.py examples/21_mode1_differential_benchmark.py
git commit -m "test: validate mode1 differential semantics"
```

---

### Task 4: Profile the existing mode1 direct-final implementation

**Files:**
- Modify: `examples/21_mode1_differential_benchmark.py`
- Create locally only: `artifacts/mode1_profile/` (never commit)

**Interfaces:**
- Produces: per-shape latency, kernel count, median kernel duration, direct-final hit counters, and Nsight trace.

- [ ] **Step 1: Add profile shapes and paired modes**

Support CLI values:

```text
--path reference|pair-direct
--shape qkv|gate-up|down|lm-head
--read-var 0.0
--warmup 5
--repeat 20
--json-out PATH
```

- [ ] **Step 2: Run paired microbenchmarks on an idle Pro6000**

Run reference and pair-direct for each shape. Expected: JSON includes mean/min latency and fast-path counters.

- [ ] **Step 3: Capture Nsight Systems for pair-direct**

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o artifacts/mode1_profile/lm_head_pair_direct \
  /home/zzw/.venvs/sglang-dflash/bin/python examples/21_mode1_differential_benchmark.py \
  --path pair-direct --shape lm-head --read-var 0.0 --warmup 2 --repeat 5
```

- [ ] **Step 4: Apply the optimization decision rule**

Use the following deterministic rule:

```text
Always implement the zero-variation single differential-index path.
Implement activation-voltage reuse in the same path because it removes repeated DAC work without changing ADC order.
Implement atomic-free reduction only if atomic kernels or atomic serialization account for at least 15% of pair-direct GPU time, or if input_tile_group scaling stops improving before group=all.
```

- [ ] **Step 5: Record the diagnostic result in the benchmark JSON, not in release docs**

Do not commit Nsight files.

---

### Task 5: Add the zero-variation single-difference mode1 kernel

**Files:**
- Modify: `memintelli/pimpy/data_formats_multimode.py`
- Modify: `memintelli/pimpy/memmat_tensor_multimode.py`
- Modify: `memintelli/pimpy/triton_fast_accumulate.py`
- Modify: `tests/test_mode1_execution_contract.py`
- Modify: `tests/test_mode1_differential_correctness.py`

**Interfaces:**
- Produces: `triton_mode1_gdiff_direct_final(vin, gdiff_idx, w_scale, ...) -> torch.Tensor`.
- Adds engine option: `triton_mode1_gdiff_direct_final=True`.

- [ ] **Step 1: Write the failing G-difference storage test**

```python
def test_mode1_zero_variation_builds_signed_difference_indices():
    mat = build_mode1_weight(read_var=0.0)
    gp, gn = mat.G_indices
    assert mat.mode1_gdiff_indices is not None
    torch.testing.assert_close(
        mat.mode1_gdiff_indices.to(torch.int16),
        gp.to(torch.int16) - gn.to(torch.int16),
        rtol=0,
        atol=0,
    )
```

- [ ] **Step 2: Run the storage test and verify RED**

Expected: FAIL because `mode1_gdiff_indices` does not exist.

- [ ] **Step 3: Add compact signed difference storage**

Add `mode1_gdiff_indices` to `SlicedDataMultiMode`. Build it only when mode1 conductance levels fit signed int8; otherwise use int16. Do not discard `G_indices`, because read variation requires independent branch noise.

- [ ] **Step 4: Write the failing direct-final kernel test**

Compare the new kernel with the existing pair-index kernel at `read_var=0.0` for tail and non-tail dimensions. Require BF16 allclose and assert `mode1_gdiff_direct_final_success_count > 0`.

- [ ] **Step 5: Verify the kernel test fails**

Expected: FAIL because `triton_mode1_gdiff_direct_final` is missing.

- [ ] **Step 6: Implement precomputed signed DAC voltage**

Add a helper that computes once per Linear:

```python
vin = vread * torch.round(x_2d.float() / x_max * (rdac - 1)) / (rdac - 1)
vin = vin.to(compute_dtype)
```

The helper is used only by the zero-variation specialization. It must preserve the reference signed DAC formula.

- [ ] **Step 7: Implement `triton_mode1_gdiff_direct_final`**

Clone the proven grouped mode1 scheduling structure, but load one `gdiff_idx` tensor and compute:

```python
w = gdiff_idx * q_g
cur += tl.dot(vin, w)
q = round_even(cur / adc_ref_tile * (radc - 1)) / (radc - 1)
tile_total += q * adc_ref_tile * tile_scale * x_max / (vread * q_g * (g_level - 1))
```

Keep ADC per input tile and FP32 tile accumulation unchanged.

- [ ] **Step 8: Dispatch only when semantic guards hold**

Use the new path only when:

```text
mode == 1
read_var == 0.0
vnoise == 0
mode1_adc_per_tile is true
mode1_gdiff_indices exists
backend is triton or triton_gidx
```

Otherwise retain the existing pair-index direct-final path.

- [ ] **Step 9: Run all mode1 correctness tests**

```bash
pytest -q tests/test_mode1_execution_contract.py tests/test_mode1_differential_correctness.py
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add memintelli/pimpy/data_formats_multimode.py memintelli/pimpy/memmat_tensor_multimode.py memintelli/pimpy/triton_fast_accumulate.py tests
git commit -m "perf: add mode1 zero-variation differential kernel"
```

---

### Task 6: Add atomic-free reduction only when the profile gate fires

**Files:**
- Modify: `memintelli/pimpy/triton_fast_accumulate.py`
- Modify: `memintelli/pimpy/memmat_tensor_multimode.py`
- Modify: `tests/test_mode1_differential_correctness.py`

**Interfaces:**
- Adds option: `triton_mode1_atomic_free=False` by default until benchmarked.

- [ ] **Step 1: Skip this task when Task 4 reports less than 15% atomic cost**

Record `atomic_free_not_implemented_reason="profile_below_15_percent"` in the local benchmark JSON. No production option is added in that case.

- [ ] **Step 2: If the gate fires, write a failing grouped-reduction test**

The test compares atomic and partial-reduction outputs at groups 2, 4, and 8, requiring BF16 allclose and finite output.

- [ ] **Step 3: Implement bounded partial reduction**

Write per-group FP32 partial output shaped `[groups, tokens, out_chunk]`, then reduce groups in ascending order into final output. Cap partial bytes at 512 MiB; when the cap is exceeded, reduce the output chunk width or fall back to atomic grouping.

- [ ] **Step 4: Benchmark atomic versus partial reduction**

Retain the mechanism only if full-model mean latency improves by at least 5% without increasing CUDA peak by more than 1 GiB at Qwen4B.

- [ ] **Step 5: Commit only the retained implementation**

```bash
git add memintelli/pimpy/triton_fast_accumulate.py memintelli/pimpy/memmat_tensor_multimode.py tests/test_mode1_differential_correctness.py
git commit -m "perf: reduce mode1 grouped accumulation overhead"
```

---

### Task 7: Run full-model correctness and performance validation

**Files:**
- Modify: `examples/21_mode1_differential_benchmark.py`
- Create locally only: `artifacts/mode1_results/` (never commit)

**Interfaces:**
- Produces final benchmark JSON for reference, pair-direct, and gdiff-direct.

- [ ] **Step 1: Add full-model mode to the focused benchmark**

Support Qwen 0.8B, 4B, and 9B model paths; replace every `torch.nn.Linear`, including lm_head. Reject a run when any required mode1 fast-path fallback count is nonzero.

- [ ] **Step 2: Run read_var=0 correctness**

Compare final logits for reference and optimized BF16 paths. Report all metrics defined in Task 3.

- [ ] **Step 3: Run three-process performance points**

For each available model, run:

```text
reference: backend=torch
pair-direct: existing G+/G- Triton direct-final
gdiff-direct: new zero-variation specialization
```

Each process uses warmup=1 and repeat=10. Report process-level means and aggregate mean/std.

- [ ] **Step 4: Run read_var=0.05 compatibility**

Confirm the dispatcher uses the existing pair-index path, outputs are finite, and no required-path fallback occurs. Do not require cross-run equality.

- [ ] **Step 5: Run mode0 and mode2 smoke regressions**

Run one representative Linear and one Qwen0.8B prefill for mode0; run representative mode2 Linear correctness. Expected: no regression or import failure.

- [ ] **Step 6: Update `docs/V2_1_RELEASE_MANIFEST.md` with supported paths only**

Document mode1 reference, pair-direct, gdiff-direct guards, and read-var behavior. Do not include large result tables or claims unsupported by the final JSON.

- [ ] **Step 7: Commit benchmark and documentation**

```bash
git add examples/21_mode1_differential_benchmark.py docs/V2_1_RELEASE_MANIFEST.md
git commit -m "docs: publish mode1 validation workflow"
```

---

### Task 8: Audit and publish the clean v2.1 branch

**Files:**
- Audit the complete release worktree.

**Interfaces:**
- Produces: clean `origin/v2.1` and a fresh-clone verification result.

- [ ] **Step 1: Run the release verification suite**

```bash
python -m compileall memintelli examples/21_mode1_differential_benchmark.py
pytest -q tests/test_mode1_execution_contract.py tests/test_mode1_differential_correctness.py
```

Expected: exit code 0.

- [ ] **Step 2: Audit tracked content**

```powershell
git status --short
git diff origin/v2.1...HEAD --check
git diff origin/v2.1...HEAD --stat
git ls-files | Select-String -Pattern 'paper_2026|artifacts|\.json$|\.nsys-rep$|__pycache__|\.pyc$'
```

Expected: clean status; no prohibited tracked paths.

- [ ] **Step 3: Scan for secrets and machine-specific paths**

```powershell
rg -n "BEGIN (RSA|OPENSSH) PRIVATE KEY|password\s*=|/home/zzw|D:\\OneDrive" memintelli tests examples docs
```

Expected: no secrets; benchmark defaults use CLI arguments rather than private absolute paths.

- [ ] **Step 4: Update the remote branch**

```bash
git fetch origin
git rebase origin/v2.1
git push origin HEAD:v2.1
```

- [ ] **Step 5: Verify from a fresh clone**

Clone `origin/v2.1` into `_verify_memintelli_flash_v2_1_mode1`, run compileall and the smallest CUDA correctness test, and verify the reported commit matches the pushed commit.

