# Multimode LLM Inference

This branch adds a multimode LinearMem path for LLM inference while keeping the
old Memintelli APIs source-compatible. Existing code can still import
`DPETensor` and `SlicedData`; by default they run Mode 0 through the updated
backend. New experiments can import `DPETensorMultiMode` directly.

## Modes

- `mode=0`: standard signed bit-sliced mapping. Input and weight tensors are
  quantized, bit-sliced, mapped to conductance, accumulated, ADC-quantized, and
  restored to the Linear output domain. This is the closest path to the classic
  Memintelli bit-sliced Linear simulation.
- `mode=1`: differential-pair weight projection. The weight is represented by
  positive/negative conductance branches and the input is driven by a signed DAC
  value. This mode is useful for faster LLM Linear simulation when the study
  focuses on differential-pair weight behavior.
- `mode=2`: differential-pair sliced mapping. Weight uses positive/negative
  conductance branches, and the input can use either signed drive or differential
  input drive (`mode2_input_mode="signed"` or `"differential"`). This mode keeps
  a more explicit signed-array readout structure.

## Execution Layout

The LLM example exposes three execution layouts:

- `speed`: keep prepared conductance/index tensors resident on GPU after
  preparation. This is the fastest layout when GPU memory is sufficient.
- `memory`: offload prepared conductance/index tensors to CPU and stream each
  Linear layer to GPU during forward. This reduces GPU residency and is intended
  for large models or smaller GPUs.
- `auto`: prepare all layers safely, then keep as many layers resident as the
  GPU memory budget allows and stream the rest.

By default, `examples/13_llama_inference.py` creates replacement `LinearMem`
parameters on CPU and prepares each layer sequentially. This avoids unnecessary
GPU peak memory during model conversion on smaller cards. On large-memory GPUs,
use `--linearmem-device engine --execution speed` to create replacement
`LinearMem` parameters directly on the engine device and keep prepared
conductance/index tensors resident on GPU.

## Triton Fast Path

On CUDA devices the LLM example uses `triton_gidx` by default. The fast path
keeps the same analog simulation order while reducing software overhead:

- direct-final mode-0 kernels fuse input-voltage quantization, conductance
  restoration, current accumulation, ADC quantization, slice weighting, and
  output writeback for common 2-D Linear inference shapes;
- shape-aware auto configuration (`--triton-auto-config`, enabled by default)
  selects conservative block and output-chunk sizes per Linear shape instead of
  applying one global tile choice to attention, MLP, and lm_head layers;
- mode-1 differential-pair inference reuses the same guarded scheduling path
  but keeps `--mode1-input-tile-group 1` by default because larger input-tile
  groups can be slower on Qwen/Llama-style MLP layers.

The auto configuration is a runtime scheduling guard only. It does not change
the configured array size, DAC/ADC precision, conductance levels, read/write
variation, or the selected mapping mode.

## Llama-3.1-8B WikiText Example

Install the optional LLM packages:

```bash
pip install transformers datasets accelerate sentencepiece
```

Run a quick configuration check without downloading the model:

```bash
python examples/13_llama_inference.py --dry-run --mode 0
python examples/13_llama_inference.py --dry-run --mode 1
```

Run Mode 0 on WikiText-2:

```bash
python examples/13_llama_inference.py \
  --model meta-llama/Llama-3.1-8B \
  --mode 0 \
  --execution speed \
  --linearmem-device engine \
  --array-size 64 \
  --input-slice 1,1,1,1,1 \
  --weight-slice 1,1,1,1,1 \
  --triton-auto-config
```

Run Mode 1 with differential-pair weights:

```bash
python examples/13_llama_inference.py \
  --model meta-llama/Llama-3.1-8B \
  --mode 1 \
  --execution speed \
  --array-size 64 \
  --g-level 16 \
  --rdac-bits 4 \
  --mode1-input-tile-group 1
```

For smaller GPUs, switch to streaming:

```bash
python examples/13_llama_inference.py \
  --model meta-llama/Llama-3.1-8B \
  --mode 0 \
  --execution memory \
  --array-size 64
```

`meta-llama/Llama-3.1-8B` is a gated Hugging Face model, so users need access to
the model repository and may need to pass `--hf-token`.
