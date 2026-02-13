# -*- coding:utf-8 -*-
# @File  : Qwen3.py
# @Author: Zhou
# @Date  : 2026/2/13

from typing import Optional, Union, Any

import torch
import torch.nn as nn

from memintelli.NN_layers import LinearMem

try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class Qwen3MemWrapper(nn.Module):
    """Qwen3 wrapper with optional memristive linear layers."""

    def __init__(self, model: nn.Module, mem_enabled: bool = False):
        super().__init__()
        self.model = model
        self.mem_enabled = mem_enabled

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_weight(self):
        if not self.mem_enabled:
            return
        for module in self.modules():
            if isinstance(module, LinearMem):
                module.update_weight()
    
    def prepare_for_inference(self) -> None:
        """Prepare the model for optimized inference.
        
        This method:
        1. Sets eval mode (BatchNorm, Dropout, etc.)
        2. Enables inference_mode on all memristive layers (bypasses autograd)
        3. Compresses G to uint8 indices if possible (~4x memory savings)
        4. Frees training-only data (quantized_data, sliced_data) from weight SlicedData
        
        Call this after update_weight() and before inference.
        For lower peak memory during initialization, use update_weight_and_prepare() instead.
        """
        self.eval()  # set BatchNorm, Dropout to eval mode
        
        if not self.mem_enabled:
            return

        import gc
        for module in self.modules():
            if isinstance(module, LinearMem):
                # Enable inference mode on the layer
                module.inference_mode = True
                # Mark weight SlicedData as inference (for engine routing)
                module.weight_sliced.inference = True
                # Compress G to uint8 indices (~4x memory savings)
                # Valid when write_variation=0 (G values are exactly on conductance levels)
                engine = module.engine
                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)
                # Free training-only data from weight
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None
        
        # Force garbage collection to reclaim freed memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_weight_and_prepare(self, streaming: bool = False, free_weights: bool = True) -> None:
        """Combined update + prepare that minimizes peak GPU memory.
        
        Processes each LinearMem layer sequentially:
          1. Move weight from CPU to engine device (GPU) temporarily
          2. Compute G from weights → compress to uint8
          3. Free the weight immediately (back to empty CPU tensor)
          4. Move bias and internal tensors to engine device
          5. Optionally offload G to CPU (streaming mode)
          6. Garbage collect before next layer
        
        Because LinearMem weights are created on CPU during replacement,
        only ONE layer's weight + G exists on GPU at any time. This enables
        running models far larger than GPU memory:
        
        Peak GPU memory = embedding + norms + lm_head + ONE_layer_weight + ONE_layer_G
        
        For Qwen3-8B streaming: peak ~2.5GB on 24GB GPU!
        For Qwen3-4B streaming: peak ~1.5GB
        
        Args:
            streaming: If True, offload each layer's G data to CPU after processing.
            free_weights: If True (default), free nn.Parameter weight data after G.
        """
        self.eval()
        
        if not self.mem_enabled:
            return

        import gc
        layer_count = 0
        for module in self.modules():
            if isinstance(module, LinearMem):
                layer_count += 1
                engine = module.engine
                engine_device = engine.device
                
                # Set inference BEFORE update
                module.weight_sliced.inference = True
                module.inference_mode = True
                
                # update_weight() moves weight to engine device internally via
                # self.weight.detach().t().to(self.engine.device)
                # This works whether weight is on CPU or GPU.
                module.update_weight()
                
                # Compress G to uint8 indices if possible
                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)
                
                # Free original weight parameters → empty CPU tensor
                if free_weights:
                    module.weight.data = torch.empty(0, device='cpu', dtype=module.weight.dtype)
                
                # Free training-only data
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None
                
                # Move bias to engine device (if it was created on CPU)
                if module.bias is not None and module.bias.device != engine_device:
                    module.bias.data = module.bias.data.to(engine_device)
                
                # Move slice method tensors to engine device
                if module.input_slice_method.device != engine_device:
                    module.input_slice_method = module.input_slice_method.to(engine_device)
                if module.weight_slice_method.device != engine_device:
                    module.weight_slice_method = module.weight_slice_method.to(engine_device)
                
                # Streaming: offload G data to CPU after processing
                if streaming:
                    module._streaming = True
                    module._offload_to_cpu()
                
                # Per-layer GC to keep peak memory low
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"[Qwen3] Processed {layer_count} LinearMem layers"
              f" ({'streaming' if streaming else 'GPU-resident'},"
              f" weights {'freed' if free_weights else 'kept'})")

    def enable_streaming(self) -> None:
        """Enable streaming mode: offload all layer G data to CPU, load on-demand.
        
        Call this after update_weight_and_prepare() to further reduce GPU memory.
        During inference, each LinearMem layer will:
          1. Load its G_indices/G + max_data from CPU → GPU
          2. Perform the memristive dot product
          3. Offload G_indices/G + max_data back to CPU
        
        This means only ONE layer's G data is on GPU at any time.
        
        Memory savings example (Qwen3-0.6B, 4 slices, write_variation=0):
          - Without streaming: ~2GB GPU for all compressed G_indices
          - With streaming: ~100MB GPU (largest single layer's G_indices)
          
        Speed tradeoff: ~5-15ms overhead per layer per forward pass (PCIe transfer).
        """
        if not self.mem_enabled:
            return
        for module in self.modules():
            if isinstance(module, LinearMem):
                module._streaming = True
                module._offload_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def config(self):
        return self.model.config

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def _replace_linear_with_linearmem(module: nn.Module, mem_args: dict,
                                    skip_modules: set = None, prefix: str = ""):
    """Recursively replace nn.Linear with LinearMem, skipping specified modules.
    
    MEMORY OPTIMIZATION: Creates LinearMem on CPU with the original model dtype
    (e.g. bfloat16) instead of GPU float32. This prevents GPU memory doubling
    during replacement. For 8B models: avoids 16GB(old) + 32GB(new) = 48GB peak.
    
    The actual weight→conductance mapping happens later in update_weight_and_prepare(),
    which processes ONE layer at a time to minimize peak GPU memory.
    """
    import gc
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if skip_modules and full_name in skip_modules:
                continue  # keep as regular nn.Linear
            # Create on CPU with original dtype to avoid GPU memory explosion.
            # For 8B bfloat16: this prevents trying to hold 48GB on a 24GB GPU.
            new_layer = LinearMem(
                engine=mem_args["engine"],
                in_features=child.in_features,
                out_features=child.out_features,
                input_slice=mem_args["input_slice"],
                weight_slice=mem_args["weight_slice"],
                bias=child.bias is not None,
                device='cpu',  # CPU → avoids GPU memory doubling
                dtype=child.weight.dtype,  # match original (bfloat16, not float32)
                bw_e=mem_args["bw_e"],
                input_paral_size=mem_args["input_paral_size"],
                weight_paral_size=mem_args["weight_paral_size"],
                input_quant_gran=mem_args["input_quant_gran"],
                weight_quant_gran=mem_args["weight_quant_gran"],
                skip_initial_mapping=True,
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias.data)
            setattr(module, name, new_layer)
            del child  # explicitly free old layer (releases GPU weight memory)
            gc.collect()
        else:
            _replace_linear_with_linearmem(child, mem_args, skip_modules, full_name)


def qwen3_zoo(
    model_name: str = "Qwen/Qwen3-0.6B",
    pretrained: bool = True,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bw_e: Optional[Any] = None,
    input_paral_size: Optional[Union[torch.Tensor, list]] = (1, 32),
    weight_paral_size: Optional[Union[torch.Tensor, list]] = (32, 32),
    input_quant_gran: Optional[Union[torch.Tensor, list]] = (1, 64),
    weight_quant_gran: Optional[Union[torch.Tensor, list]] = (64, 64),
    torch_dtype: Optional[torch.dtype] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    skip_embedding_and_head: bool = True,
) -> Qwen3MemWrapper:
    """Qwen3 model factory with optional linear-to-LinearMem conversion.
    
    Args:
        skip_embedding_and_head: If True (default), keep embedding and lm_head as
            standard layers (no RRAM simulation). Highly recommended because:
              - lm_head is the LARGEST linear layer (e.g. 2560→151936 for Qwen3-4B)
              - Its G_indices alone occupy ~1.5GB, more than all other layers combined
              - RRAM simulation for lm_head takes ~50% of total inference time
              - Skipping it roughly HALVES total inference time and saves huge memory
              - lm_head is a simple projection; RRAM non-idealities there rarely matter
              - Embedding (nn.Embedding) is never replaced regardless of this flag
            Set to False to simulate ALL linear layers including lm_head.
    """
    if AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is required for qwen3_zoo. "
            "Please install it via `pip install transformers`."
        ) from _TRANSFORMERS_IMPORT_ERROR

    if not pretrained:
        raise ValueError("qwen3_zoo currently supports pretrained=True only.")

    # Build skip set from the bool flag
    skip_modules = {"lm_head"} if skip_embedding_and_head else set()

    _dtype = dtype if dtype is not None else torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_dtype,
        trust_remote_code=trust_remote_code,
    )

    if mem_enabled:
        if engine is None:
            raise ValueError("`engine` must be provided when mem_enabled=True.")
        mem_args = {
            "engine": engine,
            "input_slice": input_slice,
            "weight_slice": weight_slice,
            "device": device,
            "bw_e": bw_e,
            "input_paral_size": input_paral_size,
            "weight_paral_size": weight_paral_size,
            "input_quant_gran": input_quant_gran,
            "weight_quant_gran": weight_quant_gran
        }
        _replace_linear_with_linearmem(model, mem_args, skip_modules=skip_modules)
        if skip_modules:
            print(f"[Qwen3] Skipped RRAM simulation for: {skip_modules} (using standard nn.Linear)")

    return Qwen3MemWrapper(model=model, mem_enabled=mem_enabled)
