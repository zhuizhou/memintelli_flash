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
        3. Frees training-only data (quantized_data, sliced_data) from weight SlicedData
        4. Enables memory-efficient slice-by-slice dot product
        
        Call this after update_weight() and before inference.
        Memory savings: significant reduction in both static and dynamic GPU memory.
        Speed improvement: skips backward-related computation and uses optimized paths.
        """
        self.eval()  # set BatchNorm, Dropout to eval mode
        
        if not self.mem_enabled:
            return

        import gc
        for module in self.modules():
            if isinstance(module, ( LinearMem)):
                # Enable inference mode on the layer
                module.inference_mode = True
                # Mark weight SlicedData as inference (for engine routing)
                module.weight_sliced.inference = True
                # Free training-only data from weight
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None
        
        # Force garbage collection to reclaim freed memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def config(self):
        return self.model.config

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def _replace_linear_with_linearmem(module: nn.Module, mem_args: dict):
    """Recursively replace nn.Linear with LinearMem."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = LinearMem(
                engine=mem_args["engine"],
                in_features=child.in_features,
                out_features=child.out_features,
                input_slice=mem_args["input_slice"],
                weight_slice=mem_args["weight_slice"],
                bias=child.bias is not None,
                device=mem_args["device"],
                bw_e=mem_args["bw_e"],
                input_paral_size=mem_args["input_paral_size"],
                weight_paral_size=mem_args["weight_paral_size"],
                input_quant_gran=mem_args["input_quant_gran"],
                weight_quant_gran=mem_args["weight_quant_gran"],
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            _replace_linear_with_linearmem(child, mem_args)


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
    trust_remote_code: bool = True
) -> Qwen3MemWrapper:
    """Qwen3 model factory with optional linear-to-LinearMem conversion."""
    if AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is required for qwen3_zoo. "
            "Please install it via `pip install transformers`."
        ) from _TRANSFORMERS_IMPORT_ERROR

    if not pretrained:
        raise ValueError("qwen3_zoo currently supports pretrained=True only.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code
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
        _replace_linear_with_linearmem(model, mem_args)

    return Qwen3MemWrapper(model=model, mem_enabled=mem_enabled)
