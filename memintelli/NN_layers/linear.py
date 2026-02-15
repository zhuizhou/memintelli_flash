# -*- coding:utf-8 -*-
# @File  : linear.py
# @Author: Zhou
# @Date  : 2023/3/22
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from memintelli.pimpy.data_formats import SlicedData
from memintelli.NN_layers.functions import linear_mem_func
from matplotlib import pyplot as plt
from memintelli.pimpy import DPETensor
from memintelli.pimpy.utils import SNR

class LinearMem(nn.Module):
    _transfer_stream = None  # Class-level CUDA stream for async prefetch

    @classmethod
    def _get_transfer_stream(cls, device):
        """Get (or create) a dedicated CUDA stream for async data transfers."""
        if cls._transfer_stream is None:
            cls._transfer_stream = torch.cuda.Stream(device=device)
        return cls._transfer_stream

    def __init__(self, engine, in_features: int, out_features: int, input_slice:[list, tuple], weight_slice:[list, tuple],
                 bias: bool = True, device=None, dtype=torch.float32, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32), skip_initial_mapping=False):
        '''
        :param in_features: the input neuron number
        :param out_features: the output neuron number
        :param bias: use bias or not, default is True
        :param input_sli_mod: the slice method of the input matrix, default is (1, 1, 2, 4)
        :param weight_sli_mod: the slice method of the weight matrix, default is (1, 1, 2, 4)
        :param bw_e: the bit width of the input and weight, default is None, which means use the INT
        :param device: use cuda or cpu, default is None, which means use cpu
        :param dtype:
        :param skip_initial_mapping: if True, skip the initial weight→conductance mapping.
            Use when weights will be overwritten (e.g. loading pretrained) and update_weight() 
            will be called later. Saves significant GPU memory during model initialization.
        '''
        super(LinearMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight_slice_method = torch.tensor(weight_slice).to(device)
        self.input_slice_method = torch.tensor(input_slice).to(device)

        self.weight_sliced = SlicedData(self.weight_slice_method, device=device, bw_e=bw_e, is_weight=True, paral_size=weight_paral_size, quant_gran=weight_quant_gran)
        self.engine = engine
        if not skip_initial_mapping:
            self.weight_sliced.slice_data_imp(engine, self.weight.detach().t())
        self.input_paral_size = input_paral_size
        self.input_quant_gran = input_quant_gran
        self.inference_mode = False  # set True via prepare_for_inference()
        self._streaming = False  # CPU offloading mode for streaming inference
        self._input_sliced_cache = None  # reusable SlicedData for inference (avoids re-creation)
        # Use object.__setattr__ to avoid nn.Module registering these as submodules.
        # _next_streaming_layer will hold a reference to another LinearMem (nn.Module),
        # and PyTorch's __setattr__ would auto-register it as a child, creating circular
        # module trees that cause RecursionError in .to() / state_dict() etc.
        object.__setattr__(self, '_next_streaming_layer', None)  # prefetch chain link
        object.__setattr__(self, '_prefetch_event', None)  # CUDA event for prefetch sync
        object.__setattr__(self, '_pinned_buffers', {})  # persistent pinned CPU buffers

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self._forward_inference(input)
        input_sliced = SlicedData(self.input_slice_method, device=input.device, bw_e=self.weight_sliced.bw_e,is_weight=False, paral_size=self.input_paral_size, quant_gran=self.input_quant_gran)
        input_sliced.slice_data_imp(self.engine, input.detach())
        return linear_mem_func(self.engine, input, self.weight, input_sliced, self.weight_sliced, self.bias)

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """Optimized inference forward with async prefetch support.
        
        Execution flow for streaming layers:
          1. If previous layer started a prefetch → just wait for the event (near-zero cost)
             Otherwise (first layer) → synchronous load from pinned CPU
          2. Compute MapReduceDot
          3. Start async prefetch for next streaming layer (overlaps with norms/attention)
          4. Release GPU tensors (point references back to pinned buffers — NO D2H copy!)
        """
        # Streaming: ensure G data is on GPU
        if self._streaming:
            if self._prefetch_event is not None:
                # Async prefetch was started by the previous layer — wait on GPU stream only
                torch.cuda.current_stream(self.engine.device).wait_event(self._prefetch_event)
                self._prefetch_event = None
            else:
                # First streaming layer in this forward pass — synchronous load
                self._load_to_device(self.engine.device)
        
        # Reuse cached SlicedData object (avoids creating new one each call)
        if self._input_sliced_cache is None:
            self._input_sliced_cache = SlicedData(
                self.input_slice_method, device=input.device,
                bw_e=self.weight_sliced.bw_e, is_weight=False,
                paral_size=self.input_paral_size,
                quant_gran=self.input_quant_gran,
                inference=True)
        input_sliced = self._input_sliced_cache

        # Input-dimension chunking to bound peak memory during slice_data_imp().
        # This complements engine-side matmul chunking (which chunks along weight cols).
        chunk_budget = getattr(self.engine, 'inference_chunk_size', None) or 32 * 1024 * 1024
        in_features = max(1, input.shape[-1])
        expansion_factor = 8  # conservative estimate for slicing intermediates
        max_positions = max(1, chunk_budget // max(1, in_features * expansion_factor))
        total_positions = input.shape[-2]

        if max_positions >= total_positions:
            input_sliced.slice_data_imp(self.engine, input)
            output = self.engine.MapReduceDot(input_sliced, self.weight_sliced)
            # Free input data immediately (keep the cache object structure)
            input_sliced.sliced_data = None
            input_sliced.max_data = None
            input_sliced.e_bias = None
        else:
            out_chunks = []
            # Chunk along the "row/token" dimension (last-2 dim): [N, C] or [B, T, C]
            if input.dim() == 2:
                for start in range(0, total_positions, max_positions):
                    end = min(start + max_positions, total_positions)
                    x_chunk = input[start:end, :]
                    input_sliced.slice_data_imp(self.engine, x_chunk)
                    out_chunks.append(self.engine.MapReduceDot(input_sliced, self.weight_sliced))
                    input_sliced.sliced_data = None
                    input_sliced.max_data = None
                    input_sliced.e_bias = None
                output = torch.cat(out_chunks, dim=0)
            else:
                for start in range(0, total_positions, max_positions):
                    end = min(start + max_positions, total_positions)
                    x_chunk = input[:, start:end, :]
                    input_sliced.slice_data_imp(self.engine, x_chunk)
                    out_chunks.append(self.engine.MapReduceDot(input_sliced, self.weight_sliced))
                    input_sliced.sliced_data = None
                    input_sliced.max_data = None
                    input_sliced.e_bias = None
                output = torch.cat(out_chunks, dim=1)
            del out_chunks
        
        # Async prefetch next streaming layer (overlaps with attention/norms between layers)
        if self._next_streaming_layer is not None and self._next_streaming_layer._streaming:
            self._next_streaming_layer._async_prefetch(self.engine.device)
        
        # Release GPU tensors — point references back to pinned CPU buffers (NO D2H copy!)
        if self._streaming:
            self._release_gpu_tensors()
        
        # Move output back to input device if engine is on a different device
        if output.device != input.device:
            output = output.to(input.device)
        
        # CRITICAL: Cast output back to input dtype (e.g., float32 → bfloat16).
        if output.dtype != input.dtype:
            output = output.to(input.dtype)
        
        if self.bias is not None:
            output = output + self.bias
        return output

    def _offload_to_cpu(self):
        """Initial offload: move G data from GPU to pinned CPU memory.
        
        Called ONCE during update_weight_and_prepare(streaming=True).
        Creates persistent pinned buffers in self._pinned_buffers for
        fast async prefetch during inference. These buffers are never freed
        and are reused on every forward pass.
        """
        ws = self.weight_sliced
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure G computation is complete
        
        for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
            tensor = getattr(ws, attr, None)
            if tensor is not None:
                if tensor.device.type != 'cpu':
                    cpu_t = tensor.cpu()
                else:
                    cpu_t = tensor
                if not cpu_t.is_pinned():
                    cpu_t = cpu_t.pin_memory()
                self._pinned_buffers[attr] = cpu_t
                setattr(ws, attr, cpu_t)

    def _load_to_device(self, device):
        """Synchronous load: copy pinned CPU → GPU, blocks until complete.
        
        Used for the first streaming layer in a forward pass (no prefetch available).
        Subsequent layers should use _async_prefetch for overlap.
        """
        ws = self.weight_sliced
        for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
            pinned = self._pinned_buffers.get(attr)
            if pinned is not None:
                setattr(ws, attr, pinned.to(device, non_blocking=True))
        torch.cuda.synchronize(device)

    def _async_prefetch(self, device):
        """Start loading G data to GPU on a dedicated transfer stream (non-blocking).
        
        This is called by the PREVIOUS layer in the prefetch chain, so the H2D
        transfer overlaps with whatever computation happens between the two layers
        (attention, norms, residuals, etc.).
        
        OOM-safe: if GPU doesn't have enough free memory for the prefetch allocation,
        we reset back to pinned CPU buffers and clear the event. The next layer's
        forward() will then fall back to synchronous loading (which works because
        the current layer's GPU tensors will have been released by then).
        """
        try:
            stream = LinearMem._get_transfer_stream(device)
            ws = self.weight_sliced
            with torch.cuda.stream(stream):
                for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
                    pinned = self._pinned_buffers.get(attr)
                    if pinned is not None:
                        setattr(ws, attr, pinned.to(device, non_blocking=True))
            object.__setattr__(self, '_prefetch_event', stream.record_event())
        except torch.cuda.OutOfMemoryError:
            # Prefetch failed — reset to pinned CPU, next forward will sync-load
            self._release_gpu_tensors()
            object.__setattr__(self, '_prefetch_event', None)
            torch.cuda.empty_cache()

    def _release_gpu_tensors(self):
        """Release GPU tensors by pointing references back to pinned CPU buffers.
        
        KEY INSIGHT: G data never changes during inference, so we DON'T need
        D2H copies. We just set the SlicedData attributes back to the persistent
        pinned CPU tensors. The old GPU tensors become unreferenced and are freed
        by PyTorch's caching allocator.
        
        This eliminates 50% of the PCIe transfer overhead compared to the old
        approach (which did GPU→CPU copy + re-pin on every forward pass).
        """
        ws = self.weight_sliced
        for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
            pinned = self._pinned_buffers.get(attr)
            if pinned is not None:
                setattr(ws, attr, pinned)

    def update_weight(self):
        self.weight_sliced.slice_data_imp(self.engine, self.weight.detach().t().to(self.engine.device))

def _test(mode=0):
    if mode == 0:
        print("-----------------cuda-----------------")
        engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.0,          # Write variation
        rate_stuck_HGS=0.001,          # Rate of stuck at HGS
        rate_stuck_LGS=0.000,          # Rate of stuck at LGS
        read_variation={0:0.05, 1:0.05, 2:0.05, 3:0.05},           # Read variation
        vnoise=0.05,                   # Random Gaussian noise of voltage
        rdac=2**2,                      # Number of DAC resolution 
        g_level=2**2,                   # Number of conductance levels
        radc=2**12
        )
        xblk = [1, 1, 2, 2]
        mblk = [1, 1, 2, 2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = torch.randn(500, 100, requires_grad=True).to(device)
        layer = LinearMem(engine, 100, 300, bias=False, input_slice=xblk, weight_slice=mblk, device=device, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32))
        output = layer(input)
        #output.backward(torch.ones_like(output, dtype=torch.float))
        weight = layer.weight.data
        #weight.requires_grad = True
        output_ideal = F.linear(input.to(device), weight)
        output_ideal.backward(torch.ones_like(output_ideal, dtype=torch.float))

        output = output.cpu().detach().numpy()
        output_ideal = output_ideal.cpu().detach().numpy()
        print(SNR(output_ideal, output))


if __name__ == '__main__':
    _test(0)