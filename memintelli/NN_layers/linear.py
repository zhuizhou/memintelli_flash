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
        """Optimized inference forward: skips autograd, quantized_data, and uses memory-efficient dot.
        
        Optimizations:
          - Reuses a cached SlicedData object for input (avoids __init__ + _init_data overhead per call)
          - Casts output back to input dtype to enable Flash Attention and save memory
        """
        # Streaming mode: load G data from CPU to GPU before computation
        if self._streaming:
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
        input_sliced.slice_data_imp(self.engine, input)
        output = self.engine.MapReduceDot(input_sliced, self.weight_sliced)
        # Free input data immediately (keep the cache object structure)
        input_sliced.sliced_data = None
        input_sliced.max_data = None
        input_sliced.e_bias = None
        
        # Streaming mode: offload G data back to CPU after computation
        if self._streaming:
            self._offload_to_cpu()
        
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
        """Offload G/G_indices and max_data to pinned CPU RAM for fast async transfers.
        
        Uses pin_memory() for page-locked host memory, enabling 2-3x faster
        CPU↔GPU DMA transfers compared to regular pageable memory.
        Only the large tensors are moved; small tensors stay on GPU.
        """
        ws = self.weight_sliced
        if ws.G_indices is not None and ws.G_indices.device.type != 'cpu':
            ws.G_indices = ws.G_indices.to('cpu', non_blocking=True)
        if ws.G is not None and ws.G.device.type != 'cpu':
            ws.G = ws.G.to('cpu', non_blocking=True)
        if ws.max_data is not None and ws.max_data.device.type != 'cpu':
            ws.max_data = ws.max_data.to('cpu', non_blocking=True)
        if ws.e_bias is not None and ws.e_bias.device.type != 'cpu':
            ws.e_bias = ws.e_bias.to('cpu', non_blocking=True)
        # Synchronize before pinning (ensure non_blocking transfers complete)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Pin memory for faster future CPU→GPU transfers
        if ws.G_indices is not None and ws.G_indices.device.type == 'cpu' and not ws.G_indices.is_pinned():
            ws.G_indices = ws.G_indices.pin_memory()
        if ws.G is not None and ws.G.device.type == 'cpu' and not ws.G.is_pinned():
            ws.G = ws.G.pin_memory()
        if ws.max_data is not None and ws.max_data.device.type == 'cpu' and not ws.max_data.is_pinned():
            ws.max_data = ws.max_data.pin_memory()
        if ws.e_bias is not None and ws.e_bias.device.type == 'cpu' and not ws.e_bias.is_pinned():
            ws.e_bias = ws.e_bias.pin_memory()

    def _load_to_device(self, device):
        """Load G/G_indices and max_data from CPU back to GPU for computation.
        
        Uses non_blocking transfers + synchronize for efficiency.
        """
        ws = self.weight_sliced
        if ws.G_indices is not None and ws.G_indices.device != device:
            ws.G_indices = ws.G_indices.to(device, non_blocking=True)
        if ws.G is not None and ws.G.device != device:
            ws.G = ws.G.to(device, non_blocking=True)
        if ws.max_data is not None and ws.max_data.device != device:
            ws.max_data = ws.max_data.to(device, non_blocking=True)
        if ws.e_bias is not None and ws.e_bias.device != device:
            ws.e_bias = ws.e_bias.to(device, non_blocking=True)
        # Synchronize to ensure all data is ready before computation
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

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