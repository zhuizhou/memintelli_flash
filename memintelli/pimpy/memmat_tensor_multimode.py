import secrets
import time

import torch
from memintelli.pimpy.utils import dot_high_dim

try:
    from .data_formats_multimode import SlicedDataMultiMode
except ImportError:
    from data_formats_multimode import SlicedDataMultiMode

class DPETensorMultiMode(object):
    """Memory-efficient multi-mode dot product engine."""

    def __init__(
        self,
        HGS=1e-5, LGS=1e-7, g_level=16,
        write_variation=0.02, read_variation=0.02,
        vnoise=0.05, wire_resistance=0,
        rdac=2 ** 4, radc=2 ** 8, vread=0.2,
        rate_stuck_HGS=0.001, rate_stuck_LGS=0.000,
        drift_coefficient=0.0,
        drift_time=1.0,
        drift_reference_time=1.0,
        mode=0,
        mode2_input_mode="signed",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        mode1_paral_size=(64, 64),
        mode1_adc_per_tile=True,
        inference_chunk_size=None,
        fast_inference=False,
        fast_inference_backend="torch",
        triton_input_precision="ieee",
        triton_block_r=32,
        triton_block_l=16,
        triton_block_k=64,
        triton_output_chunk_limit=256,
        triton_auto_config=False,
        triton_gidx_read_noise=False,
        triton_gidx_fused_restore_read_noise=True,
        triton_gidx_fuse_input_slices=False,
        triton_reuse_input_voltage=False,
        triton_fuse_restored_input_slices=False,
        triton_fuse_activation_slices=True,
        triton_fuse_output_finalize=False,
        triton_direct_final_output=True,
        triton_gidx_direct_final_output=True,
        triton_mode1_gidx_direct_final=True,
        triton_mode1_input_tile_group=1,
        triton_mode1_chunked_direct_final=False,
        triton_mode2_diff_direct_final=True,
        triton_mode2_diff_gidx_from_slices=False,
        triton_mode2_diff_activation_slices=False,
        triton_mode2_diff_presubtract=False,
        triton_mode2_diff_fuse_input_slices=True,
        triton_mode2_diff_block_r_cap=16,
        triton_mode2_diff_block_l_cap=8,
        mode1_grouped_tile_gemm=True,
        direct_output_chunk_write=True,
        read_variation_seed=None,
        write_variation_mode="materialized",
        conductance_dtype=torch.float32,
        compute_dtype=torch.float32,
        profile=False,
        profile_sync_cuda=True,
    ):
        """
        Initialize the multi-mode dot product engine.
        
        Parameters:
            HGS (float): High conductance state.
            LGS (float): Low conductance state.
            g_level (int): Number of conductance levels.
            write_variation (float): Lognormal write variation applied to conductance.
            read_variation (dict or float): Read variation per conductance level, or one value shared by all levels.
            vnoise (float): Multiplicative input voltage noise standard deviation.
            wire_resistance (float): Wire resistance value. Nonzero values are not supported.
            rdac (int): DAC resolution used for input voltage quantization.
            radc (int, list, or tuple): ADC resolution, optionally one value per weight slice.
            vread (float): Read voltage.
            rate_stuck_HGS (float): Probability of stuck-at-HGS faults.
            rate_stuck_LGS (float): Probability of stuck-at-LGS faults.
            drift_coefficient (float): Conductance drift exponent nu. Zero disables drift.
            drift_time (float): Effective elapsed time for drift evaluation.
            drift_reference_time (float): Reference programming time t0 for drift.
            mode (int): Mapping mode. 0 is standard slicing, 1 is linear differential-pair projection, 2 is differential-pair slicing.
            mode2_input_mode (str): Mode 2 input drive. "signed" uses the signed sliced voltage view; "differential"
                drives positive and negative input branches in two read phases and subtracts them digitally.
            device (torch.device): Device used for all generated tensors.
            mode1_paral_size (tuple or None): Mode 1 tile size as (input_features, output_features).
            mode1_adc_per_tile (bool): If True, quantize Mode 1 current per input tile; otherwise quantize after accumulation
                only when the full input dimension fits in one Mode 1 tile.
            inference_chunk_size (int or None): Optional number of output tile columns processed per inference chunk.
            fast_inference (bool): If True, fuse all weight slices for each input slice in the inference path
                when a scalar ADC is used. This reduces small-kernel launch overhead at the cost of a larger
                per-chunk intermediate tensor.
            fast_inference_backend (str): Backend for the fused inference path. "torch" is the default.
                "triton" enables experimental CUDA kernels for 2-D Linear inference and falls back to torch
                for unsupported cases, including a native two-phase mode2 differential-input kernel.
                "triton_gidx" enables the compressed G-index fused backend for legal 2-D CUDA
                inference cases and falls back through the non-G-index path when unsupported.
            triton_input_precision (str): Triton dot precision, one of "ieee", "tf32", or "tf32x3".
            triton_block_r (int): Triton tile size across output rows.
            triton_block_l (int): Triton tile size across tile output columns.
            triton_block_k (int): Triton tile size across tile input columns.
            triton_output_chunk_limit (int): Maximum number of output tile columns
                per Triton launch. This bounds pointer offsets and wide-layer
                temporary tensors; larger values reduce chunk launches at higher
                peak memory.
            triton_auto_config (bool): Use conservative shape-aware block and
                chunk choices for mode-0 Triton kernels. This is a software
                scheduling guard only; it does not change analog simulation
                semantics.
            triton_gidx_read_noise (bool): Try the experimental compressed G-index kernel with
                uniform read variation generated inside the Triton kernel.
            triton_gidx_fused_restore_read_noise (bool): Try the experimental fused restore
                kernel for compressed G-index tensors with uniform read variation.
            triton_gidx_fuse_input_slices (bool): Try the experimental mode-0 kernel
                that fuses all input slices while preserving per-slice ADC.
            triton_reuse_input_voltage (bool): In fused input-slice G-index kernels,
                quantize each input-voltage tile once and reuse it across weight
                slices. Per-slice ADC and read variation are still applied.
            triton_fuse_restored_input_slices (bool): Try the experimental mode-0
                kernel that fuses input slices after conductance has already been
                restored, including the read-variation restored path.
            triton_fuse_activation_slices (bool): Try the experimental mode-0
                activation slicing kernel for the restricted LLM SLC path where
                quantization granularity equals the array tile.
            triton_fuse_output_finalize (bool): Try the experimental mode-0
                2-D finalize kernel that fuses output scale, input-tile
                reduction, and layout writeback after the VMM kernel.
            triton_direct_final_output (bool): Try the experimental mode-0
                restored-conductance 2-D kernel that writes the finalized
                [tokens, output_features] chunk directly, avoiding the 5-D
                MapReduce intermediate.
            triton_gidx_direct_final_output (bool): When direct-final-output
                is enabled, try the compressed G-index direct-final kernel
                before restoring conductance tensors.
            triton_mode1_input_tile_group (int): Number of Mode-1 input tiles
                reduced inside one direct-final Triton program before writing
                the output. Values >1 reduce final-output atomic writes when
                the full input dimension spans multiple Mode-1 tiles.
            triton_mode1_chunked_direct_final (bool): Try Mode-1 direct-final
                on output chunks for very wide layers. This is experimental
                and default-off because it can be slower than the grouped GEMM
                fallback on LLM MLP expansion layers.
            direct_output_chunk_write (bool): Write finalized 2-D output
                chunks into the final output buffer directly instead of keeping
                all chunks and concatenating them at the end.
            read_variation_seed (int or None): Optional seed for reproducible Triton read-noise
                streams. None uses a system-entropy base seed per engine instance.
            write_variation_mode (str): "materialized" stores programmed write variation in G tensors.
                "virtual" stores level indices and regenerates deterministic write variation per chunk.
            conductance_dtype (torch.dtype or str): dtype used for stored materialized conductance tensors.
            compute_dtype (torch.dtype or str): dtype used for restored conductance and input-voltage tensors.
            profile (bool): Collect synchronized timing events for the inference path.
            profile_sync_cuda (bool): Synchronize CUDA before/after profiled regions for accurate timings.
        """
        self.HGS = HGS
        self.LGS = LGS
        self.g_level = g_level
        self.write_variation = write_variation
        if write_variation_mode not in ("materialized", "virtual"):
            raise ValueError("write_variation_mode must be 'materialized' or 'virtual'.")
        self.write_variation_mode = write_variation_mode
        self.conductance_dtype = self._resolve_float_dtype(conductance_dtype, "conductance_dtype")
        self.compute_dtype = self._resolve_float_dtype(compute_dtype, "compute_dtype")
        self.mode = mode
        if mode2_input_mode not in ("signed", "differential"):
            raise ValueError("mode2_input_mode must be 'signed' or 'differential'.")
        self.mode2_input_mode = mode2_input_mode

        if mode == 1 and g_level <= 2:
            raise ValueError("Mode 1 requires g_level > 2.")

        if isinstance(read_variation, dict):
            if set(read_variation.keys()) != set(range(g_level)):
                raise ValueError(
                    f"read_variation dict must have keys for all g levels 0..{g_level - 1}."
                )
            self.read_variation = read_variation
        elif isinstance(read_variation, (int, float)):
            self.read_variation = {i: read_variation for i in range(g_level)}
        else:
            raise ValueError("read_variation must be a dict or float.")

        self.vnoise = vnoise
        self.rdac = rdac

        if isinstance(radc, (list, tuple)):
            self.radc = torch.tensor(radc, device=device).flip(0)
            self.radc_is_list = True
        else:
            self.radc = radc
            self.radc_is_list = False

        self.vread = vread
        self.wire_resistance = wire_resistance
        self.rate_stuck_HGS = rate_stuck_HGS
        self.rate_stuck_LGS = rate_stuck_LGS
        self.drift_coefficient = drift_coefficient
        self.drift_time = drift_time
        self.drift_reference_time = drift_reference_time
        self.mode1_adc_per_tile = mode1_adc_per_tile
        self.mode1_paral_size = mode1_paral_size
        self.inference_chunk_size = inference_chunk_size
        self.fast_inference = fast_inference
        if fast_inference_backend not in ("torch", "triton", "triton_gidx"):
            raise ValueError("fast_inference_backend must be 'torch', 'triton', or 'triton_gidx'.")
        if triton_input_precision not in ("ieee", "tf32", "tf32x3"):
            raise ValueError("triton_input_precision must be 'ieee', 'tf32', or 'tf32x3'.")
        self.fast_inference_backend = fast_inference_backend
        self.triton_input_precision = triton_input_precision
        self.triton_block_r = int(triton_block_r)
        self.triton_block_l = int(triton_block_l)
        self.triton_block_k = int(triton_block_k)
        self.triton_output_chunk_limit = max(1, int(triton_output_chunk_limit))
        self.triton_auto_config = bool(triton_auto_config)
        self.triton_gidx_read_noise = bool(triton_gidx_read_noise)
        self.triton_gidx_fused_restore_read_noise = bool(triton_gidx_fused_restore_read_noise)
        self.triton_gidx_fuse_input_slices = bool(triton_gidx_fuse_input_slices)
        self.triton_reuse_input_voltage = bool(triton_reuse_input_voltage)
        self.triton_fuse_restored_input_slices = bool(triton_fuse_restored_input_slices)
        self.triton_fuse_activation_slices = bool(triton_fuse_activation_slices)
        self.triton_fuse_output_finalize = bool(triton_fuse_output_finalize)
        self.triton_direct_final_output = bool(triton_direct_final_output)
        self.triton_gidx_direct_final_output = bool(triton_gidx_direct_final_output)
        self.triton_mode1_gidx_direct_final = bool(triton_mode1_gidx_direct_final)
        self.triton_mode1_input_tile_group = max(1, int(triton_mode1_input_tile_group))
        self.triton_mode1_chunked_direct_final = bool(triton_mode1_chunked_direct_final)
        self.triton_mode2_diff_direct_final = bool(triton_mode2_diff_direct_final)
        self.triton_mode2_diff_gidx_from_slices = bool(triton_mode2_diff_gidx_from_slices)
        self.triton_mode2_diff_activation_slices = bool(triton_mode2_diff_activation_slices)
        self.triton_mode2_diff_presubtract = bool(triton_mode2_diff_presubtract)
        self.triton_mode2_diff_fuse_input_slices = bool(triton_mode2_diff_fuse_input_slices)
        self.triton_mode2_diff_block_r_cap = int(triton_mode2_diff_block_r_cap)
        self.triton_mode2_diff_block_l_cap = int(triton_mode2_diff_block_l_cap)
        self.mode1_grouped_tile_gemm = bool(mode1_grouped_tile_gemm)
        self.direct_output_chunk_write = bool(direct_output_chunk_write)
        self.read_variation_seed = None if read_variation_seed is None else int(read_variation_seed)
        if self.read_variation_seed is None:
            self._read_noise_seed_base = secrets.randbelow(2**31 - 1)
        else:
            self._read_noise_seed_base = self.read_variation_seed
        self.profile = profile
        self.profile_sync_cuda = profile_sync_cuda
        self.profile_events = []
        self.fastpath_counters = {}
        self._triton_auto_plan_cache = {}
        self._triton_auto_counter_seen = set()
        self._read_noise_restore_counter = 0
        self.device = device

        if self.radc_is_list:
            if torch.any(self.radc < 2):
                raise ValueError("All ADC resolution values must be >= 2.")
        else:
            if self.radc < 2:
                raise ValueError("ADC resolution must be >= 2.")
        if self.rdac < 2:
            raise ValueError("DAC resolution must be >= 2.")
        if self.g_level < 2:
            raise ValueError("g_level must be >= 2.")
        if self.LGS >= self.HGS:
            raise ValueError("LGS must be < HGS.")
        if self.rate_stuck_HGS + self.rate_stuck_LGS > 1:
            raise ValueError("Sum of stuck-at rates must not exceed 1.")
        if self.drift_coefficient < 0:
            raise ValueError("drift_coefficient must be non-negative.")
        if self.drift_time <= 0 or self.drift_reference_time <= 0:
            raise ValueError("drift_time and drift_reference_time must be positive.")
        if self.mode1_paral_size is not None:
            if (not isinstance(self.mode1_paral_size, (list, tuple))
                    or len(self.mode1_paral_size) != 2
                    or any(v <= 0 for v in self.mode1_paral_size)):
                raise ValueError(
                    "mode1_paral_size must be None or a 2-tuple of positive ints."
                )

        self.Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        self.conductance_levels = torch.tensor(
            [self.LGS + i * self.Q_G for i in range(self.g_level)],
            device=self.device,
            dtype=self.conductance_dtype,
        )

        # Pre-build per-level variation tensor for fast _gen_read_noise
        self._rv_all_same = all(
            v == list(self.read_variation.values())[0]
            for v in self.read_variation.values()
        )
        self._rv_sigma = list(self.read_variation.values())[0] if self._rv_all_same else 0.0
        self._has_read_noise = any(v > 0 for v in self.read_variation.values())
        self._read_variation_tensor = torch.tensor(
            [self.read_variation[i] for i in range(self.g_level)],
            device=self.device,
            dtype=torch.float32,
        )
        self._has_drift = (
            self.drift_coefficient > 0
            and self.drift_time != self.drift_reference_time
        )
        self._drift_scale = (
            (self.drift_time / self.drift_reference_time) ** (-self.drift_coefficient)
            if self._has_drift
            else 1.0
        )
        self._g_index_dtype = self._resolve_g_index_dtype()

    @staticmethod
    def _resolve_float_dtype(dtype, name):
        """Resolve a user-facing dtype value to a floating torch dtype."""
        if isinstance(dtype, str):
            key = dtype.replace("torch.", "").lower()
            if key in ("float32", "fp32"):
                dtype = torch.float32
            elif key in ("float16", "fp16", "half"):
                dtype = torch.float16
            elif key in ("bfloat16", "bf16"):
                dtype = torch.bfloat16
            else:
                raise ValueError(f"{name} must be float32, float16, or bfloat16.")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"{name} must be float32, float16, or bfloat16.")
        return dtype

    def _write_variation_is_virtual(self):
        return self.write_variation > 0 and self.write_variation_mode == "virtual"

    def reset_fastpath_counters(self):
        self.fastpath_counters = {
            "gidx_attempt_count": 0,
            "gidx_success_count": 0,
            "gidx_fallback_count": 0,
            "gidx_read_noise_fallback_count": 0,
            "gidx_virtual_write_fallback_count": 0,
            "gidx_import_fallback_count": 0,
            "gidx_exception_fallback_count": 0,
            "gidx_read_noise_aware_attempt_count": 0,
            "gidx_read_noise_aware_success_count": 0,
            "gidx_input_slice_fused_attempt_count": 0,
            "gidx_input_slice_fused_success_count": 0,
            "gidx_input_slice_fused_fallback_count": 0,
            "gidx_read_noise_input_slice_fused_attempt_count": 0,
            "gidx_read_noise_input_slice_fused_success_count": 0,
            "gidx_read_noise_input_slice_fused_fallback_count": 0,
            "restored_input_slice_fused_attempt_count": 0,
            "restored_input_slice_fused_success_count": 0,
            "restored_input_slice_fused_fallback_count": 0,
            "output_finalize_fused_attempt_count": 0,
            "output_finalize_fused_success_count": 0,
            "output_finalize_fused_fallback_count": 0,
            "direct_final_output_attempt_count": 0,
            "direct_final_output_success_count": 0,
            "direct_final_output_fallback_count": 0,
            "direct_final_output_store_success_count": 0,
            "gidx_direct_final_output_attempt_count": 0,
            "gidx_direct_final_output_success_count": 0,
            "gidx_direct_final_output_fallback_count": 0,
            "gidx_direct_final_output_store_success_count": 0,
            "mode1_tile_group_attempt_count": 0,
            "mode1_tile_group_success_count": 0,
            "mode1_tile_group_fallback_count": 0,
            "mode2_diff_triton_attempt_count": 0,
            "mode2_diff_triton_success_count": 0,
            "mode2_diff_triton_fallback_count": 0,
            "mode2_diff_slice_triton_attempt_count": 0,
            "mode2_diff_slice_triton_success_count": 0,
            "mode2_diff_slice_triton_fallback_count": 0,
            "mode2_diff_gidx_slice_triton_attempt_count": 0,
            "mode2_diff_gidx_slice_triton_success_count": 0,
            "mode2_diff_gidx_slice_triton_fallback_count": 0,
            "mode2_diff_direct_final_attempt_count": 0,
            "mode2_diff_direct_final_success_count": 0,
            "mode2_diff_direct_final_fallback_count": 0,
            "mode2_diff_direct_final_store_success_count": 0,
            "mode2_diff_gidx_direct_final_attempt_count": 0,
            "mode2_diff_gidx_direct_final_success_count": 0,
            "mode2_diff_gidx_direct_final_fallback_count": 0,
            "mode2_diff_gidx_direct_final_store_success_count": 0,
            "mode2_diff_presubtract_count": 0,
            "mode2_diff_presubtract_triton_success_count": 0,
            "mode2_diff_input_slice_fused_attempt_count": 0,
            "mode2_diff_input_slice_fused_success_count": 0,
            "mode2_diff_input_slice_fused_fallback_count": 0,
            "mode1_gidx_direct_final_attempt_count": 0,
            "mode1_gidx_direct_final_success_count": 0,
            "mode1_gidx_direct_final_fallback_count": 0,
            "mode1_gidx_direct_final_grouped_success_count": 0,
            "mode1_chunked_direct_final_attempt_count": 0,
            "mode1_chunked_direct_final_success_count": 0,
            "mode1_chunked_direct_final_fallback_count": 0,
            "mode0_triton_auto_config_count": 0,
            "mode0_triton_auto_chunk_count": 0,
            "mode1_triton_auto_config_count": 0,
            "mode1_triton_auto_chunk_count": 0,
            "mode1_triton_auto_group_count": 0,
        }
        self._read_noise_restore_counter = 0
        self._triton_auto_counter_seen = set()
        return self

    def get_fastpath_counters(self):
        if not getattr(self, "fastpath_counters", None):
            self.reset_fastpath_counters()
        return dict(self.fastpath_counters)

    def _fastpath_count(self, key, value=1):
        if not getattr(self, "fastpath_counters", None):
            self.reset_fastpath_counters()
        self.fastpath_counters[key] = self.fastpath_counters.get(key, 0) + value

    def _gidx_fallback_reason(self, vin, mat):
        if self.fast_inference_backend != "triton_gidx":
            return "backend_not_gidx"
        if len(vin.shape) != 4:
            return "not_2d_linear"
        if not vin.is_cuda:
            return "not_cuda"
        if self.mode not in (0, 2):
            return "unsupported_mode"
        if self.radc_is_list:
            return "radc_list"
        if not getattr(mat, "G_is_compressed", False):
            return "not_compressed"
        if self._write_variation_is_virtual():
            return "virtual_write_variation"
        if self._has_read_noise:
            return "read_noise"
        return "unknown"

    def _record_gidx_fallback(self, reason):
        self._fastpath_count("gidx_fallback_count")
        if reason == "read_noise":
            self._fastpath_count("gidx_read_noise_fallback_count")
        elif reason == "virtual_write_variation":
            self._fastpath_count("gidx_virtual_write_fallback_count")
        key = f"gidx_fallback_reason_{reason}"
        self._fastpath_count(key)

    @staticmethod
    def _shape_int(value, default=1):
        try:
            return max(1, int(value))
        except Exception:
            return max(1, int(default))

    def _mode0_triton_shape(self, x=None, mat=None, *, g=None, gidx=None, vin=None):
        rows = None
        tile_cols = None
        tile_k = None
        weight_slices = None
        in_tiles = None
        out_tiles = None
        in_features = None
        out_features = None

        x_sliced = getattr(x, "sliced_data", None) if x is not None else None
        if x_sliced is not None and getattr(x_sliced, "dim", lambda: 0)() >= 5:
            rows = int(x_sliced.shape[0]) * int(x_sliced.shape[3])
            tile_k = int(x_sliced.shape[4])
            in_tiles = int(x_sliced.shape[1])
        elif vin is not None and getattr(vin, "dim", lambda: 0)() >= 4:
            rows = int(vin.shape[0]) * int(vin.shape[2])
            tile_k = int(vin.shape[3])
            in_tiles = int(vin.shape[1])
        elif x is not None and getattr(x, "shape", None) is not None:
            rows = int(x.shape[0]) if len(x.shape) > 0 else 1
            if len(x.shape) > 1:
                in_features = int(x.shape[-1])

        source = gidx if gidx is not None else g
        if source is None and mat is not None:
            if getattr(mat, "G_indices", None) is not None and not isinstance(mat.G_indices, tuple):
                source = mat.G_indices
            elif getattr(mat, "G", None) is not None and not isinstance(mat.G, tuple):
                source = mat.G
        if source is not None and getattr(source, "dim", lambda: 0)() >= 5:
            weight_slices = int(source.shape[2])
            tile_k = int(source.shape[3])
            tile_cols = int(source.shape[4])
            in_tiles = int(source.shape[0]) if in_tiles is None else in_tiles
            out_tiles = int(source.shape[1])
        if mat is not None and getattr(mat, "shape", None) is not None:
            in_features = int(mat.shape[0])
            out_features = int(mat.shape[1])
        if mat is not None and getattr(mat, "max_data", None) is not None:
            out_tiles = int(mat.max_data.shape[1]) if out_tiles is None else out_tiles

        return {
            "rows": self._shape_int(rows),
            "tile_cols": self._shape_int(tile_cols, self.triton_block_l),
            "tile_k": self._shape_int(tile_k, self.triton_block_k),
            "weight_slices": self._shape_int(weight_slices),
            "in_tiles": self._shape_int(in_tiles),
            "out_tiles": self._shape_int(out_tiles),
            "in_features": self._shape_int(in_features),
            "out_features": self._shape_int(out_features),
        }

    def _triton_auto_plan(self, mode, shape):
        rows = self._shape_int(shape.get("rows"))
        tile_cols = self._shape_int(shape.get("tile_cols"), self.triton_block_l)
        tile_k = self._shape_int(shape.get("tile_k"), self.triton_block_k)
        in_tiles = self._shape_int(shape.get("in_tiles"))
        out_tiles = self._shape_int(shape.get("out_tiles"))
        in_features = self._shape_int(shape.get("in_features"))
        out_features = self._shape_int(shape.get("out_features"))
        key = (
            int(mode),
            rows,
            tile_cols,
            tile_k,
            in_tiles,
            out_tiles,
            in_features,
            out_features,
            self._shape_int(getattr(self, "triton_output_chunk_limit", 256)),
            self._shape_int(getattr(self, "triton_block_r", 32)),
            self._shape_int(getattr(self, "triton_block_l", 16)),
            self._shape_int(getattr(self, "triton_block_k", 64)),
        )
        cached = self._triton_auto_plan_cache.get(key)
        if cached is not None:
            return cached

        block_r = max(1, int(self.triton_block_r))
        block_l = max(1, int(self.triton_block_l))
        block_k = max(1, int(self.triton_block_k))
        chunk_limit = max(1, int(getattr(self, "triton_output_chunk_limit", 256)))
        input_tile_group = max(1, int(getattr(self, "triton_mode1_input_tile_group", 1)))

        if int(mode) == 0:
            if rows >= 64:
                block_r = 64
            elif rows >= 32:
                block_r = 32
            else:
                block_r = 16

            if tile_cols >= 16:
                block_l = 16
            elif tile_cols >= 8:
                block_l = 8
            else:
                block_l = max(1, tile_cols)

            block_k = 64 if tile_k >= 64 else max(16, tile_k)
            chunk_limit = min(out_tiles, chunk_limit)
            if out_features >= 32768 and rows >= 64:
                chunk_limit = min(chunk_limit, 512)
            elif out_features <= 8192:
                chunk_limit = min(chunk_limit, max(1, out_tiles))
        elif int(mode) == 1:
            # Mode 1 uses compact differential-pair indices. Prior sweeps showed
            # group>1 and chunk512 can regress on Qwen-style MLP/lm_head layers,
            # so the auto plan keeps those guards while still specializing
            # block/chunk metadata per shape.
            block_r = 32 if rows >= 32 else 16
            block_l = 16 if tile_cols >= 16 else max(1, tile_cols)
            block_k = 64 if tile_k >= 64 else max(16, tile_k)
            chunk_limit = min(out_tiles, chunk_limit, 256)
            if out_features <= 4096:
                chunk_limit = min(chunk_limit, max(1, out_tiles))
            input_tile_group = 1

        if block_k not in (16, 32, 64, 128):
            block_k = 64 if block_k > 64 else 32 if block_k > 32 else 16
        if block_r not in (16, 32, 64, 128):
            block_r = 64 if block_r > 64 else 32 if block_r > 32 else 16
        if block_l not in (1, 2, 4, 8, 16, 32):
            block_l = 16 if block_l > 16 else 8 if block_l > 8 else 4

        plan = {
            "block_r": int(block_r),
            "block_l": int(block_l),
            "block_k": int(block_k),
            "chunk_limit": int(max(1, chunk_limit)),
            "input_tile_group": int(max(1, input_tile_group)),
            "shape_key": (
                f"auto:mode={int(mode)};br={int(block_r)};bl={int(block_l)};bk={int(block_k)};"
                f"chunk={int(max(1, chunk_limit))};group={int(max(1, input_tile_group))};"
                f"rows={rows};in={in_features};out={out_features};"
                f"tiles={in_tiles}x{out_tiles};l={tile_cols};k={tile_k}"
            ),
        }
        self._triton_auto_plan_cache[key] = plan
        return plan

    def _mode0_triton_blocks(self, x=None, mat=None, *, g=None, gidx=None, vin=None):
        block_r = max(1, int(self.triton_block_r))
        block_l = max(1, int(self.triton_block_l))
        block_k = max(1, int(self.triton_block_k))
        if not (self.triton_auto_config and self.mode == 0):
            return block_r, block_l, block_k, "manual"

        shape = self._mode0_triton_shape(x, mat, g=g, gidx=gidx, vin=vin)
        plan = self._triton_auto_plan(0, shape)
        block_r = plan["block_r"]
        block_l = plan["block_l"]
        block_k = plan["block_k"]
        key = f"mode0_triton_auto_block_r{block_r}_l{block_l}_k{block_k}"
        self._fastpath_count("mode0_triton_auto_config_count")
        self._fastpath_count(key)
        return block_r, block_l, block_k, plan["shape_key"]

    def _mode0_triton_chunk_limit(self, ndc_y, mat=None):
        limit = max(1, int(getattr(self, "triton_output_chunk_limit", 256)))
        if not (self.triton_auto_config and self.mode == 0):
            return limit
        ndc_y = max(1, int(ndc_y))
        shape = self._mode0_triton_shape(mat=mat)
        shape["out_tiles"] = ndc_y
        selected = min(ndc_y, self._triton_auto_plan(0, shape)["chunk_limit"])
        self._fastpath_count("mode0_triton_auto_chunk_count")
        self._fastpath_count(f"mode0_triton_auto_chunk_{selected}")
        return selected

    def _mode1_triton_shape(self, x_2d=None, mat=None, *, tile_in=None, tile_out=None, out_cols=None):
        rows = int(x_2d.shape[0]) if x_2d is not None and getattr(x_2d, "dim", lambda: 0)() >= 2 else None
        in_features = int(x_2d.shape[1]) if x_2d is not None and getattr(x_2d, "dim", lambda: 0)() >= 2 else None
        out_features = int(out_cols) if out_cols is not None else None
        if mat is not None and getattr(mat, "shape", None) is not None:
            in_features = int(mat.shape[0]) if in_features is None else in_features
            out_features = int(mat.shape[1]) if out_features is None else out_features
        tile_in = self._shape_int(tile_in, getattr(mat, "paral_size", (64, 64))[0] if mat is not None else 64)
        tile_out = self._shape_int(tile_out, getattr(mat, "paral_size", (64, 64))[1] if mat is not None else 64)
        in_tiles = (self._shape_int(in_features) + tile_in - 1) // tile_in
        out_tiles = (self._shape_int(out_features) + tile_out - 1) // tile_out
        return {
            "rows": self._shape_int(rows),
            "tile_cols": tile_out,
            "tile_k": tile_in,
            "weight_slices": 1,
            "in_tiles": self._shape_int(in_tiles),
            "out_tiles": self._shape_int(out_tiles),
            "in_features": self._shape_int(in_features),
            "out_features": self._shape_int(out_features),
        }

    def _mode1_triton_plan(self, x_2d, mat, tile_in, tile_out, *, out_cols=None):
        if not (self.triton_auto_config and self.mode == 1):
            return {
                "block_r": max(1, int(self.triton_block_r)),
                "block_l": max(1, int(self.triton_block_l)),
                "block_k": max(1, int(self.triton_block_k)),
                "chunk_limit": max(1, int(getattr(self, "triton_output_chunk_limit", 256))),
                "input_tile_group": max(1, int(getattr(self, "triton_mode1_input_tile_group", 1))),
                "shape_key": "manual",
            }
        rows = int(x_2d.shape[0]) if x_2d is not None and getattr(x_2d, "dim", lambda: 0)() >= 2 else 1
        planned_out_cols = int(out_cols) if out_cols is not None else int(mat.shape[1])
        cache_key = (
            rows,
            planned_out_cols,
            int(tile_in),
            int(tile_out),
            int(getattr(self, "triton_output_chunk_limit", 256)),
            int(getattr(self, "triton_block_r", 32)),
            int(getattr(self, "triton_block_l", 16)),
            int(getattr(self, "triton_block_k", 64)),
        )
        cache = getattr(mat, "_mode1_triton_plan_cache", None)
        if cache is None:
            cache = {}
            setattr(mat, "_mode1_triton_plan_cache", cache)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        shape = self._mode1_triton_shape(x_2d, mat, tile_in=tile_in, tile_out=tile_out, out_cols=out_cols)
        plan = self._triton_auto_plan(1, shape)
        counter_key = ("mode1_plan", plan["shape_key"])
        if counter_key not in self._triton_auto_counter_seen:
            self._triton_auto_counter_seen.add(counter_key)
            self._fastpath_count("mode1_triton_auto_config_count")
            self._fastpath_count(
                f"mode1_triton_auto_block_r{plan['block_r']}_l{plan['block_l']}_k{plan['block_k']}"
            )
            self._fastpath_count("mode1_triton_auto_group_count")
            self._fastpath_count(f"mode1_triton_auto_group_{plan['input_tile_group']}")
        cache[cache_key] = plan
        return plan

    def _mode1_triton_chunk_limit(self, ndc_y, mat=None):
        limit = max(1, int(getattr(self, "triton_output_chunk_limit", 256)))
        if not (self.triton_auto_config and self.mode == 1):
            return limit
        ndc_y = max(1, int(ndc_y))
        tile_in, tile_out = self._resolve_mode1_tile_size(
            mat,
            int(mat.shape[0]),
            int(mat.shape[1]),
        ) if mat is not None else (64, 64)
        if mat is not None:
            cache_key = (
                ndc_y,
                int(tile_in),
                int(tile_out),
                int(getattr(self, "triton_output_chunk_limit", 256)),
                int(getattr(self, "triton_block_r", 32)),
                int(getattr(self, "triton_block_l", 16)),
                int(getattr(self, "triton_block_k", 64)),
            )
            cache = getattr(mat, "_mode1_triton_chunk_cache", None)
            if cache is None:
                cache = {}
                setattr(mat, "_mode1_triton_chunk_cache", cache)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        shape = self._mode1_triton_shape(None, mat, tile_in=tile_in, tile_out=tile_out)
        shape["out_tiles"] = ndc_y
        selected = min(ndc_y, self._triton_auto_plan(1, shape)["chunk_limit"])
        counter_key = ("mode1_chunk", ndc_y, selected)
        if counter_key not in self._triton_auto_counter_seen:
            self._triton_auto_counter_seen.add(counter_key)
            self._fastpath_count("mode1_triton_auto_chunk_count")
            self._fastpath_count(f"mode1_triton_auto_chunk_{selected}")
        if mat is not None:
            cache[cache_key] = selected
        return selected

    def reset_profile(self):
        """Clear collected profiling events."""
        self.profile_events = []

    def get_profile_summary(self):
        """Return per-label total/mean timings for collected profiling events."""
        summary = {}
        for event in self.profile_events:
            label = event["label"]
            item = summary.setdefault(label, {
                "count": 0,
                "total_ms": 0.0,
                "mean_ms": 0.0,
                "statuses": {},
                "reasons": {},
                "messages": {},
                "metadata": {},
            })
            item["count"] += 1
            item["total_ms"] += event["ms"]
            status = event.get("status")
            if status is not None:
                status = str(status)
                item["statuses"][status] = item["statuses"].get(status, 0) + 1
            reason = event.get("reason")
            if reason is not None:
                reason = str(reason)
                item["reasons"][reason] = item["reasons"].get(reason, 0) + 1
            message = event.get("message")
            if message is not None:
                message = str(message)
                if len(message) > 180:
                    message = message[:177] + "..."
                item["messages"][message] = item["messages"].get(message, 0) + 1
            shape_key = event.get("shape_key")
            if shape_key is not None:
                shape_key = str(shape_key)
                group = item["metadata"].setdefault(shape_key, {
                    "count": 0,
                    "total_ms": 0.0,
                    "mean_ms": 0.0,
                })
                group["count"] += 1
                group["total_ms"] += event["ms"]
        for item in summary.values():
            item["mean_ms"] = item["total_ms"] / max(item["count"], 1)
            for group in item.get("metadata", {}).values():
                group["mean_ms"] = group["total_ms"] / max(group["count"], 1)
        return summary

    def _profile_synchronize(self):
        device_type = self.device.type if hasattr(self.device, "type") else torch.device(self.device).type
        if (
            self.profile
            and self.profile_sync_cuda
            and torch.cuda.is_available()
            and device_type == "cuda"
        ):
            torch.cuda.synchronize(self.device)

    def _profile_start(self, label):
        if not self.profile:
            return None
        self._profile_synchronize()
        nvtx_pushed = False
        device_type = self.device.type if hasattr(self.device, "type") else torch.device(self.device).type
        if torch.cuda.is_available() and device_type == "cuda":
            torch.cuda.nvtx.range_push(f"memintelli:{label}")
            nvtx_pushed = True
        return label, time.perf_counter(), nvtx_pushed

    def _profile_stop(self, token, **metadata):
        if token is None:
            return
        self._profile_synchronize()
        label, t0, nvtx_pushed = token
        event = {"label": label, "ms": (time.perf_counter() - t0) * 1000.0}
        event.update(metadata)
        self.profile_events.append(event)
        if nvtx_pushed:
            torch.cuda.nvtx.range_pop()


    def __call__(self, x: SlicedDataMultiMode, mat: SlicedDataMultiMode):
        """
        Run the dot product engine.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
        
        Returns:
            torch.Tensor: Matrix multiplication result.
        """
        return self.MapReduceDot(x, mat)

    def MapReduceDot(self, x: SlicedDataMultiMode, mat: SlicedDataMultiMode):
        """
        Compute matrix multiplication through the selected memory-array mode.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedDataMultiMode): Sliced weight tensor with shape (n, p).
        
        Returns:
            torch.Tensor: Output tensor with shape (m, p) or (batch, m, p).
        """
        self._validate_sliced_data_mode(x, "x")
        self._validate_sliced_data_mode(mat, "mat")

        if mat.device.type != x.device.type:
            raise ValueError("x and mat must be on the same device.")
        if x.shape[-1] != mat.shape[-2]:
            raise ValueError("Input / weight dimension mismatch.")

        if self.mode != 1:
            self._validate_radc_with_slices(mat)
        if self.wire_resistance > 0:
            raise NotImplementedError("Wire resistance is not supported.")

        use_inference = getattr(x, "inference", False) or getattr(mat, "inference", False)
        if self.mode == 1:
            if use_inference:
                return self._dot_mode1_inference(x, mat)
            return self._dot_mode1(x, mat, self._num2V, self._gen_read_noise)

        # Mode 0 / 2: use memory-efficient inference path when applicable
        if use_inference:
            return self._dot_inference(x, mat)
        return self._dot(x, mat, self._num2V, self._gen_read_noise)


    def _validate_sliced_data_mode(self, sd: SlicedDataMultiMode, name: str):
        """
        Validate that a sliced data object matches this engine.
        
        Parameters:
            sd (SlicedDataMultiMode): Object to validate.
            name (str): Name used in error messages.
        
        Returns:
            None.
        """
        if not isinstance(sd, SlicedDataMultiMode):
            raise TypeError(
                f"'{name}' must be a SlicedDataMultiMode instance "
                f"(from the multimode flash data format module). Got {type(sd).__name__}."
            )
        if sd.mode != self.mode:
            raise ValueError(
                f"Mode mismatch: engine mode={self.mode} but "
                f"'{name}' was created with mode={sd.mode}. "
                f"Use SlicedDataMultiMode(..., mode={self.mode})."
            )

    def _validate_radc_with_slices(self, mat: SlicedDataMultiMode):
        """
        Validate per-slice ADC resolution length.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor whose slice count is checked.
        
        Returns:
            None.
        """
        if self.radc_is_list:
            n = len(mat)
            if len(self.radc) != n:
                raise ValueError(
                    f"radc list length ({len(self.radc)}) must match "
                    f"number of weight slices ({n})."
                )

    def _resolve_mode1_tile_size(self, mat, in_features, out_features):
        """
        Resolve the Mode 1 tile size from weight metadata or engine defaults.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor that may carry a paral_size override.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        
        Returns:
            tuple: (tile_in, tile_out) clipped to the actual matrix dimensions.
        """
        tile_size = None
        if hasattr(mat, "paral_size") and mat.paral_size is not None:
            tile_size = mat.paral_size
        elif self.mode1_paral_size is not None:
            tile_size = self.mode1_paral_size
        if tile_size is None:
            return in_features, out_features
        return (
            max(1, min(int(tile_size[0]), in_features)),
            max(1, min(int(tile_size[1]), out_features)),
        )

    def _quantize_mode1_current(self, current, adc_ref):
        """
        Apply ADC quantization to a Mode 1 current tensor.
        
        Parameters:
            current (torch.Tensor): Current tensor before ADC quantization.
            adc_ref (float or torch.Tensor): ADC reference current for normalization.
        
        Returns:
            torch.Tensor: Quantized normalized current.
        """
        radc_val = int(self.radc.item()) if self.radc_is_list else int(self.radc)
        return torch.round(current / adc_ref * (radc_val - 1)) / (radc_val - 1)

    def _sliced_quant_qmax(self, sd: SlicedDataMultiMode):
        bits = int(torch.sum(sd.slice_method).item())
        if self.mode == 2:
            return max(2 ** bits - 1, 1)
        return max(2 ** (bits - 1) - 1, 1)


    def _resolve_g_index_dtype(self):
        """
        Choose the smallest dtype that can store conductance level indices.
        
        Parameters:
            None.
        
        Returns:
            torch.dtype or None: Integer dtype for compressed weights, or None if g_level is too large.
        """
        if self.g_level <= 256:
            return torch.uint8
        if self.g_level <= 32768:
            return torch.int16
        if self.g_level <= 2 ** 31:
            return torch.int32
        return None

    def _should_compress_weight(self, mat: SlicedDataMultiMode):
        """
        Decide whether an inference weight can be stored as conductance indices.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor being prepared.
        
        Returns:
            bool: True when compressed index storage is safe.
        """
        return (
            self.mode in (0, 1, 2)
            and getattr(mat, "inference", False)
            and (self.write_variation == 0 or self._write_variation_is_virtual())
            and not self._has_drift
            and self._g_index_dtype is not None
        )

    def _prepare_weight_conductance(self, mat: SlicedDataMultiMode):
        """
        Prepare conductance tensors or compressed conductance indices for weights.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight sliced data object to populate.
        
        Returns:
            None. Updates mat.G or mat.G_indices in place.
        """
        mat.G = None
        mat.G_indices = None
        mat.G_index_dtype = None
        mat.G_is_compressed = False
        mat.mode1_w_max = None

        if self.mode == 1:
            if mat.quantized_data is None:
                raise ValueError("Mode 1 weight preparation requires quantized_data.")
            tile_in, tile_out = self._resolve_mode1_tile_size(
                mat,
                mat.quantized_data.shape[-2],
                mat.quantized_data.shape[-1],
            )
            w_max = self._build_mode1_tile_scale_grid(
                mat.quantized_data,
                tile_in,
                tile_out,
            )
            mat.mode1_w_max = w_max
            if self._should_compress_weight(mat):
                gp_idx, gn_idx = self._build_mode1_level_indices(
                    mat.quantized_data,
                    w_max,
                    tile_in,
                    tile_out,
                )
                mat.G_indices = (gp_idx, gn_idx)
                mat.G_index_dtype = gp_idx.dtype
                mat.G_is_compressed = True
            else:
                w_max_full = self._expand_mode1_tile_scales(
                    w_max,
                    mat.quantized_data.shape,
                    tile_in,
                    tile_out,
                )
                mat.G = self._num2G_mode1(mat.quantized_data, w_max_full)
            return

        max_weights = mat.sliced_max_weights.reshape(1, 1, -1, 1, 1)
        if self.mode == 2:
            if mat.sliced_data_p is None or mat.sliced_data_n is None:
                raise ValueError("Mode 2 weight preparation requires differential branches.")
            if self._should_compress_weight(mat):
                gp_idx = self._build_level_indices(mat.sliced_data_p, max_weights, seed_stuck=123)
                gn_idx = self._build_level_indices(mat.sliced_data_n, max_weights, seed_stuck=124)
                mat.G_indices = (gp_idx, gn_idx)
                mat.G_index_dtype = gp_idx.dtype
                mat.G_is_compressed = True
            else:
                mat.G = self._num2G_mode2(mat.sliced_data_p, mat.sliced_data_n, max_weights)
            return

        if self._should_compress_weight(mat):
            g_idx = self._build_level_indices(mat.sliced_data, max_weights, seed_stuck=123)
            mat.G_indices = g_idx
            mat.G_index_dtype = g_idx.dtype
            mat.G_is_compressed = True
        else:
            mat.G = self._num2G_mode0(mat.sliced_data, max_weights)

    def _build_mode1_tile_scale_grid(self, data, tile_in, tile_out):
        """
        Build a compact grid whose entries hold each Mode 1 hardware tile scale.

        Mode 1 maps each paral_size tile independently: all weights in the same
        tile share one scale before being projected onto the g_level conductance
        states.
        """
        if data.dim() != 2:
            raise ValueError("Mode 1 weight conductance preparation expects a 2-D matrix.")
        data_f = data.to(torch.float32)
        n_tiles_in = (data.shape[0] + tile_in - 1) // tile_in
        n_tiles_out = (data.shape[1] + tile_out - 1) // tile_out
        scale = torch.empty((n_tiles_in, n_tiles_out), device=data.device, dtype=torch.float32)
        padded_cols = n_tiles_out * tile_out
        for ri, r0 in enumerate(range(0, data.shape[0], tile_in)):
            r1 = min(r0 + tile_in, data.shape[0])
            abs_block = data_f[r0:r1, :].abs()
            if padded_cols != data.shape[1]:
                padded = torch.zeros(
                    (abs_block.shape[0], padded_cols),
                    device=data.device,
                    dtype=torch.float32,
                )
                padded[:, : data.shape[1]] = abs_block
                abs_block = padded
            row_scale = abs_block.reshape(abs_block.shape[0], n_tiles_out, tile_out).amax(dim=(0, 2))
            row_scale = torch.where(row_scale > 0, row_scale, torch.ones_like(row_scale))
            scale[ri] = row_scale
        return scale

    def _expand_mode1_tile_scales(self, scale_grid, shape, tile_in, tile_out):
        """Expand compact Mode 1 tile scales to a full matrix for legacy paths."""
        full = torch.empty(shape, device=scale_grid.device, dtype=torch.float32)
        for ri, r0 in enumerate(range(0, shape[0], tile_in)):
            r1 = min(r0 + tile_in, shape[0])
            for ci, c0 in enumerate(range(0, shape[1], tile_out)):
                c1 = min(c0 + tile_out, shape[1])
                full[r0:r1, c0:c1] = scale_grid[ri, ci]
        return full

    def _mode1_scale_cols(self, scale_grid, row_tile_idx, c0, c1, tile_out, out_features, device, dtype):
        """Return per-output-column Mode 1 scales for one input tile and output chunk."""
        vals = scale_grid[row_tile_idx, c0:c1].to(device=device, dtype=dtype)
        if vals.numel() == 0:
            return vals
        repeated = vals.repeat_interleave(tile_out)
        out_start = c0 * tile_out
        out_end = min(c1 * tile_out, out_features)
        return repeated[: max(0, out_end - out_start)]

    def _build_mode1_level_indices(self, data, scale_grid, tile_in, tile_out):
        """Build compressed positive/negative conductance level indices for Mode 1."""
        if data.dim() != 2:
            raise ValueError("Mode 1 compressed weight preparation expects a 2-D matrix.")
        gp_idx = torch.empty(data.shape, device=data.device, dtype=self._g_index_dtype)
        gn_idx = torch.empty_like(gp_idx)
        data_f = data.to(torch.float32)
        qmax = float(self.g_level - 1)
        out_features = int(data.shape[1])
        for ri, r0 in enumerate(range(0, data.shape[0], tile_in)):
            r1 = min(r0 + tile_in, data.shape[0])
            col_scale = scale_grid[ri].to(device=data.device, dtype=torch.float32).repeat_interleave(tile_out)
            col_scale = col_scale[:out_features].view(1, -1)
            lvl = torch.round(data_f[r0:r1, :] / col_scale * qmax)
            lvl.clamp_(min=-qmax, max=qmax)
            gp_idx[r0:r1, :] = lvl.clamp(min=0).to(self._g_index_dtype)
            lvl.neg_().clamp_(min=0)
            gn_idx[r0:r1, :] = lvl.to(self._g_index_dtype)
        self._apply_stuck_to_indices_inplace(gp_idx, 123)
        self._apply_stuck_to_indices_inplace(gn_idx, 124)
        return gp_idx, gn_idx

    def _build_mode1_tile_scales(self, data, tile_in, tile_out):
        """Compatibility wrapper returning full-size Mode 1 tile scales."""
        return self._expand_mode1_tile_scales(
            self._build_mode1_tile_scale_grid(data, tile_in, tile_out),
            data.shape,
            tile_in,
            tile_out,
        )

    def _map_to_level_work(self, data, max_weights):
        """
        Map sliced integer data to floating conductance level indices.
        
        Parameters:
            data (torch.Tensor): Sliced integer data.
            max_weights (torch.Tensor): Maximum representable value for each slice.
        
        Returns:
            torch.Tensor: Floating level index tensor clamped to valid conductance levels.
        """
        level_work = data.to(torch.float32)
        level_work.div_(max_weights.to(device=level_work.device, dtype=torch.float32))
        level_work.mul_(self.g_level - 1)
        level_work.round_()
        level_work.clamp_(0, self.g_level - 1)
        return level_work

    def _apply_write_noise_inplace(self, G, seed):
        """
        Apply fixed write variation to a conductance tensor in place.
        
        Parameters:
            G (torch.Tensor): Conductance tensor to modify.
            seed (int): Random seed used for deterministic write variation.
        
        Returns:
            None.
        """
        if self.write_variation <= 0:
            return
        gen = torch.Generator(device=G.device)
        gen.manual_seed(seed)
        max_noise_elems = 64 * 1024 * 1024
        if G.numel() <= max_noise_elems:
            noise = torch.randn(
                G.shape,
                generator=gen,
                device=G.device,
                dtype=G.dtype,
            )
            noise.mul_(self.write_variation).exp_()
            G.mul_(noise)
            del noise
            return

        flat = G.reshape(-1)
        for start in range(0, flat.numel(), max_noise_elems):
            end = min(start + max_noise_elems, flat.numel())
            chunk = flat[start:end]
            noise = torch.randn(
                chunk.shape,
                generator=gen,
                device=G.device,
                dtype=G.dtype,
            )
            noise.mul_(self.write_variation).exp_()
            chunk.mul_(noise)
            del noise

    def _derived_seed(self, seed, *parts):
        """Build a deterministic non-negative seed for virtual per-chunk noise."""
        value = int(seed) & 0x7FFFFFFF
        for part in parts:
            value = (value * 1103515245 + 12345 + int(part)) & 0x7FFFFFFF
        return value

    def _apply_virtual_write_noise_to_absolute(self, G_abs, seed, *seed_parts):
        """
        Apply deterministic per-chunk write variation to absolute conductance.

        Virtual write variation keeps only compact level indices in memory and
        regenerates the fixed programmed mismatch when a chunk is restored.
        """
        if not self._write_variation_is_virtual():
            return G_abs
        gen = torch.Generator(device=G_abs.device)
        gen.manual_seed(self._derived_seed(seed, *seed_parts))
        noise = torch.randn(
            G_abs.shape,
            generator=gen,
            device=G_abs.device,
            dtype=G_abs.dtype,
        )
        noise.mul_(self.write_variation).exp_()
        G_abs.mul_(noise)
        del noise
        return G_abs.clamp_(self.LGS, self.HGS)

    def _apply_drift_inplace(self, G):
        """
        Apply deterministic conductance drift in place.

        The model follows a compact relaxation law:
        G_eff = LGS + (G - LGS) * (t / t0)^(-nu).
        This preserves LGS as the lower bound and is disabled when nu is zero.
        """
        if not self._has_drift:
            return
        G.sub_(self.LGS)
        G.mul_(self._drift_scale)
        G.add_(self.LGS)

    def _apply_stuck_to_indices_inplace(self, level_work, seed):
        """
        Apply stuck-at faults to compressed conductance level indices in place.
        
        Parameters:
            level_work (torch.Tensor): Conductance level index tensor to modify.
            seed (int): Random seed used for deterministic stuck-at masks.
        
        Returns:
            None.
        """
        if self.rate_stuck_HGS <= 0 and self.rate_stuck_LGS <= 0:
            return
        gen = torch.Generator(device=level_work.device)
        gen.manual_seed(seed)
        rv = torch.rand(level_work.shape, generator=gen, device=level_work.device)
        if self.rate_stuck_HGS > 0:
            level_work[rv < self.rate_stuck_HGS] = self.g_level - 1
        if self.rate_stuck_LGS > 0:
            stuck_lgs = (rv >= self.rate_stuck_HGS) & (
                rv < self.rate_stuck_HGS + self.rate_stuck_LGS
            )
            level_work[stuck_lgs] = 0
        del rv

    def _apply_stuck_to_conductance_inplace(self, G, seed):
        """
        Apply stuck-at faults to a conductance tensor in place.
        
        Parameters:
            G (torch.Tensor): Conductance tensor to modify.
            seed (int): Random seed used for deterministic stuck-at masks.
        
        Returns:
            None.
        """
        if self.rate_stuck_HGS <= 0 and self.rate_stuck_LGS <= 0:
            return
        gen = torch.Generator(device=G.device)
        gen.manual_seed(seed)
        rv = torch.rand(G.shape, generator=gen, device=G.device)
        if self.rate_stuck_HGS > 0:
            G[rv < self.rate_stuck_HGS] = self.HGS
        if self.rate_stuck_LGS > 0:
            stuck_lgs = (rv >= self.rate_stuck_HGS) & (
                rv < self.rate_stuck_HGS + self.rate_stuck_LGS
            )
            G[stuck_lgs] = self.LGS
        del rv

    def _can_use_direct_binary_indices(self, data, max_weights):
        """
        Detect the common SLC case where sliced bits are already conductance indices.
        """
        if self.g_level != 2 or self._g_index_dtype != torch.uint8:
            return False
        if data.dtype != torch.uint8:
            return False
        return bool(torch.all(max_weights == 1).item())

    def _build_level_indices(self, data, max_weights, seed_stuck):
        """
        Build compressed conductance level indices for an inference weight branch.
        
        Parameters:
            data (torch.Tensor): Sliced integer weight branch.
            max_weights (torch.Tensor): Maximum representable value for each slice.
            seed_stuck (int): Random seed for stuck-at faults.
        
        Returns:
            torch.Tensor: Conductance level indices stored in the selected compact dtype.
        """
        if self._can_use_direct_binary_indices(data, max_weights):
            level_indices = data
            if self.rate_stuck_HGS > 0 or self.rate_stuck_LGS > 0:
                level_indices = level_indices.clone()
                self._apply_stuck_to_indices_inplace(level_indices, seed_stuck)
            return level_indices

        level_work = self._map_to_level_work(data, max_weights)
        self._apply_stuck_to_indices_inplace(level_work, seed_stuck)
        level_indices = level_work.to(self._g_index_dtype)
        del level_work
        return level_indices

    def _levels_to_conductance(self, level_work):
        """
        Convert conductance level indices to conductance values in place.
        
        Parameters:
            level_work (torch.Tensor): Floating conductance level indices.
        
        Returns:
            torch.Tensor: Conductance tensor after scaling by Q_G and adding LGS.
        """
        level_work.mul_(self.Q_G).add_(self.LGS)
        if level_work.dtype != self.conductance_dtype:
            level_work = level_work.to(self.conductance_dtype)
        return level_work

    def _build_branch_conductance(self, data, max_weights, seed_write, seed_stuck):
        """
        Build one conductance branch from sliced data.
        
        Parameters:
            data (torch.Tensor): Sliced integer branch data.
            max_weights (torch.Tensor): Maximum representable value for each slice.
            seed_write (int): Random seed for write variation.
            seed_stuck (int): Random seed for stuck-at faults.
        
        Returns:
            torch.Tensor: Conductance tensor clamped between LGS and HGS.
        """
        G = self._levels_to_conductance(self._map_to_level_work(data, max_weights))
        if not self._write_variation_is_virtual():
            self._apply_write_noise_inplace(G, seed_write)
        self._apply_drift_inplace(G)
        self._apply_stuck_to_conductance_inplace(G, seed_stuck)
        G.clamp_(self.LGS, self.HGS)
        return G

    def _num2G(self, data, max_weights):
        """
        Dispatch numeric weight data to the mode-specific conductance mapper.
        
        Parameters:
            data (torch.Tensor or tuple): Sliced weight data, or (positive, negative) branches for Mode 2.
            max_weights (torch.Tensor): Maximum representable value for each slice.
        
        Returns:
            torch.Tensor or tuple: Conductance tensor for Mode 0, or (Gp, Gn) for Mode 1/2.
        """
        if self.mode == 1:
            return self._num2G_mode1(data, max_weights)
        elif self.mode == 2:
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("Mode 2 _num2G expects a (data_p, data_n) tuple.")
            return self._num2G_mode2(data[0], data[1], max_weights)
        else:
            return self._num2G_mode0(data, max_weights)

    def _num2G_mode0(self, data, max_weights):
        """
        Map standard sliced weights to conductance for Mode 0.
        
        Parameters:
            data (torch.Tensor): Sliced integer weight data.
            max_weights (torch.Tensor): Maximum representable value for each slice.
        
        Returns:
            torch.Tensor: Conductance tensor.
        """
        return self._build_branch_conductance(
            data,
            max_weights,
            seed_write=42,
            seed_stuck=123,
        )

    def _num2G_mode1(self, data, max_weights):
        """
        Map quantized signed weights to tile-local linear differential-pair conductance for Mode 1.
        
        Parameters:
            data (torch.Tensor): Quantized signed weight matrix.
            max_weights (torch.Tensor): Absolute scale used to normalize the weight matrix.
        
        Returns:
            tuple: (Gp, Gn) positive and negative conductance branches.
        """
        data_f = data.to(torch.float32)
        max_weights = max_weights.to(device=data.device, dtype=torch.float32)
        if max_weights.dim() == 0:
            max_weights = max_weights.expand_as(data_f)
        safe_max = torch.where(max_weights > 0, max_weights, torch.ones_like(max_weights))
        lvl = torch.round(data_f / safe_max * (self.g_level - 1))
        lvl = torch.clamp(lvl, -(self.g_level - 1), self.g_level - 1)
        pos = lvl >= 0
        low = torch.full_like(lvl, self.LGS)
        Gp = torch.where(pos, lvl * self.Q_G + self.LGS, low)
        Gn = torch.where(~pos, torch.abs(lvl) * self.Q_G + self.LGS, low)
        Gp, Gn = self._apply_write_nonideals_pair(Gp, Gn, 42, 43, 123, 124)
        return (torch.clamp(Gp, self.LGS, self.HGS),
                torch.clamp(Gn, self.LGS, self.HGS))

    def _num2G_mode2(self, data_p, data_n, max_weights):
        """
        Map positive and negative sliced branches to conductance for Mode 2.
        
        Parameters:
            data_p (torch.Tensor): Positive sliced branch.
            data_n (torch.Tensor): Negative sliced branch.
            max_weights (torch.Tensor): Maximum representable value for each slice.
        
        Returns:
            tuple: (Gp, Gn) conductance branches.
        """
        Gp = self._levels_to_conductance(self._map_to_level_work(data_p, max_weights))
        Gn = self._levels_to_conductance(self._map_to_level_work(data_n, max_weights))
        Gp, Gn = self._apply_write_nonideals_pair(Gp, Gn, 42, 43, 123, 124)
        Gp.clamp_(self.LGS, self.HGS)
        Gn.clamp_(self.LGS, self.HGS)
        return (Gp, Gn)

    def _apply_write_nonideals_pair(self, Gp, Gn, seed_p, seed_n,
                                    seed_stuck_p, seed_stuck_n):
        """
        Apply write variation and stuck-at faults to a conductance pair.
        
        Parameters:
            Gp (torch.Tensor): Positive conductance branch.
            Gn (torch.Tensor): Negative conductance branch.
            seed_p (int): Random seed for positive branch write variation.
            seed_n (int): Random seed for negative branch write variation.
            seed_stuck_p (int): Random seed for positive branch stuck-at faults.
            seed_stuck_n (int): Random seed for negative branch stuck-at faults.
        
        Returns:
            tuple: (Gp, Gn) after non-ideal effects.
        """
        if self.write_variation > 0 and not self._write_variation_is_virtual():
            for G, seed in ((Gp, seed_p), (Gn, seed_n)):
                self._apply_write_noise_inplace(G, seed)
        if self._has_drift:
            self._apply_drift_inplace(Gp)
            self._apply_drift_inplace(Gn)
        if self.rate_stuck_HGS > 0 or self.rate_stuck_LGS > 0:
            for G, seed in ((Gp, seed_stuck_p), (Gn, seed_stuck_n)):
                self._apply_stuck_to_conductance_inplace(G, seed)
        return Gp, Gn


    def _gen_read_noise(self, mat: SlicedDataMultiMode):
        """
        Apply read variation to prepared conductance tensors.

        Parameters:
            mat (SlicedDataMultiMode): Weight tensor containing mat.G.
        
        Returns:
            torch.Tensor or tuple: Conductance tensor, or (Gp, Gn), with read noise applied when enabled.
        """
        if self.mode == 1:
            raise RuntimeError(
                "_gen_read_noise must not be called for Mode 1; "
                "read noise is applied inside _dot_mode1."
            )
        G = mat.G
        if self._has_read_noise:
            if isinstance(G, tuple):
                G = (
                    self._apply_read_noise_tensor(G[0]),
                    self._apply_read_noise_tensor(G[1]),
                )
            else:
                G = self._apply_read_noise_tensor(G)
        return G

    def _apply_read_noise_tensor(self, G: torch.Tensor) -> torch.Tensor:
        """
        Apply lognormal read noise to one conductance tensor.
        
        Parameters:
            G (torch.Tensor): Conductance tensor before read noise.
        
        Returns:
            torch.Tensor: Conductance tensor after read noise.
        """
        """Lognormal read noise with O(N) memory (shared helper)."""
        if self._rv_all_same:
            if self._rv_sigma > 0:
                noise = torch.randn_like(G)
                noise.mul_(self._rv_sigma).exp_()
                noise.mul_(G)
                G = noise
        else:
            # Per-level: compute level index by rounding (no g_level dimension)
            level_idx = torch.round((G - self.LGS) / self.Q_G).long().clamp(
                0, self.g_level - 1
            )
            var_t = self._read_variation_tensor.to(device=G.device, dtype=G.dtype)
            std_per_el = var_t[level_idx]
            del level_idx, var_t
            noise = torch.randn_like(G)
            noise.mul_(std_per_el).exp_()
            del std_per_el
            noise.mul_(G)
            G = noise
        return G


    def _num2V(self, x: SlicedDataMultiMode):
        """
        Convert sliced activation data to quantized input voltage.

        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
        
        Returns:
            torch.Tensor: Input voltage tensor after DAC quantization and voltage noise.
        """
        return self._num2V_from_sliced_data(x, x.sliced_data)

    def _num2V_from_sliced_data(self, x: SlicedDataMultiMode, sliced_data: torch.Tensor):
        """
        Convert one sliced activation branch to quantized input voltage.
        """
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:
            xmax = xmax.reshape(1, 1, -1, 1, 1)
        elif len(x.shape) == 3:
            xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
        else:
            raise ValueError("Input data must be 2-D or 3-D.")

        V_in = sliced_data.to(self.compute_dtype)
        V_in.div_(xmax)
        V_in.mul_(self.rdac - 1)
        V_in.round_()
        V_in.mul_(self.vread / (self.rdac - 1))

        if self.vnoise > 0:
            noise = torch.randn_like(V_in)
            noise.mul_(self.vnoise).add_(1.0)
            V_in.mul_(noise)
            del noise
        return V_in


    def _apply_read_noise_single(self, G: torch.Tensor) -> torch.Tensor:
        """
        Apply read variation to one Mode 1 conductance branch.
        
        Parameters:
            G (torch.Tensor): One conductance branch, either Gp or Gn.
        
        Returns:
            torch.Tensor: Conductance branch after read noise.
        """
        """Per-level or uniform lognormal noise for Mode 1 (Gp or Gn)."""
        return self._apply_read_noise_tensor(G)

    def _build_vin_for_slice(self, x_slice, xmax_i):
        """
        Build quantized voltage for one input slice in the inference path.
        
        Parameters:
            x_slice (torch.Tensor): One input slice block.
            xmax_i (torch.Tensor): Maximum representable value for this input slice.
        
        Returns:
            torch.Tensor: Quantized voltage tensor for the slice.
        """
        Vin_i = x_slice.to(self.compute_dtype)
        Vin_i.div_(xmax_i)
        Vin_i.mul_(self.rdac - 1)
        Vin_i.round_()
        Vin_i.mul_(self.vread / (self.rdac - 1))
        if self.vnoise > 0:
            noise = torch.randn_like(Vin_i)
            noise.mul_(self.vnoise).add_(1.0)
            Vin_i.mul_(noise)
            del noise
        return Vin_i

    def _restore_shifted_from_indices(self, idx_chunk, seed_write=42, c0=0, branch=0):
        """
        Restore shifted conductance from compressed level indices.
        
        Parameters:
            idx_chunk (torch.Tensor): Compressed conductance level indices for one output chunk.
        
        Returns:
            torch.Tensor: Conductance values with LGS removed, optionally including read noise.
        """
        if (
            self._has_read_noise
            and idx_chunk.is_cuda
            and self._rv_all_same
            and self._rv_sigma > 0
            and not self._write_variation_is_virtual()
            and self.triton_gidx_fused_restore_read_noise
        ):
            try:
                from .triton_fast_accumulate import triton_restore_gidx_read_noise
                noise_offset_base = self._read_noise_restore_counter * idx_chunk.numel()
                self._read_noise_restore_counter += 1
                return triton_restore_gidx_read_noise(
                    idx_chunk,
                    lgs=self.LGS,
                    q_g=self.Q_G,
                    read_sigma=float(self._rv_sigma),
                    dtype=self.compute_dtype,
                    noise_seed=self._read_noise_seed_base,
                    noise_offset_base=noise_offset_base,
                )
            except Exception:
                pass

        G_chunk = idx_chunk.to(self.compute_dtype)
        G_chunk.mul_(self.Q_G)
        if self._write_variation_is_virtual() or self._has_read_noise:
            G_chunk.add_(self.LGS)
        if self._write_variation_is_virtual():
            G_chunk = self._apply_virtual_write_noise_to_absolute(
                G_chunk,
                seed_write,
                branch,
                c0,
                *tuple(int(v) for v in idx_chunk.shape),
            )
        if self._has_read_noise:
            G_chunk = self._apply_read_noise_tensor(G_chunk)
        if self._write_variation_is_virtual() or self._has_read_noise:
            G_chunk.sub_(self.LGS)
        return G_chunk

    def _restore_shifted_from_conductance(self, G_chunk):
        """
        Restore shifted conductance from stored conductance values.
        
        Parameters:
            G_chunk (torch.Tensor): Conductance tensor for one output chunk.
        
        Returns:
            torch.Tensor: Conductance values with LGS removed, optionally including read noise.
        """
        if G_chunk.dtype != self.compute_dtype:
            G_chunk = G_chunk.to(self.compute_dtype)
        if self._has_read_noise:
            G_chunk = self._apply_read_noise_tensor(G_chunk)
            G_chunk.sub_(self.LGS)
            return G_chunk
        return G_chunk - self.LGS

    def _get_mode0_shifted_chunk(self, mat: SlicedDataMultiMode, c0: int, c1: int):
        """
        Fetch one shifted Mode 0 weight conductance chunk.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor containing G or compressed G_indices.
            c0 (int): Inclusive output tile-column start index.
            c1 (int): Exclusive output tile-column end index.
        
        Returns:
            torch.Tensor: Shifted conductance chunk for Mode 0.
        """
        if mat.G_is_compressed:
            return self._restore_shifted_from_indices(
                mat.G_indices[:, c0:c1, :, :, :],
                seed_write=42,
                c0=c0,
                branch=0,
            )
        if mat.G is None:
            raise ValueError("Mode 0 weight conductance is missing.")
        return self._restore_shifted_from_conductance(mat.G[:, c0:c1, :, :, :])

    def _get_mode2_shifted_chunk(self, mat: SlicedDataMultiMode, c0: int, c1: int):
        """
        Fetch one shifted Mode 2 positive/negative conductance chunk.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor containing G branches or compressed G_indices branches.
            c0 (int): Inclusive output tile-column start index.
            c1 (int): Exclusive output tile-column end index.
        
        Returns:
            tuple: (Gp_shifted, Gn_shifted) conductance chunks with LGS removed.
        """
        if mat.G_is_compressed:
            gp_idx, gn_idx = mat.G_indices
            return (
                self._restore_shifted_from_indices(
                    gp_idx[:, c0:c1, :, :, :],
                    seed_write=42,
                    c0=c0,
                    branch=0,
                ),
                self._restore_shifted_from_indices(
                    gn_idx[:, c0:c1, :, :, :],
                    seed_write=43,
                    c0=c0,
                    branch=1,
                ),
            )
        if mat.G is None:
            raise ValueError("Mode 2 weight conductance is missing.")
        Gp, Gn = mat.G
        return (
            self._restore_shifted_from_conductance(Gp[:, c0:c1, :, :, :]),
            self._restore_shifted_from_conductance(Gn[:, c0:c1, :, :, :]),
        )

    def _try_get_mode2_gdiff_shifted_chunk(self, mat: SlicedDataMultiMode, c0: int, c1: int):
        if not (
            mat.G_is_compressed
            and isinstance(getattr(mat, "G_indices", None), tuple)
            and self._rv_all_same
            and not self._write_variation_is_virtual()
            and self.triton_gidx_fused_restore_read_noise
        ):
            return None
        gp_idx, gn_idx = mat.G_indices
        gp_chunk = gp_idx[:, c0:c1, :, :, :]
        gn_chunk = gn_idx[:, c0:c1, :, :, :]
        if not gp_chunk.is_cuda or gp_chunk.shape != gn_chunk.shape:
            return None
        try:
            from .triton_fast_accumulate import triton_restore_mode2_gdiff_gidx_read_noise

            read_sigma = float(self._rv_sigma) if self._has_read_noise else 0.0
            noise_offset_base = self._read_noise_restore_counter * gp_chunk.numel() * 2
            self._read_noise_restore_counter += 1
            return triton_restore_mode2_gdiff_gidx_read_noise(
                gp_chunk,
                gn_chunk,
                lgs=self.LGS,
                q_g=self.Q_G,
                read_sigma=read_sigma,
                dtype=self.compute_dtype,
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
            )
        except Exception:
            return None

    def _get_mode1_shifted_chunk(
        self,
        mat: SlicedDataMultiMode,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        branch_base: int = 0,
    ):
        """
        Fetch one shifted Mode 1 positive/negative conductance matrix chunk.

        Mode 1 stores a single differential-pair level per weight rather than
        a 5-D bit-slice tensor, so c0/c1 are tile-column indices and r0/r1 are
        raw input-feature bounds.
        """
        tile_out = self._mode1_tile_out(mat)
        out_start = c0 * tile_out
        out_end = min(c1 * tile_out, int(mat.shape[1]))
        if mat.G_is_compressed:
            gp_idx, gn_idx = mat.G_indices
            return (
                self._restore_shifted_from_indices(
                    gp_idx[r0:r1, out_start:out_end],
                    seed_write=42,
                    c0=out_start,
                    branch=branch_base,
                ),
                self._restore_shifted_from_indices(
                    gn_idx[r0:r1, out_start:out_end],
                    seed_write=43,
                    c0=out_start,
                    branch=branch_base + 1,
                ),
            )
        if mat.G is None:
            raise ValueError("Mode 1 weight conductance is missing.")
        Gp, Gn = mat.G
        return (
            self._restore_shifted_from_conductance(Gp[r0:r1, out_start:out_end]),
            self._restore_shifted_from_conductance(Gn[r0:r1, out_start:out_end]),
        )

    def _mode1_tile_out(self, mat):
        _, tile_out = self._resolve_mode1_tile_size(
            mat,
            int(mat.shape[0]),
            int(mat.shape[1]),
        )
        return tile_out

    def _mode2_diff_block_r(self) -> int:
        cap = max(1, int(getattr(self, "triton_mode2_diff_block_r_cap", 16)))
        return min(int(self.triton_block_r), cap)

    def _mode2_diff_block_l(self) -> int:
        cap = max(1, int(getattr(self, "triton_mode2_diff_block_l_cap", 8)))
        return min(int(self.triton_block_l), cap)

    def _iter_output_chunks(self, mat: SlicedDataMultiMode):
        """
        Iterate output tile-column chunks for memory-efficient inference.
        
        Parameters:
            mat (SlicedDataMultiMode): Weight tensor whose output tile dimension is chunked.
        
        Returns:
            iterator: Pairs of (c0, c1) output tile-column bounds.
        """
        if self.mode == 1:
            if mat.mode1_w_max is None:
                raise ValueError("Mode 1 inference requires mode1_w_max.")
            ndc_y = int(mat.mode1_w_max.shape[1])
        else:
            ndc_y = mat.max_data.shape[1] if mat.max_data is not None else mat.e_bias.shape[1]
        chunk_ndc = ndc_y
        if self.inference_chunk_size is not None:
            chunk_ndc = max(1, min(int(self.inference_chunk_size), ndc_y))
        if self.fast_inference_backend in ("triton", "triton_gidx"):
            # Very wide LLM heads can make the flattened Triton pointer offsets
            # exceed the 32-bit range when all output tile columns are launched
            # at once. Keep compressed-index launches bounded; the surrounding
            # code concatenates chunks back into the same logical output.
            g_source = None
            if self.mode in (1, 2) and isinstance(getattr(mat, "G_indices", None), tuple):
                g_source = mat.G_indices[0]
            elif getattr(mat, "G_indices", None) is not None:
                g_source = mat.G_indices
            elif self.mode == 2 and isinstance(getattr(mat, "G", None), tuple):
                g_source = mat.G[0]
            elif getattr(mat, "G", None) is not None:
                g_source = mat.G
            if g_source is not None and len(g_source.shape) == 5:
                m_dim, _, s_dim, k_dim, l_dim = [max(1, int(v)) for v in g_source.shape]
                per_m_tile = max(1, s_dim * k_dim * l_dim)
                max_by_offset = max(1, 2_000_000_000 // max(1, m_dim * per_m_tile))
                chunk_limit = self._mode0_triton_chunk_limit(ndc_y, mat=mat)
                chunk_ndc = max(1, min(chunk_ndc, ndc_y, max_by_offset, chunk_limit))
            elif self.mode == 1:
                chunk_limit = self._mode1_triton_chunk_limit(ndc_y, mat=mat)
                chunk_ndc = max(1, min(chunk_ndc, ndc_y, chunk_limit))
        for c0 in range(0, ndc_y, chunk_ndc):
            yield c0, min(c0 + chunk_ndc, ndc_y)

    def _can_direct_write_output_chunks_2d(self, x, mat, chunks) -> bool:
        return (
            self.direct_output_chunk_write
            and
            len(x.shape) == 2
            and bool(chunks)
            and mat.shape is not None
            and (getattr(mat, "max_data", None) is not None or self.mode == 1)
            and int(mat.shape[1]) > 0
        )

    def _alloc_direct_output_2d(self, x, mat, dtype):
        rows = int(x.shape[0])
        cols = int(mat.shape[1])
        device = x.device
        return torch.empty((rows, cols), device=device, dtype=dtype)

    def _direct_output_col_range_2d(self, mat, c0, c1):
        tile_cols = int(mat.paral_size[1])
        out_start = int(c0) * tile_cols
        out_end = min(int(c1) * tile_cols, int(mat.shape[1]))
        return out_start, out_end

    def _zero_direct_output_chunk_2d(self, output, mat, c0, c1):
        out_start, out_end = self._direct_output_col_range_2d(mat, c0, c1)
        if out_start < out_end:
            output[:, out_start:out_end].zero_()

    def _write_direct_output_chunk_2d(self, output, chunk, mat, c0, c1):
        if chunk is None:
            return
        out_start, out_end = self._direct_output_col_range_2d(mat, c0, c1)
        if out_start >= out_end:
            return
        width = out_end - out_start
        output[:, out_start:out_end].copy_(chunk[:, :width])

    def _finalize_inference_chunk_2d(self, out, x, mat, c0, c1):
        """
        Apply scaling and reshape one 2-D inference output chunk.
        
        Parameters:
            out (torch.Tensor): Accumulated ADC-normalized chunk output.
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            c0 (int): Inclusive output tile-column start index.
            c1 (int): Exclusive output tile-column end index.
        
        Returns:
            torch.Tensor: Finalized 2-D output chunk in matrix layout.
        """
        fused = self._triton_finalize_inference_chunk_2d(out, x, mat, c0, c1)
        if fused is not None:
            return fused
        if x.bw_e is None:
            mat_max_chunk = mat.max_data[:, c0:c1, :, :]
            bm = torch.einsum("nmij, mpij->nmpij", x.max_data, mat_max_chunk)
            x_qmax = self._sliced_quant_qmax(x)
            mat_qmax = self._sliced_quant_qmax(mat)
            out = (
                out * bm
                / x_qmax
                / mat_qmax
            )
            del bm
        else:
            mat_ebias_chunk = mat.e_bias[:, c0:c1, :, :]
            eb = torch.einsum(
                "nmij, mpij->nmpij",
                2.0 ** x.e_bias,
                2.0 ** mat_ebias_chunk,
            )
            out = out * eb * 2.0 ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            del eb

        out = out.sum(dim=1).permute(0, 2, 1, 3)
        return out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])

    def _can_use_triton_output_finalize(self, out, x, mat) -> bool:
        return (
            bool(getattr(self, "triton_fuse_output_finalize", False))
            and self.fast_inference_backend in ("triton", "triton_gidx")
            and self.mode in (0, 2)
            and x.bw_e is None
            and len(x.shape) == 2
            and out is not None
            and out.dim() == 5
            and out.is_cuda
            and getattr(x, "max_data", None) is not None
            and getattr(mat, "max_data", None) is not None
            and x.max_data.is_cuda
            and mat.max_data.is_cuda
        )

    def _triton_finalize_inference_chunk_2d(self, out, x, mat, c0, c1):
        if not self._can_use_triton_output_finalize(out, x, mat):
            return None
        self._fastpath_count("output_finalize_fused_attempt_count")
        try:
            from .triton_fast_accumulate import triton_finalize_2d_tile_reduce
        except Exception as exc:
            self._fastpath_count("output_finalize_fused_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_output_finalize_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_output_finalize")
        try:
            result = triton_finalize_2d_tile_reduce(
                out,
                x.max_data,
                mat.max_data[:, c0:c1, :, :],
                x_qmax=float(self._sliced_quant_qmax(x)),
                mat_qmax=float(self._sliced_quant_qmax(mat)),
                block_r=max(1, min(int(self.triton_block_r), 64)),
                block_c=max(1, min(int(self.triton_block_l) * 2, 128)),
            )
        except Exception as exc:
            self._fastpath_count("output_finalize_fused_fallback_count")
            self._profile_stop(
                token,
                backend="triton_output_finalize",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("output_finalize_fused_success_count")
        self._profile_stop(token, backend="triton_output_finalize", status="ok")
        return result

    def _finalize_inference_chunk_3d(self, out, x, mat, c0, c1):
        """
        Apply scaling and reshape one batched inference output chunk.
        
        Parameters:
            out (torch.Tensor): Accumulated ADC-normalized chunk output.
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            c0 (int): Inclusive output tile-column start index.
            c1 (int): Exclusive output tile-column end index.
        
        Returns:
            torch.Tensor: Finalized batched output chunk in matrix layout.
        """
        if x.bw_e is None:
            mat_max_chunk = mat.max_data[:, c0:c1, :, :]
            bm = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat_max_chunk)
            x_qmax = self._sliced_quant_qmax(x)
            mat_qmax = self._sliced_quant_qmax(mat)
            out = (
                out * bm
                / x_qmax
                / mat_qmax
            )
            del bm
        else:
            mat_ebias_chunk = mat.e_bias[:, c0:c1, :, :]
            eb = torch.einsum(
                "bnmij, mpij->bnmpij",
                2.0 ** x.e_bias,
                2.0 ** mat_ebias_chunk,
            )
            out = out * eb * 2.0 ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            del eb

        out = out.sum(dim=2).permute(0, 1, 3, 2, 4)
        return out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])

    def _can_use_fast_inference(self, differential_input: bool) -> bool:
        """
        Decide whether the fused weight-slice inference path is legal.
        
        Parameters:
            differential_input (bool): Whether Mode 2 uses two input read phases.
        
        Returns:
            bool: True when all weight slices can share one fused ADC operation.
        """
        return (
            self.fast_inference
            and not self.radc_is_list
            and self.mode in (0, 2)
        )

    def _can_use_triton_fast_accumulate(self, vin) -> bool:
        """
        Decide whether the optional Triton fast-accumulate kernel should be tried.
        
        Parameters:
            vin (torch.Tensor): Quantized input voltage for one input slice.
        
        Returns:
            bool: True when the experimental Triton 2-D Linear kernel is applicable.
        """
        return (
            self.fast_inference_backend in ("triton", "triton_gidx")
            and len(vin.shape) == 4
            and vin.is_cuda
            and self.mode in (0, 2)
            and not self.radc_is_list
        )

    def _can_use_triton_gidx_accumulate(self, vin, mat) -> bool:
        """
        Decide whether the compressed G-index Triton kernel should be tried.

        The current kernel is deterministic and therefore only legal when read
        noise is disabled; otherwise restoring conductance in Python applies
        per-access noise that the index kernel cannot reproduce.
        """
        return (
            self.fast_inference_backend == "triton_gidx"
            and len(vin.shape) == 4
            and vin.is_cuda
            and self.mode in (0, 2)
            and not self.radc_is_list
            and getattr(mat, "G_is_compressed", False)
            and not self._has_read_noise
            and not self._write_variation_is_virtual()
        )

    def _can_use_triton_gidx_read_noise_accumulate(self, vin, mat) -> bool:
        return (
            self.fast_inference_backend == "triton_gidx"
            and len(vin.shape) == 4
            and vin.is_cuda
            and self.mode == 0
            and not self.radc_is_list
            and getattr(mat, "G_is_compressed", False)
            and self._has_read_noise
            and self._rv_all_same
            and self._rv_sigma > 0
            and self.triton_gidx_read_noise
            and not self._write_variation_is_virtual()
        )

    def _can_use_triton_gidx_input_slice_fusion(self, x, mat) -> bool:
        return (
            self.fast_inference_backend == "triton_gidx"
            and self.triton_gidx_fuse_input_slices
            and self.mode == 0
            and not self.radc_is_list
            and self.rdac >= 2
            and len(x.shape) == 2
            and getattr(x, "sliced_data", None) is not None
            and x.sliced_data.is_cuda
            and getattr(mat, "G_is_compressed", False)
            and getattr(mat, "G_indices", None) is not None
            and not self._has_read_noise
            and not self._write_variation_is_virtual()
            and self.vnoise == 0
        )

    def _can_use_triton_gidx_read_noise_input_slice_fusion(self, x, mat) -> bool:
        return (
            self.fast_inference_backend == "triton_gidx"
            and self.triton_gidx_fuse_input_slices
            and self.triton_gidx_read_noise
            and self.mode == 0
            and not self.radc_is_list
            and self.rdac >= 2
            and len(x.shape) == 2
            and getattr(x, "sliced_data", None) is not None
            and x.sliced_data.is_cuda
            and getattr(mat, "G_is_compressed", False)
            and getattr(mat, "G_indices", None) is not None
            and self._has_read_noise
            and self._rv_all_same
            and self._rv_sigma > 0
            and not self._write_variation_is_virtual()
            and self.vnoise == 0
        )

    def _can_use_triton_restored_input_slice_fusion(self, x, mat) -> bool:
        return (
            self.fast_inference_backend in ("triton", "triton_gidx")
            and self.triton_fuse_restored_input_slices
            and self.mode == 0
            and not self.radc_is_list
            and self.rdac >= 2
            and len(x.shape) == 2
            and getattr(x, "sliced_data", None) is not None
            and x.sliced_data.is_cuda
            and self.vnoise == 0
        )

    def _triton_gidx_input_slice_fused_accumulate(self, x, mat, c0, c1, slice_scale, adc_ref):
        if not self._can_use_triton_gidx_input_slice_fusion(x, mat):
            return None
        self._fastpath_count("gidx_input_slice_fused_attempt_count")
        try:
            from .triton_fast_accumulate import triton_gidx_accumulate_2d_input_slices
        except Exception as exc:
            self._fastpath_count("gidx_import_fallback_count")
            self._fastpath_count("gidx_input_slice_fused_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_gidx_input_slice_fused_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gidx = mat.G_indices[:, c0:c1, :, :, :]
        token = self._profile_start("triton_gidx_input_slice_fused_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(x, mat, gidx=gidx)
            out = triton_gidx_accumulate_2d_input_slices(
                x.sliced_data,
                gidx,
                x.sliced_max_weights,
                slice_scale,
                self.Q_G,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._fastpath_count("gidx_exception_fallback_count")
            self._fastpath_count("gidx_input_slice_fused_fallback_count")
            self._profile_stop(token, backend="triton_gidx_input_slice_fused", status="fallback", reason=type(exc).__name__, message=str(exc))
            return None
        self._fastpath_count("gidx_input_slice_fused_success_count")
        self._profile_stop(token, backend="triton_gidx_input_slice_fused", status="ok", shape_key=block_shape_key)
        return out

    def _triton_gidx_read_noise_input_slice_fused_accumulate(self, x, mat, c0, c1, slice_scale, adc_ref):
        if not self._can_use_triton_gidx_read_noise_input_slice_fusion(x, mat):
            return None
        self._fastpath_count("gidx_read_noise_input_slice_fused_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_gidx_accumulate_2d_input_slices_read_noise,
                triton_gidx_accumulate_2d_input_slices_read_noise_reuse_v,
            )
        except Exception as exc:
            self._fastpath_count("gidx_import_fallback_count")
            self._fastpath_count("gidx_read_noise_input_slice_fused_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_gidx_read_noise_input_slice_fused_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gidx = mat.G_indices[:, c0:c1, :, :, :]
        token = self._profile_start("triton_gidx_read_noise_input_slice_fused_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(x, mat, gidx=gidx)
            noise_offset_base = self._read_noise_restore_counter * gidx.numel()
            self._read_noise_restore_counter += 1
            runner = (
                triton_gidx_accumulate_2d_input_slices_read_noise_reuse_v
                if self.triton_reuse_input_voltage
                else triton_gidx_accumulate_2d_input_slices_read_noise
            )
            out = runner(
                x.sliced_data,
                gidx,
                x.sliced_max_weights,
                slice_scale,
                self.LGS,
                self.Q_G,
                float(self._rv_sigma),
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._fastpath_count("gidx_exception_fallback_count")
            self._fastpath_count("gidx_read_noise_input_slice_fused_fallback_count")
            self._profile_stop(
                token,
                backend="triton_gidx_read_noise_input_slice_fused",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("gidx_read_noise_input_slice_fused_success_count")
        backend = (
            "triton_gidx_read_noise_input_slice_fused_reuse_v"
            if self.triton_reuse_input_voltage
            else "triton_gidx_read_noise_input_slice_fused"
        )
        self._profile_stop(token, backend=backend, status="ok", shape_key=block_shape_key)
        return out

    def _triton_restored_input_slice_fused_accumulate(self, x, mat, c0, c1, g_shifted, slice_scale, adc_ref):
        if not self._can_use_triton_restored_input_slice_fusion(x, mat):
            return None
        self._fastpath_count("restored_input_slice_fused_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_fast_accumulate_2d_input_slices,
                triton_fast_accumulate_2d_input_slices_reuse_v,
            )
        except Exception as exc:
            self._fastpath_count("restored_input_slice_fused_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_restored_input_slice_fused_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_restored_input_slice_fused_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(x, mat, g=g_shifted)
            runner = (
                triton_fast_accumulate_2d_input_slices_reuse_v
                if self.triton_reuse_input_voltage
                else triton_fast_accumulate_2d_input_slices
            )
            out = runner(
                x.sliced_data,
                g_shifted,
                x.sliced_max_weights,
                slice_scale,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._fastpath_count("restored_input_slice_fused_fallback_count")
            self._profile_stop(token, backend="triton_restored_input_slice_fused", status="fallback", reason=type(exc).__name__, message=str(exc))
            return None
        self._fastpath_count("restored_input_slice_fused_success_count")
        backend = (
            "triton_restored_input_slice_fused_reuse_v"
            if self.triton_reuse_input_voltage
            else "triton_restored_input_slice_fused"
        )
        self._profile_stop(token, backend=backend, status="ok", shape_key=block_shape_key)
        return out

    def _triton_restored_direct_final_output(
        self,
        x,
        mat,
        c0,
        c1,
        g_shifted,
        slice_scale,
        adc_ref,
        out_buffer=None,
    ):
        if not (
            bool(getattr(self, "triton_direct_final_output", False))
            and self._can_use_triton_restored_input_slice_fusion(x, mat)
            and getattr(x, "max_data", None) is not None
            and getattr(mat, "max_data", None) is not None
            and x.max_data.is_cuda
            and mat.max_data.is_cuda
            and x.bw_e is None
        ):
            return None
        self._fastpath_count("direct_final_output_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_fast_accumulate_2d_input_slices_direct_final,
            )
        except Exception as exc:
            self._fastpath_count("direct_final_output_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_direct_final_output_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_direct_final_output")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(x, mat, g=g_shifted)
            out = triton_fast_accumulate_2d_input_slices_direct_final(
                x.sliced_data,
                g_shifted,
                x.sliced_max_weights,
                slice_scale,
                x.max_data,
                mat.max_data[:, c0:c1, :, :],
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                x_qmax=float(self._sliced_quant_qmax(x)),
                mat_qmax=float(self._sliced_quant_qmax(mat)),
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
                out=out_buffer,
                out_col_offset=self._direct_output_col_range_2d(mat, c0, c1)[0] if out_buffer is not None else 0,
                out_cols=int(mat.shape[1]) if out_buffer is not None else None,
            )
        except Exception as exc:
            self._fastpath_count("direct_final_output_fallback_count")
            self._profile_stop(
                token,
                backend="triton_direct_final_output",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("direct_final_output_success_count")
        if out_buffer is not None:
            self._fastpath_count("direct_final_output_store_success_count")
        self._profile_stop(
            token,
            backend="triton_direct_final_output",
            status="ok",
            direct_store=out_buffer is not None,
            shape_key=block_shape_key,
        )
        return out

    def _triton_gidx_direct_final_output(self, x, mat, c0, c1, slice_scale, adc_ref, out_buffer=None):
        if not (
            bool(getattr(self, "triton_direct_final_output", False))
            and bool(getattr(self, "triton_gidx_direct_final_output", True))
            and self.fast_inference_backend in ("triton", "triton_gidx")
            and self.mode == 0
            and not self.radc_is_list
            and self.rdac >= 2
            and len(x.shape) == 2
            and getattr(x, "sliced_data", None) is not None
            and getattr(mat, "G_indices", None) is not None
            and getattr(x, "max_data", None) is not None
            and getattr(mat, "max_data", None) is not None
            and x.sliced_data.is_cuda
            and x.max_data.is_cuda
            and mat.max_data.is_cuda
            and x.bw_e is None
            and self.vnoise == 0
            and not self._has_read_noise
            and not self._write_variation_is_virtual()
        ):
            return None
        self._fastpath_count("gidx_direct_final_output_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_gidx_accumulate_2d_input_slices_direct_final,
            )
        except Exception as exc:
            self._fastpath_count("gidx_direct_final_output_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_gidx_direct_final_output_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gidx = mat.G_indices[:, c0:c1, :, :, :]
        token = self._profile_start("triton_gidx_direct_final_output")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(x, mat, gidx=gidx)
            noise_offset_base = 0
            read_sigma = 0.0
            if self._has_read_noise and self._rv_sigma > 0:
                noise_offset_base = self._read_noise_restore_counter * gidx.numel()
                self._read_noise_restore_counter += 1
                read_sigma = float(self._rv_sigma)
            out = triton_gidx_accumulate_2d_input_slices_direct_final(
                x.sliced_data,
                gidx,
                x.sliced_max_weights,
                slice_scale,
                x.max_data,
                mat.max_data[:, c0:c1, :, :],
                self.LGS,
                self.Q_G,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                x_qmax=float(self._sliced_quant_qmax(x)),
                mat_qmax=float(self._sliced_quant_qmax(mat)),
                read_sigma=read_sigma,
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
                out=out_buffer,
                out_col_offset=self._direct_output_col_range_2d(mat, c0, c1)[0] if out_buffer is not None else 0,
                out_cols=int(mat.shape[1]) if out_buffer is not None else None,
            )
        except Exception as exc:
            self._fastpath_count("gidx_direct_final_output_fallback_count")
            self._profile_stop(
                token,
                backend="triton_gidx_direct_final_output",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("gidx_direct_final_output_success_count")
        if out_buffer is not None:
            self._fastpath_count("gidx_direct_final_output_store_success_count")
        self._profile_stop(
            token,
            backend="triton_gidx_direct_final_output",
            status="ok",
            direct_store=out_buffer is not None,
            shape_key=block_shape_key,
        )
        return out

    def _triton_gidx_read_noise_weight_slice_accumulate(
        self,
        vin,
        mat,
        c0,
        c1,
        slice_scale_row,
        adc_ref,
    ):
        if not self._can_use_triton_gidx_read_noise_accumulate(vin, mat):
            return None
        self._fastpath_count("gidx_read_noise_aware_attempt_count")
        try:
            from .triton_fast_accumulate import triton_gidx_accumulate_2d_read_noise
        except Exception as exc:
            self._fastpath_count("gidx_import_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_gidx_read_noise_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gidx = mat.G_indices[:, c0:c1, :, :, :]
        token = self._profile_start("triton_gidx_read_noise_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(mat=mat, gidx=gidx, vin=vin)
            noise_offset_base = self._read_noise_restore_counter * gidx.numel()
            self._read_noise_restore_counter += 1
            out = triton_gidx_accumulate_2d_read_noise(
                vin,
                gidx,
                slice_scale_row,
                self.LGS,
                self.Q_G,
                float(self._rv_sigma),
                adc_ref,
                int(self.radc),
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._fastpath_count("gidx_exception_fallback_count")
            self._profile_stop(token, backend="triton_gidx_read_noise", status="fallback", reason=type(exc).__name__, message=str(exc))
            return None
        self._fastpath_count("gidx_read_noise_aware_success_count")
        self._profile_stop(token, backend="triton_gidx_read_noise", status="ok", shape_key=block_shape_key)
        return out

    def _triton_gidx_weight_slice_accumulate(
        self,
        vin,
        mat,
        c0,
        c1,
        slice_scale_row,
        adc_ref,
    ):
        """
        Try the compressed G-index Triton fast path without restoring conductance first.
        """
        self._fastpath_count("gidx_attempt_count")
        if not self._can_use_triton_gidx_accumulate(vin, mat):
            reason = self._gidx_fallback_reason(vin, mat)
            if reason == "read_noise":
                out = self._triton_gidx_read_noise_weight_slice_accumulate(
                    vin,
                    mat,
                    c0,
                    c1,
                    slice_scale_row,
                    adc_ref,
                )
                if out is not None:
                    self._fastpath_count("gidx_success_count")
                    return out
            self._record_gidx_fallback(reason)
            return None
        try:
            from .triton_fast_accumulate import triton_gidx_accumulate_2d
        except Exception as exc:
            self._fastpath_count("gidx_import_fallback_count")
            self._record_gidx_fallback("import_error")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_gidx_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        if self.mode == 2:
            gp_idx, gn_idx = mat.G_indices
            gidx = None
            gp_chunk = gp_idx[:, c0:c1, :, :, :]
            gn_chunk = gn_idx[:, c0:c1, :, :, :]
        else:
            gidx = mat.G_indices[:, c0:c1, :, :, :]
            gp_chunk = None
            gn_chunk = None

        token = self._profile_start("triton_gidx_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(mat=mat, gidx=(gp_chunk if gp_chunk is not None else gidx), vin=vin)
            out = triton_gidx_accumulate_2d(
                vin,
                gidx,
                slice_scale_row,
                self.Q_G,
                adc_ref,
                int(self.radc),
                self.mode,
                gp_idx=gp_chunk,
                gn_idx=gn_chunk,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._fastpath_count("gidx_exception_fallback_count")
            self._record_gidx_fallback(type(exc).__name__)
            self._profile_stop(token, backend="triton_gidx", status="fallback", reason=type(exc).__name__, message=str(exc))
            return None
        self._fastpath_count("gidx_success_count")
        self._profile_stop(token, backend="triton_gidx", status="ok", shape_key=block_shape_key)
        return out

    def _triton_fast_weight_slice_accumulate(
        self,
        vin,
        g_shifted,
        slice_scale_row,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
    ):
        """
        Try the optional Triton fast-accumulate kernel.
        
        Parameters:
            vin (torch.Tensor): Quantized input voltage for one input slice.
            g_shifted (torch.Tensor or None): Mode 0 shifted conductance chunk.
            slice_scale_row (torch.Tensor): Per-weight-slice reconstruction scale for this input slice.
            adc_ref (float): ADC reference current.
            gp_shifted (torch.Tensor or None): Mode 2 positive branch conductance chunk.
            gn_shifted (torch.Tensor or None): Mode 2 negative branch conductance chunk.
        
        Returns:
            torch.Tensor or None: Triton output, or None if the optional kernel is unavailable.
        """
        if not self._can_use_triton_fast_accumulate(vin):
            return None
        try:
            from .triton_fast_accumulate import triton_fast_accumulate_2d
        except Exception as exc:
            if self.profile:
                self.profile_events.append({
                    "label": "triton_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_fast_accumulate")
        try:
            block_r, block_l, block_k, block_shape_key = self._mode0_triton_blocks(mat=None, g=g_shifted if g_shifted is not None else gp_shifted, vin=vin)
            out = triton_fast_accumulate_2d(
                vin,
                g_shifted,
                slice_scale_row,
                adc_ref,
                int(self.radc),
                self.mode,
                gp_shifted=gp_shifted,
                gn_shifted=gn_shifted,
                input_precision=self.triton_input_precision,
                block_r=block_r,
                block_l=block_l,
                block_k=block_k,
            )
        except Exception as exc:
            self._profile_stop(token, backend="triton", status="fallback", reason=type(exc).__name__)
            return None
        self._profile_stop(token, backend="triton", status="ok", shape_key=block_shape_key)
        return out

    def _triton_diff_input_accumulate(
        self,
        vin_p,
        vin_n,
        slice_scale_row,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
    ):
        """
        Try the native Triton mode2 differential-input fast path.

        The positive and negative input phases are ADC-quantized separately
        inside the kernel before digital subtraction, matching the slow path's
        differential-input order while avoiding the phase-batched cat/chunk
        intermediate.
        """
        if not (
            self.fast_inference_backend in ("triton", "triton_gidx")
            and len(vin_p.shape) == 4
            and len(vin_n.shape) == 4
            and vin_p.is_cuda
            and vin_n.is_cuda
            and self.mode == 2
            and not self.radc_is_list
        ):
            return None
        self._fastpath_count("mode2_diff_triton_attempt_count")
        try:
            from .triton_fast_accumulate import triton_diff_input_accumulate_2d
        except Exception as exc:
            self._fastpath_count("mode2_diff_triton_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_diff_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_diff_input_accumulate")
        try:
            out = triton_diff_input_accumulate_2d(
                vin_p,
                vin_n,
                gp_shifted,
                gn_shifted,
                slice_scale_row,
                adc_ref,
                int(self.radc),
                input_precision=self.triton_input_precision,
                # The two-phase kernel keeps two accumulators live; a smaller
                # row tile avoids register pressure on down-projection shapes.
                block_r=self._mode2_diff_block_r(),
                block_l=self.triton_block_l,
                block_k=self.triton_block_k,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_triton_fallback_count")
            self._profile_stop(token, backend="triton", status="fallback", reason=type(exc).__name__)
            return None
        self._fastpath_count("mode2_diff_triton_success_count")
        self._profile_stop(token, backend="triton", status="ok")
        return out

    def _triton_diff_input_accumulate_from_slices(
        self,
        x_slice_p,
        x_slice_n,
        xmax_i,
        slice_scale_row,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
    ):
        """
        Try the native Triton mode2 differential-input path with DAC fused.

        This path consumes integer sliced input phases directly and performs
        DAC quantization inside the same kernel as current accumulation and
        ADC. It is limited to deterministic 2-D inference because voltage
        noise requires random per-element perturbations.
        """
        if not (
            self.fast_inference_backend in ("triton", "triton_gidx")
            and self.vnoise == 0
            and len(x_slice_p.shape) == 4
            and len(x_slice_n.shape) == 4
            and x_slice_p.is_cuda
            and x_slice_n.is_cuda
            and self.mode == 2
            and not self.radc_is_list
        ):
            return None
        self._fastpath_count("mode2_diff_slice_triton_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_diff_input_accumulate_2d_from_slices,
                triton_diff_input_accumulate_2d_from_slices_gdiff,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_slice_triton_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_diff_fused_vin_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_diff_fused_vin_accumulate")
        try:
            if gn_shifted is None:
                out = triton_diff_input_accumulate_2d_from_slices_gdiff(
                    x_slice_p,
                    x_slice_n,
                    gp_shifted,
                    slice_scale_row,
                    float(xmax_i),
                    adc_ref,
                    int(self.rdac),
                    int(self.radc),
                    float(self.vread),
                    input_precision=self.triton_input_precision,
                    block_r=self._mode2_diff_block_r(),
                    block_l=self._mode2_diff_block_l(),
                    block_k=self.triton_block_k,
                )
                self._fastpath_count("mode2_diff_presubtract_triton_success_count")
            else:
                out = triton_diff_input_accumulate_2d_from_slices(
                    x_slice_p,
                    x_slice_n,
                    gp_shifted,
                    gn_shifted,
                    slice_scale_row,
                    float(xmax_i),
                    adc_ref,
                    int(self.rdac),
                    int(self.radc),
                    float(self.vread),
                    input_precision=self.triton_input_precision,
                    block_r=self._mode2_diff_block_r(),
                    block_l=self._mode2_diff_block_l(),
                    block_k=self.triton_block_k,
                )
        except Exception as exc:
            self._fastpath_count("mode2_diff_slice_triton_fallback_count")
            self._profile_stop(token, backend="triton", status="fallback", reason=type(exc).__name__)
            return None
        self._fastpath_count("mode2_diff_slice_triton_success_count")
        self._profile_stop(token, backend="triton", status="ok")
        return out

    def _triton_diff_input_all_input_slices_accumulate(
        self,
        x,
        slice_scale,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
    ):
        if not (
            bool(getattr(self, "triton_mode2_diff_fuse_input_slices", False))
            and self.fast_inference_backend in ("triton", "triton_gidx")
            and self.vnoise == 0
            and self.mode == 2
            and self.mode2_input_mode == "differential"
            and not self.radc_is_list
            and len(x.shape) == 2
            and getattr(x, "sliced_data_p", None) is not None
            and getattr(x, "sliced_data_n", None) is not None
            and x.sliced_data_p.is_cuda
            and x.sliced_data_n.is_cuda
            and gp_shifted is not None
            and gn_shifted is not None
        ):
            return None
        self._fastpath_count("mode2_diff_input_slice_fused_attempt_count")
        try:
            from .triton_fast_accumulate import triton_diff_input_accumulate_2d_all_input_slices
        except Exception as exc:
            self._fastpath_count("mode2_diff_input_slice_fused_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_mode2_diff_input_slice_fused_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_mode2_diff_input_slice_fused")
        try:
            out = triton_diff_input_accumulate_2d_all_input_slices(
                x.sliced_data_p,
                x.sliced_data_n,
                gp_shifted,
                gn_shifted,
                x.sliced_max_weights,
                slice_scale,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                float(self.vread),
                input_precision=self.triton_input_precision,
                block_r=self._mode2_diff_block_r(),
                block_l=self._mode2_diff_block_l(),
                block_k=self.triton_block_k,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_input_slice_fused_fallback_count")
            self._profile_stop(
                token,
                backend="triton_mode2_diff_input_slice_fused",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("mode2_diff_input_slice_fused_success_count")
        self._profile_stop(token, backend="triton_mode2_diff_input_slice_fused", status="ok")
        return out

    def _triton_diff_input_gidx_accumulate_from_slices(
        self,
        x_slice_p,
        x_slice_n,
        xmax_i,
        slice_scale_row,
        adc_ref,
        mat,
        c0,
        c1,
    ):
        if not (
            bool(getattr(self, "triton_mode2_diff_gidx_from_slices", False))
            and self.fast_inference_backend == "triton_gidx"
            and self.vnoise == 0
            and len(x_slice_p.shape) == 4
            and len(x_slice_n.shape) == 4
            and x_slice_p.is_cuda
            and x_slice_n.is_cuda
            and self.mode == 2
            and not self.radc_is_list
            and getattr(mat, "G_is_compressed", False)
            and isinstance(getattr(mat, "G_indices", None), tuple)
            and self._rv_all_same
            and not self._has_read_noise
            and not self._write_variation_is_virtual()
        ):
            return None
        self._fastpath_count("mode2_diff_gidx_slice_triton_attempt_count")
        try:
            from .triton_fast_accumulate import triton_diff_input_accumulate_2d_from_slices_gidx
        except Exception as exc:
            self._fastpath_count("mode2_diff_gidx_slice_triton_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_diff_gidx_fused_vin_fallback_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gp_idx, gn_idx = mat.G_indices
        gp_chunk = gp_idx[:, c0:c1, :, :, :]
        gn_chunk = gn_idx[:, c0:c1, :, :, :]
        token = self._profile_start("triton_diff_gidx_fused_vin_accumulate")
        try:
            read_sigma = float(self._rv_sigma) if self._has_read_noise else 0.0
            noise_offset_base = 0
            if read_sigma > 0:
                noise_offset_base = self._read_noise_restore_counter * gp_chunk.numel() * 2
                self._read_noise_restore_counter += 1
            out = triton_diff_input_accumulate_2d_from_slices_gidx(
                x_slice_p,
                x_slice_n,
                gp_chunk,
                gn_chunk,
                slice_scale_row,
                float(xmax_i),
                self.LGS,
                self.Q_G,
                read_sigma,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                float(self.vread),
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=self._mode2_diff_block_r(),
                block_l=self._mode2_diff_block_l(),
                block_k=self.triton_block_k,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_gidx_slice_triton_fallback_count")
            self._profile_stop(
                token,
                backend="triton_gidx_diff_fused_vin",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("mode2_diff_gidx_slice_triton_success_count")
        self._profile_stop(token, backend="triton_gidx_diff_fused_vin", status="ok")
        return out

    def _triton_diff_input_direct_final_output(
        self,
        x,
        mat,
        c0,
        c1,
        slice_scale,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
        out_buffer=None,
    ):
        if not (
            bool(getattr(self, "triton_mode2_diff_direct_final", False))
            and self.fast_inference_backend in ("triton", "triton_gidx")
            and self.mode == 2
            and self.mode2_input_mode == "differential"
            and self.vnoise == 0
            and not self.radc_is_list
            and len(x.shape) == 2
            and getattr(x, "sliced_data_p", None) is not None
            and getattr(x, "sliced_data_n", None) is not None
            and getattr(x, "max_data", None) is not None
            and getattr(mat, "max_data", None) is not None
            and x.sliced_data_p.is_cuda
            and x.sliced_data_n.is_cuda
            and x.max_data.is_cuda
            and mat.max_data.is_cuda
            and gp_shifted is not None
        ):
            return None
        self._fastpath_count("mode2_diff_direct_final_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_diff_input_accumulate_2d_from_slices_direct_final,
                triton_diff_input_accumulate_2d_from_slices_gdiff_direct_final,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_direct_final_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_mode2_diff_direct_final_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_mode2_diff_direct_final")
        try:
            if gn_shifted is None:
                out = triton_diff_input_accumulate_2d_from_slices_gdiff_direct_final(
                    x.sliced_data_p,
                    x.sliced_data_n,
                    gp_shifted,
                    x.sliced_max_weights,
                    slice_scale,
                    x.max_data,
                    mat.max_data[:, c0:c1, :, :],
                    adc_ref,
                    int(self.rdac),
                    int(self.radc),
                    self.vread,
                    x_qmax=float(self._sliced_quant_qmax(x)),
                    mat_qmax=float(self._sliced_quant_qmax(mat)),
                    input_precision=self.triton_input_precision,
                    block_r=self._mode2_diff_block_r(),
                    block_l=self._mode2_diff_block_l(),
                    block_k=self.triton_block_k,
                    out=out_buffer,
                    out_col_offset=self._direct_output_col_range_2d(mat, c0, c1)[0] if out_buffer is not None else 0,
                    out_cols=int(mat.shape[1]) if out_buffer is not None else None,
                )
            else:
                out = triton_diff_input_accumulate_2d_from_slices_direct_final(
                    x.sliced_data_p,
                    x.sliced_data_n,
                    gp_shifted,
                    gn_shifted,
                    x.sliced_max_weights,
                    slice_scale,
                    x.max_data,
                    mat.max_data[:, c0:c1, :, :],
                    adc_ref,
                    int(self.rdac),
                    int(self.radc),
                    self.vread,
                    x_qmax=float(self._sliced_quant_qmax(x)),
                    mat_qmax=float(self._sliced_quant_qmax(mat)),
                    input_precision=self.triton_input_precision,
                    block_r=self._mode2_diff_block_r(),
                    block_l=self._mode2_diff_block_l(),
                    block_k=self.triton_block_k,
                    out=out_buffer,
                    out_col_offset=self._direct_output_col_range_2d(mat, c0, c1)[0] if out_buffer is not None else 0,
                    out_cols=int(mat.shape[1]) if out_buffer is not None else None,
                )
        except Exception as exc:
            self._fastpath_count("mode2_diff_direct_final_fallback_count")
            self._profile_stop(
                token,
                backend="triton_mode2_diff_direct_final",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("mode2_diff_direct_final_success_count")
        if out_buffer is not None:
            self._fastpath_count("mode2_diff_direct_final_store_success_count")
        self._profile_stop(
            token,
            backend="triton_mode2_diff_direct_final",
            status="ok",
            direct_store=out_buffer is not None,
        )
        return out

    def _triton_diff_input_gidx_direct_final_output(
        self,
        x,
        mat,
        c0,
        c1,
        slice_scale,
        adc_ref,
        *,
        out_buffer=None,
    ):
        if not (
            bool(getattr(self, "triton_mode2_diff_direct_final", False))
            and self.fast_inference_backend == "triton_gidx"
            and self.mode == 2
            and self.mode2_input_mode == "differential"
            and self.vnoise == 0
            and not self.radc_is_list
            and len(x.shape) == 2
            and getattr(x, "sliced_data_p", None) is not None
            and getattr(x, "sliced_data_n", None) is not None
            and getattr(x, "max_data", None) is not None
            and getattr(mat, "max_data", None) is not None
            and x.sliced_data_p.is_cuda
            and x.sliced_data_n.is_cuda
            and x.max_data.is_cuda
            and mat.max_data.is_cuda
            and getattr(mat, "G_is_compressed", False)
            and isinstance(getattr(mat, "G_indices", None), tuple)
            and self._rv_all_same
            and not self._has_read_noise
            and not self._write_variation_is_virtual()
        ):
            return None
        gp_idx, gn_idx = mat.G_indices
        if gp_idx.dim() != 5 or gn_idx.dim() != 5 or gp_idx.shape != gn_idx.shape:
            return None
        self._fastpath_count("mode2_diff_gidx_direct_final_attempt_count")
        try:
            from .triton_fast_accumulate import (
                triton_diff_input_accumulate_2d_from_slices_gidx_direct_final,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_gidx_direct_final_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_mode2_diff_gidx_direct_final_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        gp_chunk = gp_idx[:, c0:c1, :, :, :]
        gn_chunk = gn_idx[:, c0:c1, :, :, :]
        token = self._profile_start("triton_mode2_diff_gidx_direct_final")
        try:
            read_sigma = float(self._rv_sigma) if self._has_read_noise else 0.0
            noise_offset_base = 0
            if read_sigma > 0:
                noise_offset_base = self._read_noise_restore_counter * gp_chunk.numel() * 2
                self._read_noise_restore_counter += 1
            out = triton_diff_input_accumulate_2d_from_slices_gidx_direct_final(
                x.sliced_data_p,
                x.sliced_data_n,
                gp_chunk,
                gn_chunk,
                x.sliced_max_weights,
                slice_scale,
                x.max_data,
                mat.max_data[:, c0:c1, :, :],
                self.LGS,
                self.Q_G,
                read_sigma,
                adc_ref,
                int(self.rdac),
                int(self.radc),
                self.vread,
                x_qmax=float(self._sliced_quant_qmax(x)),
                mat_qmax=float(self._sliced_quant_qmax(mat)),
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=self._mode2_diff_block_r(),
                block_l=self._mode2_diff_block_l(),
                block_k=self.triton_block_k,
                out=out_buffer,
                out_col_offset=self._direct_output_col_range_2d(mat, c0, c1)[0] if out_buffer is not None else 0,
                out_cols=int(mat.shape[1]) if out_buffer is not None else None,
            )
        except Exception as exc:
            self._fastpath_count("mode2_diff_gidx_direct_final_fallback_count")
            self._profile_stop(
                token,
                backend="triton_mode2_diff_gidx_direct_final",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("mode2_diff_gidx_direct_final_success_count")
        if out_buffer is not None:
            self._fastpath_count("mode2_diff_gidx_direct_final_store_success_count")
        self._profile_stop(
            token,
            backend="triton_mode2_diff_gidx_direct_final",
            status="ok",
            direct_store=out_buffer is not None,
            shape_key=(
                f"n={int(x.sliced_data_p.shape[0])};m={int(x.sliced_data_p.shape[1])};"
                f"i={int(x.sliced_data_p.shape[2])};j={int(x.sliced_data_p.shape[3])};"
                f"k={int(x.sliced_data_p.shape[4])};p={int(gp_chunk.shape[1])};"
                f"s={int(gp_chunk.shape[2])};l={int(gp_chunk.shape[4])};"
                f"read={bool(self._has_read_noise)}"
            ),
        )
        return out

    def _fast_weight_slice_accumulate(
        self,
        vin,
        g_shifted,
        slice_scale_row,
        adc_ref,
        *,
        gp_shifted=None,
        gn_shifted=None,
    ):
        """
        Fuse all weight slices for one input slice in the low-memory inference path.
        
        Parameters:
            vin (torch.Tensor): Quantized input voltage for one input slice.
            g_shifted (torch.Tensor or None): Mode 0 shifted conductance chunk.
            slice_scale_row (torch.Tensor): Per-weight-slice reconstruction scale for this input slice.
            adc_ref (float): ADC reference current.
            gp_shifted (torch.Tensor or None): Mode 2 positive branch conductance chunk.
            gn_shifted (torch.Tensor or None): Mode 2 negative branch conductance chunk.
        
        Returns:
            torch.Tensor: Accumulated output chunk with the same shape used by the slow path accumulator.
        """
        triton_out = self._triton_fast_weight_slice_accumulate(
            vin,
            g_shifted,
            slice_scale_row,
            adc_ref,
            gp_shifted=gp_shifted,
            gn_shifted=gn_shifted,
        )
        if triton_out is not None:
            return triton_out

        if len(vin.shape) == 4:
            if self.mode == 2:
                token = self._profile_start("fast_einsum_gp")
                partial = torch.einsum("nmjk,mpskl->nmpsjl", vin, gp_shifted)
                self._profile_stop(token)
                token = self._profile_start("fast_einsum_gn")
                neg_partial = torch.einsum("nmjk,mpskl->nmpsjl", vin, gn_shifted)
                self._profile_stop(token)
                token = self._profile_start("fast_branch_subtract")
                partial.sub_(neg_partial)
                self._profile_stop(token)
                del neg_partial
            else:
                token = self._profile_start("fast_einsum_current")
                partial = torch.einsum("nmjk,mpskl->nmpsjl", vin, g_shifted)
                self._profile_stop(token)
            scale_view = slice_scale_row.view(1, 1, 1, -1, 1, 1)
        elif len(vin.shape) == 5:
            if self.mode == 2:
                token = self._profile_start("fast_einsum_gp")
                partial = torch.einsum("bnmjk,mpskl->bnmpsjl", vin, gp_shifted)
                self._profile_stop(token)
                token = self._profile_start("fast_einsum_gn")
                neg_partial = torch.einsum("bnmjk,mpskl->bnmpsjl", vin, gn_shifted)
                self._profile_stop(token)
                token = self._profile_start("fast_branch_subtract")
                partial.sub_(neg_partial)
                self._profile_stop(token)
                del neg_partial
            else:
                token = self._profile_start("fast_einsum_current")
                partial = torch.einsum("bnmjk,mpskl->bnmpsjl", vin, g_shifted)
                self._profile_stop(token)
            scale_view = slice_scale_row.view(1, 1, 1, 1, -1, 1, 1)
        else:
            raise ValueError("Input data must be 2-D or 3-D.")

        token = self._profile_start("fast_adc_round")
        radc_scale = self.radc - 1
        partial.div_(adc_ref)
        partial.mul_(radc_scale)
        partial.round_()
        partial.div_(radc_scale)
        self._profile_stop(token)

        token = self._profile_start("fast_scale_reduce")
        partial.mul_(scale_view)
        out = partial.sum(dim=-3)
        self._profile_stop(token)
        return out


    def _dot(self, x, mat, _num2V_func, _num2R_func):
        """
        Dispatch the training dot product path for Mode 0 or Mode 2.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            _num2V_func (callable): Function that converts input slices to voltages.
            _num2R_func (callable): Function that returns conductance tensors.
        
        Returns:
            torch.Tensor: Matrix multiplication result.
        """
        if self.mode == 2:
            return self._dot_mode2(x, mat, _num2V_func, _num2R_func)
        else:
            return self._dot_mode0(x, mat, _num2V_func, _num2R_func)


    def _dot_mode0(self, x, mat, _num2V_func, _num2R_func):
        """
        Compute the standard bit-sliced dot product for Mode 0.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            _num2V_func (callable): Function that converts input slices to voltages.
            _num2R_func (callable): Function that returns conductance tensors.
        
        Returns:
            torch.Tensor: Matrix multiplication result for Mode 0.
        """
        Vin = _num2V_func(x)
        G = _num2R_func(mat)
        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            out = dot_high_dim(Vin, G - self.LGS)
            out = self._apply_adc_2d(out, adcRef)
            out = self._reconstruct_2d(out, x, mat, adcRef)
        elif len(x.shape) == 3:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            out = dot_high_dim(Vin, G - self.LGS)
            out = self._apply_adc_3d(out, adcRef)
            out = self._reconstruct_3d(out, x, mat, adcRef)
        else:
            raise ValueError("Input data must be 2-D or 3-D.")
        return out


    def _dot_mode1(self, x, mat, _num2V_func, _num2R_func):
        """
        Compute the linear differential-pair dot product for Mode 1.
        
        Parameters:
            x (SlicedDataMultiMode): Quantized input activation data.
            mat (SlicedDataMultiMode): Weight data containing positive and negative conductance branches.
            _num2V_func (callable): Unused compatibility parameter for the shared dot interface.
            _num2R_func (callable): Unused compatibility parameter for the shared dot interface.
        
        Returns:
            torch.Tensor: Matrix multiplication result for Mode 1.
        """
        Gp, Gn = mat.G
        if any(v > 0 for v in self.read_variation.values()):
            Gp = self._apply_read_noise_single(Gp)
            Gn = self._apply_read_noise_single(Gn)
            Gp = torch.clamp(Gp, self.LGS, self.HGS)
            Gn = torch.clamp(Gn, self.LGS, self.HGS)

        if x.quantized_data is None:
            raise ValueError("Mode 1 requires x.quantized_data.")

        x_data = x.quantized_data
        if x_data.dim() == 2:
            x_2d, has_batch = x_data, False
        elif x_data.dim() == 3:
            x_2d = x_data.reshape(-1, x_data.shape[-1])
            has_batch = True
        else:
            raise ValueError("Input data must be 2-D or 3-D.")

        in_features = x_2d.shape[-1]
        out_features = Gp.shape[-1]
        if Gp.shape[0] != in_features or Gn.shape != Gp.shape:
            raise ValueError("Mode 1 conductance shape mismatches input dimensions.")

        x_max = torch.max(torch.abs(x_2d)).clamp_min(torch.finfo(x_2d.dtype).tiny)

        tile_in, tile_out = self._resolve_mode1_tile_size(mat, in_features, out_features)
        if mat.mode1_w_max is not None:
            w_max = mat.mode1_w_max.to(device=Gp.device, dtype=torch.float32)
            if w_max.dim() == 2 and tuple(w_max.shape) != tuple(Gp.shape):
                w_max = self._expand_mode1_tile_scales(
                    w_max,
                    tuple(Gp.shape),
                    tile_in,
                    tile_out,
                )
            if w_max.numel() == 1:
                w_max = w_max.expand_as(Gp)
        elif mat.quantized_data is not None:
            w_max = self._build_mode1_tile_scales(
                mat.quantized_data,
                tile_in,
                tile_out,
            ).to(device=Gp.device, dtype=torch.float32)
        else:
            raise ValueError("Mode 1 requires cached tile-local weight scale.")
        if tuple(w_max.shape) != tuple(Gp.shape):
            raise ValueError("Mode 1 cached weight scale shape mismatches conductance shape.")
        if not self.mode1_adc_per_tile and tile_in < in_features:
            raise ValueError(
                "Mode 1 tile-local weight scaling requires mode1_adc_per_tile=True "
                "when the input dimension spans multiple paral_size tiles."
            )

        V_in = self.vread * torch.round(x_2d / x_max * (self.rdac - 1)) / (self.rdac - 1)
        if self.vnoise > 0:
            V_in = V_in * (1.0 + torch.randn_like(V_in) * self.vnoise)

        result_2d = torch.zeros((V_in.shape[0], out_features),
                                device=V_in.device, dtype=V_in.dtype)

        for c0 in range(0, out_features, tile_out):
            c1 = min(c0 + tile_out, out_features)
            if self.mode1_adc_per_tile:
                out_block = torch.zeros((V_in.shape[0], c1 - c0),
                                        device=V_in.device, dtype=V_in.dtype)
            else:
                cur_block = torch.zeros((V_in.shape[0], c1 - c0),
                                        device=V_in.device, dtype=V_in.dtype)

            for r0 in range(0, in_features, tile_in):
                r1 = min(r0 + tile_in, in_features)
                tw = r1 - r0
                Vt = V_in[:, r0:r1]
                I = (torch.matmul(Vt, Gp[r0:r1, c0:c1] - self.LGS)
                     - torch.matmul(Vt, Gn[r0:r1, c0:c1] - self.LGS))
                w_tile_max = w_max[r0:r1, c0:c1].amax()
                c2w = x_max * w_tile_max / (self.vread * self.Q_G * (self.g_level - 1))

                if self.mode1_adc_per_tile:
                    adc_ref = (self.HGS - self.LGS) * self.vread * tw
                    Iq = self._quantize_mode1_current(I, adc_ref)
                    out_block = out_block + Iq * adc_ref * c2w
                    del Iq
                else:
                    cur_block = cur_block + I
                del I

            if self.mode1_adc_per_tile:
                result_2d[:, c0:c1] = out_block
                del out_block
            else:
                adc_ref_g = (self.HGS - self.LGS) * self.vread * in_features
                Iq = self._quantize_mode1_current(cur_block, adc_ref_g)
                result_2d[:, c0:c1] = Iq * adc_ref_g * c2w
                del Iq, cur_block

        if has_batch:
            return result_2d.reshape(x_data.shape[0], x_data.shape[1], out_features)
        return result_2d

    def _triton_mode1_gidx_direct_final_output(
        self,
        x_2d,
        mat,
        scale_grid,
        x_max,
        tile_in,
        tile_out,
        *,
        c0=None,
        c1=None,
    ):
        if not (
            bool(getattr(self, "triton_mode1_gidx_direct_final", False))
            and self.fast_inference_backend in ("triton", "triton_gidx")
            and self.mode == 1
            and not self.radc_is_list
            and self.vnoise == 0
            and self.mode1_adc_per_tile
            and x_2d.dim() == 2
            and x_2d.is_cuda
            and getattr(mat, "G_is_compressed", False)
            and isinstance(getattr(mat, "G_indices", None), tuple)
            and self._rv_all_same
            and not self._write_variation_is_virtual()
        ):
            return None
        gp_idx, gn_idx = mat.G_indices
        if gp_idx.dim() != 2 or gn_idx.dim() != 2 or gp_idx.shape != gn_idx.shape:
            return None
        if int(gp_idx.shape[0]) != int(x_2d.shape[-1]) or int(gp_idx.shape[1]) != int(mat.shape[1]):
            return None
        full_out_features = int(mat.shape[1])
        if c0 is None or c1 is None:
            out_start, out_end = 0, full_out_features
            scale_chunk = scale_grid
        else:
            out_start, out_end = self._direct_output_col_range_2d(mat, c0, c1)
            if out_start >= out_end:
                return None
            scale_chunk = scale_grid[:, int(c0):int(c1)]
            if scale_chunk.numel() == 0:
                return None
        if int(tile_in) != int(getattr(mat, "paral_size", (tile_in, tile_out))[0]):
            return None
        if int(tile_out) != int(getattr(mat, "paral_size", (tile_in, tile_out))[1]):
            return None
        plan = self._mode1_triton_plan(
            x_2d,
            mat,
            tile_in,
            tile_out,
            out_cols=int(out_end - out_start),
        )
        max_out_cols = int(tile_out) * max(1, int(plan["chunk_limit"]))
        if (c0 is None or c1 is None) and full_out_features > max_out_cols:
            return None
        if (out_end - out_start) > max_out_cols:
            return None

        self._fastpath_count("mode1_gidx_direct_final_attempt_count")
        try:
            from .triton_fast_accumulate import triton_mode1_gidx_direct_final
        except Exception as exc:
            self._fastpath_count("mode1_gidx_direct_final_fallback_count")
            if self.profile:
                self.profile_events.append({
                    "label": "triton_mode1_gidx_direct_final_import",
                    "ms": 0.0,
                    "reason": type(exc).__name__,
                })
            return None

        token = self._profile_start("triton_mode1_gidx_direct_final")
        try:
            gp_chunk = gp_idx[:, out_start:out_end]
            gn_chunk = gn_idx[:, out_start:out_end]
            noise_offset_base = 0
            if self._has_read_noise and self._rv_sigma > 0:
                noise_offset_base = self._read_noise_restore_counter * gp_chunk.numel() * 2
                self._read_noise_restore_counter += 1
            out = triton_mode1_gidx_direct_final(
                x_2d,
                gp_chunk,
                gn_chunk,
                scale_chunk,
                x_max=x_max,
                lgs=self.LGS,
                q_g=self.Q_G,
                read_sigma=float(self._rv_sigma if self._has_read_noise else 0.0),
                adc_ref_unit=(self.HGS - self.LGS) * self.vread,
                rdac=int(self.rdac),
                radc=int(self.radc),
                vread=self.vread,
                g_level=int(self.g_level),
                tile_in=int(tile_in),
                tile_out=int(tile_out),
                noise_seed=self._read_noise_seed_base,
                noise_offset_base=noise_offset_base,
                input_precision=self.triton_input_precision,
                block_r=int(plan["block_r"]),
                block_l=int(plan["block_l"]),
                block_k=int(plan["block_k"]),
                input_tile_group=int(plan["input_tile_group"]),
            )
        except Exception as exc:
            self._fastpath_count("mode1_gidx_direct_final_fallback_count")
            self._profile_stop(
                token,
                backend="triton_mode1_gidx_direct_final",
                status="fallback",
                reason=type(exc).__name__,
                message=str(exc),
            )
            return None
        self._fastpath_count("mode1_gidx_direct_final_success_count")
        if int(plan["input_tile_group"]) > 1:
            self._fastpath_count("mode1_gidx_direct_final_grouped_success_count")
        self._profile_stop(
            token,
            backend="triton_mode1_gidx_direct_final",
            status="ok",
            shape_key=(
                f"rows={int(x_2d.shape[0])};in={int(x_2d.shape[1])};"
                f"out={int(out_end - out_start)};tile={int(tile_in)}x{int(tile_out)};"
                f"group={int(plan['input_tile_group'])};"
                f"read={bool(self._has_read_noise)};{plan['shape_key']}"
            ),
        )
        return out

    def _triton_mode1_chunked_direct_final_output(
        self,
        x_2d,
        mat,
        scale_grid,
        x_max,
        tile_in,
        tile_out,
    ):
        if not (
            bool(getattr(self, "triton_mode1_gidx_direct_final", False))
            and bool(getattr(self, "triton_mode1_chunked_direct_final", False))
        ):
            return None
        ndc_y = int(scale_grid.shape[1])
        chunk_ndc = max(1, min(ndc_y, self._mode1_triton_chunk_limit(ndc_y, mat=mat)))
        if self.inference_chunk_size is not None:
            chunk_ndc = max(1, min(chunk_ndc, int(self.inference_chunk_size)))
        if ndc_y <= chunk_ndc:
            return None

        self._fastpath_count("mode1_chunked_direct_final_attempt_count")
        out = torch.empty(
            (int(x_2d.shape[0]), int(mat.shape[1])),
            device=x_2d.device,
            dtype=torch.float32,
        )
        for c0 in range(0, ndc_y, chunk_ndc):
            c1 = min(c0 + chunk_ndc, ndc_y)
            chunk = self._triton_mode1_gidx_direct_final_output(
                x_2d,
                mat,
                scale_grid,
                x_max,
                tile_in,
                tile_out,
                c0=c0,
                c1=c1,
            )
            if chunk is None:
                self._fastpath_count("mode1_chunked_direct_final_fallback_count")
                return None
            out_start, out_end = self._direct_output_col_range_2d(mat, c0, c1)
            out[:, out_start:out_end] = chunk[:, : out_end - out_start]
            del chunk
        self._fastpath_count("mode1_chunked_direct_final_success_count")
        return out

    def _dot_mode1_inference(self, x, mat):
        """
        Memory-efficient Mode 1 inference using differential branch chunks.

        This keeps the Mode 1 analog order: a signed DAC input drives a
        positive/negative weight pair, branch current is subtracted before ADC,
        and ADC quantization is applied per input tile.
        """
        if x.quantized_data is None:
            raise ValueError("Mode 1 inference requires x.quantized_data.")
        if mat.mode1_w_max is None:
            raise ValueError("Mode 1 inference requires compact tile scales.")

        x_data = x.quantized_data
        if x_data.dim() == 2:
            x_2d, has_batch = x_data, False
            batch_shape = None
        elif x_data.dim() == 3:
            batch_shape = x_data.shape[:2]
            x_2d = x_data.reshape(-1, x_data.shape[-1])
            has_batch = True
        else:
            raise ValueError("Input data must be 2-D or 3-D.")

        in_features = x_2d.shape[-1]
        out_features = int(mat.shape[1])
        tile_in, tile_out = self._resolve_mode1_tile_size(mat, in_features, out_features)
        scale_grid = mat.mode1_w_max.to(device=x_2d.device, dtype=torch.float32)

        x_max = torch.max(torch.abs(x_2d)).clamp_min(torch.finfo(x_2d.dtype).tiny)

        direct_mode1 = self._triton_mode1_gidx_direct_final_output(
            x_2d,
            mat,
            scale_grid,
            x_max,
            tile_in,
            tile_out,
        )
        if direct_mode1 is not None:
            if has_batch:
                return direct_mode1.reshape(batch_shape[0], batch_shape[1], out_features)
            return direct_mode1

        direct_mode1 = self._triton_mode1_chunked_direct_final_output(
            x_2d,
            mat,
            scale_grid,
            x_max,
            tile_in,
            tile_out,
        )
        if direct_mode1 is not None:
            if has_batch:
                return direct_mode1.reshape(batch_shape[0], batch_shape[1], out_features)
            return direct_mode1

        V_in = self.vread * torch.round(x_2d / x_max * (self.rdac - 1)) / (self.rdac - 1)
        if self.vnoise > 0:
            V_in = V_in * (1.0 + torch.randn_like(V_in) * self.vnoise)

        chunks = list(self._iter_output_chunks(mat))
        direct_write_output = self._can_direct_write_output_chunks_2d(x, mat, chunks) and len(chunks) > 1
        direct_output = None
        if direct_write_output:
            direct_output = torch.empty((V_in.shape[0], out_features), device=V_in.device, dtype=torch.float32)
        out_chunks = []
        row_ranges = [
            (row_tile_idx, r0, min(r0 + tile_in, in_features))
            for row_tile_idx, r0 in enumerate(range(0, in_features, tile_in))
        ]

        def run_single_row_tile(row_tile_idx, r0, r1, out_block, c0, c1, width):
            tw = r1 - r0
            Vt = V_in[:, r0:r1].to(self.compute_dtype)
            Gp_shifted, Gn_shifted = self._get_mode1_shifted_chunk(
                mat,
                r0,
                r1,
                c0,
                c1,
                branch_base=row_tile_idx * 2,
            )
            Gdiff_shifted = None
            if self.compute_dtype == torch.float32:
                # Algebraically equivalent to two branch GEMMs, and FP32
                # keeps the ADC code identical in the inference tests.
                Gp_shifted.sub_(Gn_shifted)
                Gdiff_shifted = Gp_shifted
                del Gn_shifted
                I = torch.matmul(Vt, Gdiff_shifted)
            else:
                # BF16/FP16 can move values across ADC bins when the
                # subtraction is fused before GEMM, so keep the exact
                # branch order for reduced-precision semantic runs.
                I = torch.matmul(Vt, Gp_shifted) - torch.matmul(Vt, Gn_shifted)
                del Gp_shifted, Gn_shifted
            adc_ref = (self.HGS - self.LGS) * self.vread * tw
            Iq = self._quantize_mode1_current(I, adc_ref)
            w_scale = self._mode1_scale_cols(
                scale_grid,
                row_tile_idx,
                c0,
                c1,
                tile_out,
                out_features,
                device=V_in.device,
                dtype=torch.float32,
            )
            c2w = x_max.to(torch.float32) * w_scale / (self.vread * self.Q_G * (self.g_level - 1))
            out_block.add_(Iq.to(torch.float32) * adc_ref * c2w.view(1, -1))
            if Gdiff_shifted is not None:
                del Gdiff_shifted
            del Vt, I, Iq, w_scale, c2w

        def run_row_tile_group(group, out_block, c0, c1, width):
            if (
                self._write_variation_is_virtual()
                or len(group) <= 1
            ):
                return False
            group_tiles = len(group)
            group_r0 = group[0][1]
            group_r1 = group[-1][2]
            if group_r1 - group_r0 != group_tiles * tile_in:
                return False
            self._fastpath_count("mode1_tile_group_attempt_count")
            try:
                Vg = (
                    V_in[:, group_r0:group_r1]
                    .to(self.compute_dtype)
                    .reshape(V_in.shape[0], group_tiles, tile_in)
                    .permute(1, 0, 2)
                    .contiguous()
                )
                Gp_shifted, Gn_shifted = self._get_mode1_shifted_chunk(
                    mat,
                    group_r0,
                    group_r1,
                    c0,
                    c1,
                    branch_base=group[0][0] * 2,
                )
                if self.compute_dtype == torch.float32:
                    Gp_shifted.sub_(Gn_shifted)
                    Gdiff = Gp_shifted.reshape(group_tiles, tile_in, width)
                    del Gn_shifted
                    I = torch.bmm(Vg, Gdiff).permute(1, 0, 2).contiguous()
                    del Gdiff, Gp_shifted
                else:
                    Gp = Gp_shifted.reshape(group_tiles, tile_in, width)
                    Gn = Gn_shifted.reshape(group_tiles, tile_in, width)
                    I = (torch.bmm(Vg, Gp) - torch.bmm(Vg, Gn)).permute(1, 0, 2).contiguous()
                    del Gp, Gn, Gp_shifted, Gn_shifted
                adc_ref = torch.full(
                    (group_tiles,),
                    (self.HGS - self.LGS) * self.vread * tile_in,
                    device=V_in.device,
                    dtype=torch.float32,
                )
                Iq = self._quantize_mode1_current(I, adc_ref.view(1, group_tiles, 1))
                w_scales = torch.stack(
                    [
                        self._mode1_scale_cols(
                            scale_grid,
                            row_tile_idx,
                            c0,
                            c1,
                            tile_out,
                            out_features,
                            device=V_in.device,
                            dtype=torch.float32,
                        )
                        for row_tile_idx, _, _ in group
                    ],
                    dim=0,
                )
                c2w = (
                    x_max.to(torch.float32)
                    * w_scales
                    / (self.vread * self.Q_G * (self.g_level - 1))
                )
                out_block.add_(
                    (Iq.to(torch.float32) * adc_ref.view(1, group_tiles, 1) * c2w.view(1, group_tiles, width))
                    .sum(dim=1)
                )
                del Vg, I, Iq, adc_ref, w_scales, c2w
            except Exception:
                self._fastpath_count("mode1_tile_group_fallback_count")
                return False
            self._fastpath_count("mode1_tile_group_success_count")
            return True

        for c0, c1 in chunks:
            out_start, out_end = self._direct_output_col_range_2d(mat, c0, c1)
            width = max(0, out_end - out_start)
            if width == 0:
                continue
            out_block = torch.zeros((V_in.shape[0], width), device=V_in.device, dtype=torch.float32)
            enable_grouped_mode1 = bool(getattr(self, "mode1_grouped_tile_gemm", False))
            max_tmp_elems = max(1, int(getattr(self, "inference_chunk_size", None) or 8 * 1024 * 1024))
            max_group_tiles = (
                max(1, min(len(row_ranges), max_tmp_elems // max(1, V_in.shape[0] * width)))
                if enable_grouped_mode1
                else 1
            )
            start_idx = 0
            while start_idx < len(row_ranges):
                end_idx = min(start_idx + max_group_tiles, len(row_ranges))
                group = row_ranges[start_idx:end_idx]
                if not run_row_tile_group(group, out_block, c0, c1, width):
                    for row_tile_idx, r0, r1 in group:
                        run_single_row_tile(row_tile_idx, r0, r1, out_block, c0, c1, width)
                start_idx = end_idx
            if direct_output is not None:
                direct_output[:, out_start:out_end] = out_block[:, :width]
            else:
                out_chunks.append(out_block)

        out = direct_output if direct_output is not None else torch.cat(out_chunks, dim=-1)
        if has_batch:
            return out.reshape(batch_shape[0], batch_shape[1], out_features)
        return out


    def _dot_mode2(self, x, mat, _num2V_func, _num2R_func):
        """
        Compute differential-pair bit-sliced dot product for Mode 2.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data with positive and negative branches.
            _num2V_func (callable): Function that converts input slices to voltages.
            _num2R_func (callable): Function that returns positive and negative conductance branches.
        
        Returns:
            torch.Tensor: Matrix multiplication result for Mode 2.
        """
        Gp, Gn = _num2R_func(mat)
        if self.mode2_input_mode == "differential":
            if x.sliced_data_p is None or x.sliced_data_n is None:
                raise ValueError(
                    "Mode 2 differential input requires x.sliced_data_p and x.sliced_data_n. "
                    "Create the input with inference=False, or keep differential input branches in inference mode."
                )
            Vin_p = self._num2V_from_sliced_data(x, x.sliced_data_p)
            Vin_n = self._num2V_from_sliced_data(x, x.sliced_data_n)
            if len(x.shape) == 2:
                adcRef = (self.HGS - self.LGS) * self.vread * Vin_p.shape[-1]
                out_p = dot_high_dim(Vin_p, Gp - self.LGS) - dot_high_dim(Vin_p, Gn - self.LGS)
                out_n = dot_high_dim(Vin_n, Gp - self.LGS) - dot_high_dim(Vin_n, Gn - self.LGS)
                out = self._apply_adc_2d(out_p, adcRef) - self._apply_adc_2d(out_n, adcRef)
                out = self._reconstruct_2d(out, x, mat, adcRef)
            elif len(x.shape) == 3:
                adcRef = (self.HGS - self.LGS) * self.vread * Vin_p.shape[-1]
                out_p = dot_high_dim(Vin_p, Gp - self.LGS) - dot_high_dim(Vin_p, Gn - self.LGS)
                out_n = dot_high_dim(Vin_n, Gp - self.LGS) - dot_high_dim(Vin_n, Gn - self.LGS)
                out = self._apply_adc_3d(out_p, adcRef) - self._apply_adc_3d(out_n, adcRef)
                out = self._reconstruct_3d(out, x, mat, adcRef)
            else:
                raise ValueError("Input data must be 2-D or 3-D.")
            return out

        Vin = _num2V_func(x)
        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            out = dot_high_dim(Vin, Gp - self.LGS) - dot_high_dim(Vin, Gn - self.LGS)
            out = self._apply_adc_2d(out, adcRef)
            out = self._reconstruct_2d(out, x, mat, adcRef)
        elif len(x.shape) == 3:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            out = dot_high_dim(Vin, Gp - self.LGS) - dot_high_dim(Vin, Gn - self.LGS)
            out = self._apply_adc_3d(out, adcRef)
            out = self._reconstruct_3d(out, x, mat, adcRef)
        else:
            raise ValueError("Input data must be 2-D or 3-D.")
        return out


    def _dot_inference(self, x: SlicedDataMultiMode, mat: SlicedDataMultiMode):
        """
        Compute memory-efficient inference for Mode 0 or Mode 2 by slice pairs and output chunks.
        
        Parameters:
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
        
        Returns:
            torch.Tensor: Matrix multiplication result assembled from inference chunks.
        """
        ns_x = len(x.slice_method)
        ns_y = len(mat.slice_method)
        adcRef = (self.HGS - self.LGS) * self.vread * x.sliced_data.shape[-1]
        scale_base = adcRef / self.Q_G / self.vread / (self.g_level - 1)
        chunks = list(self._iter_output_chunks(mat))
        direct_write_output = self._can_direct_write_output_chunks_2d(x, mat, chunks) and len(chunks) > 1
        direct_output = None
        out_chunks = []
        differential_input = self.mode == 2 and self.mode2_input_mode == "differential"
        slice_scale = (
            x.sliced_max_weights.view(ns_x, 1)
            * x.sliced_weights.view(ns_x, 1)
            * mat.sliced_max_weights.view(1, ns_y)
            * mat.sliced_weights.view(1, ns_y)
            * scale_base
        )
        use_fast_inference = self._can_use_fast_inference(differential_input)

        def ensure_direct_output(dtype=torch.float32):
            nonlocal direct_output
            if direct_output is None:
                direct_output = self._alloc_direct_output_2d(x, mat, dtype)
            return direct_output

        def emit_chunk(chunk, c0, c1):
            nonlocal direct_output
            if direct_write_output:
                ensure_direct_output(chunk.dtype)
                token = self._profile_start("write_output_chunk")
                self._write_direct_output_chunk_2d(direct_output, chunk, mat, c0, c1)
                self._profile_stop(token, c0=c0, c1=c1)
            else:
                out_chunks.append(chunk)

        for c0, c1 in chunks:
            G_shifted = None
            Gp_shifted = None
            Gn_shifted = None
            conductance_restored = False

            def ensure_shifted_conductance():
                nonlocal G_shifted, Gp_shifted, Gn_shifted, conductance_restored
                if conductance_restored:
                    return
                token = self._profile_start("restore_G")
                if self.mode == 2:
                    use_presubtract = (
                        differential_input
                        and bool(getattr(self, "triton_mode2_diff_presubtract", False))
                        and use_fast_inference
                        and len(x.shape) == 2
                    )
                    if use_presubtract:
                        Gp_shifted = self._try_get_mode2_gdiff_shifted_chunk(mat, c0, c1)
                        if Gp_shifted is not None:
                            Gn_shifted = None
                            self._fastpath_count("mode2_diff_presubtract_count")
                    if Gp_shifted is None:
                        Gp_shifted, Gn_shifted = self._get_mode2_shifted_chunk(mat, c0, c1)
                        if use_presubtract:
                            Gp_shifted.sub_(Gn_shifted)
                            del Gn_shifted
                            Gn_shifted = None
                            self._fastpath_count("mode2_diff_presubtract_count")
                else:
                    G_shifted = self._get_mode0_shifted_chunk(mat, c0, c1)
                self._profile_stop(token, c0=c0, c1=c1)
                conductance_restored = True

            if use_fast_inference:
                accumulated = self._triton_gidx_read_noise_input_slice_fused_accumulate(
                    x,
                    mat,
                    c0,
                    c1,
                    slice_scale,
                    adcRef,
                )
                if accumulated is not None:
                    token = self._profile_start("finalize")
                    finalized = self._finalize_inference_chunk_2d(accumulated, x, mat, c0, c1)
                    self._profile_stop(token, c0=c0, c1=c1, read_noise_input_slice_fused=True)
                    emit_chunk(finalized, c0, c1)
                    del finalized
                    del accumulated
                    continue

                accumulated = self._triton_gidx_input_slice_fused_accumulate(
                    x,
                    mat,
                    c0,
                    c1,
                    slice_scale,
                    adcRef,
                )
                if accumulated is not None:
                    token = self._profile_start("finalize")
                    if len(x.shape) == 3:
                        out_chunks.append(self._finalize_inference_chunk_3d(accumulated, x, mat, c0, c1))
                    else:
                        finalized = self._finalize_inference_chunk_2d(accumulated, x, mat, c0, c1)
                        emit_chunk(finalized, c0, c1)
                        del finalized
                    self._profile_stop(token, c0=c0, c1=c1, input_slice_fused=True)
                    del accumulated
                    continue

                if (
                    self.mode == 0
                    and self._can_use_triton_restored_input_slice_fusion(x, mat)
                ):
                    direct_store_buffer = ensure_direct_output(torch.float32) if direct_write_output else None
                    if direct_store_buffer is not None:
                        self._zero_direct_output_chunk_2d(direct_store_buffer, mat, c0, c1)
                    direct_final = self._triton_gidx_direct_final_output(
                        x,
                        mat,
                        c0,
                        c1,
                        slice_scale,
                        adcRef,
                        out_buffer=direct_store_buffer,
                    )
                    if direct_final is not None:
                        if direct_store_buffer is None:
                            emit_chunk(direct_final, c0, c1)
                            del direct_final
                        continue

                    ensure_shifted_conductance()
                    if direct_store_buffer is not None:
                        self._zero_direct_output_chunk_2d(direct_store_buffer, mat, c0, c1)
                    direct_final = self._triton_restored_direct_final_output(
                        x,
                        mat,
                        c0,
                        c1,
                        G_shifted,
                        slice_scale,
                        adcRef,
                        out_buffer=direct_store_buffer,
                    )
                    if direct_final is not None:
                        if direct_store_buffer is None:
                            emit_chunk(direct_final, c0, c1)
                            del direct_final
                        continue

                    accumulated = self._triton_restored_input_slice_fused_accumulate(
                        x,
                        mat,
                        c0,
                        c1,
                        G_shifted,
                        slice_scale,
                        adcRef,
                    )
                    if accumulated is not None:
                        token = self._profile_start("finalize")
                        finalized = self._finalize_inference_chunk_2d(accumulated, x, mat, c0, c1)
                        self._profile_stop(token, c0=c0, c1=c1, restored_input_slice_fused=True)
                        emit_chunk(finalized, c0, c1)
                        del finalized
                        del accumulated
                        continue

                if (
                    self.mode == 2
                    and differential_input
                    and len(x.shape) == 2
                    and x.sliced_data_p is not None
                    and x.sliced_data_n is not None
                ):
                    direct_store_buffer = ensure_direct_output(torch.float32) if direct_write_output else None
                    if direct_store_buffer is not None:
                        self._zero_direct_output_chunk_2d(direct_store_buffer, mat, c0, c1)
                    direct_final = self._triton_diff_input_gidx_direct_final_output(
                        x,
                        mat,
                        c0,
                        c1,
                        slice_scale,
                        adcRef,
                        out_buffer=direct_store_buffer,
                    )
                    if direct_final is not None:
                        if direct_store_buffer is None:
                            emit_chunk(direct_final, c0, c1)
                            del direct_final
                        continue

                    ensure_shifted_conductance()
                    if direct_store_buffer is not None:
                        self._zero_direct_output_chunk_2d(direct_store_buffer, mat, c0, c1)
                    direct_final = self._triton_diff_input_direct_final_output(
                        x,
                        mat,
                        c0,
                        c1,
                        slice_scale,
                        adcRef,
                        gp_shifted=Gp_shifted,
                        gn_shifted=Gn_shifted,
                        out_buffer=direct_store_buffer,
                    )
                    if direct_final is not None:
                        if direct_store_buffer is None:
                            emit_chunk(direct_final, c0, c1)
                            del direct_final
                        continue

                    accumulated = self._triton_diff_input_all_input_slices_accumulate(
                        x,
                        slice_scale,
                        adcRef,
                        gp_shifted=Gp_shifted,
                        gn_shifted=Gn_shifted,
                    )
                    if accumulated is not None:
                        token = self._profile_start("finalize")
                        finalized = self._finalize_inference_chunk_2d(accumulated, x, mat, c0, c1)
                        self._profile_stop(token, c0=c0, c1=c1, mode2_diff_input_slice_fused=True)
                        emit_chunk(finalized, c0, c1)
                        del finalized
                        del accumulated
                        continue

            accumulated = None
            for i in range(ns_x):
                xmax_i = x.sliced_max_weights[i]
                partial = None
                if (
                    use_fast_inference
                    and differential_input
                    and len(x.shape) == 2
                    and x.sliced_data_p is not None
                    and x.sliced_data_n is not None
                ):
                    x_sl_p_i = x.sliced_data_p[:, :, i, :, :]
                    x_sl_n_i = x.sliced_data_n[:, :, i, :, :]
                    partial = self._triton_diff_input_gidx_accumulate_from_slices(
                        x_sl_p_i,
                        x_sl_n_i,
                        xmax_i,
                        slice_scale[i],
                        adcRef,
                        mat,
                        c0,
                        c1,
                    )
                    if partial is None:
                        ensure_shifted_conductance()
                        partial = self._triton_diff_input_accumulate_from_slices(
                            x_sl_p_i,
                            x_sl_n_i,
                            xmax_i,
                            slice_scale[i],
                            adcRef,
                            gp_shifted=Gp_shifted,
                            gn_shifted=Gn_shifted,
                        )
                    del x_sl_p_i, x_sl_n_i

                if partial is not None:
                    token = self._profile_start("fast_einsum_adc_accumulate")
                    if accumulated is None:
                        accumulated = partial
                    else:
                        accumulated.add_(partial)
                        del partial
                    self._profile_stop(token, c0=c0, c1=c1, input_slice=i, fused_vin=True)
                    continue

                token = self._profile_start("build_Vin")
                if len(x.shape) == 3:
                    if differential_input:
                        if x.sliced_data_p is None or x.sliced_data_n is None:
                            raise ValueError("Mode 2 differential-input inference requires input branch tensors.")
                        x_sl_p_i = x.sliced_data_p[:, :, :, i, :, :]
                        x_sl_n_i = x.sliced_data_n[:, :, :, i, :, :]
                        Vin_p_i = self._build_vin_for_slice(x_sl_p_i, xmax_i)
                        Vin_n_i = self._build_vin_for_slice(x_sl_n_i, xmax_i)
                        del x_sl_p_i, x_sl_n_i
                    else:
                        x_sl_i = x.sliced_data[:, :, :, i, :, :]
                        Vin_i = self._build_vin_for_slice(x_sl_i, xmax_i)
                        del x_sl_i
                elif len(x.shape) == 2:
                    if differential_input:
                        if x.sliced_data_p is None or x.sliced_data_n is None:
                            raise ValueError("Mode 2 differential-input inference requires input branch tensors.")
                        x_sl_p_i = x.sliced_data_p[:, :, i, :, :]
                        x_sl_n_i = x.sliced_data_n[:, :, i, :, :]
                        Vin_p_i = self._build_vin_for_slice(x_sl_p_i, xmax_i)
                        Vin_n_i = self._build_vin_for_slice(x_sl_n_i, xmax_i)
                        del x_sl_p_i, x_sl_n_i
                    else:
                        x_sl_i = x.sliced_data[:, :, i, :, :]
                        Vin_i = self._build_vin_for_slice(x_sl_i, xmax_i)
                        del x_sl_i
                else:
                    raise ValueError("Input data must be 2-D or 3-D.")
                self._profile_stop(token, c0=c0, c1=c1, input_slice=i)

                if use_fast_inference:
                    token = self._profile_start("fast_einsum_adc_accumulate")
                    if differential_input:
                        ensure_shifted_conductance()
                        partial = self._triton_diff_input_accumulate(
                            Vin_p_i,
                            Vin_n_i,
                            slice_scale[i],
                            adcRef,
                            gp_shifted=Gp_shifted,
                            gn_shifted=Gn_shifted,
                        )
                        if partial is None:
                            phase_dim = 1 if len(Vin_p_i.shape) == 5 else 0
                            Vin_pair_i = torch.cat((Vin_p_i, Vin_n_i), dim=phase_dim)
                            del Vin_p_i, Vin_n_i
                            partial_pair = self._fast_weight_slice_accumulate(
                                Vin_pair_i,
                                None,
                                slice_scale[i],
                                adcRef,
                                gp_shifted=Gp_shifted,
                                gn_shifted=Gn_shifted,
                            )
                            del Vin_pair_i
                            partial_p, partial_n = partial_pair.chunk(2, dim=phase_dim)
                            partial = partial_p.sub_(partial_n)
                            del partial_p, partial_n, partial_pair
                        else:
                            del Vin_p_i, Vin_n_i
                    elif self.mode == 2:
                        partial = self._triton_gidx_weight_slice_accumulate(
                            Vin_i,
                            mat,
                            c0,
                            c1,
                            slice_scale[i],
                            adcRef,
                        )
                        if partial is None:
                            ensure_shifted_conductance()
                            partial = self._fast_weight_slice_accumulate(
                                Vin_i,
                                None,
                                slice_scale[i],
                                adcRef,
                                gp_shifted=Gp_shifted,
                                gn_shifted=Gn_shifted,
                            )
                    else:
                        partial = self._triton_gidx_weight_slice_accumulate(
                            Vin_i,
                            mat,
                            c0,
                            c1,
                            slice_scale[i],
                            adcRef,
                        )
                        if partial is None:
                            ensure_shifted_conductance()
                            partial = self._fast_weight_slice_accumulate(
                                Vin_i,
                                G_shifted,
                                slice_scale[i],
                                adcRef,
                            )
                    if accumulated is None:
                        accumulated = partial
                    else:
                        accumulated.add_(partial)
                        del partial
                    self._profile_stop(token, c0=c0, c1=c1, input_slice=i)
                    if not differential_input:
                        del Vin_i
                    continue

                for j in range(ns_y):
                    ensure_shifted_conductance()
                    adc_already_applied = False
                    token = self._profile_start("einsum_current")
                    if len(x.shape) == 3:
                        if self.mode == 2:
                            if differential_input:
                                partial_p = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_p_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial_p = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_p_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial_p.sub_(neg_partial_p)
                                del neg_partial_p
                                partial_n = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_n_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial_n = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_n_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial_n.sub_(neg_partial_n)
                                del neg_partial_n
                                self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                                token = self._profile_start("adc")
                                radc_j = self.radc[j] if self.radc_is_list else self.radc
                                partial_p.div_(adcRef).mul_(radc_j - 1).round_().div_(radc_j - 1)
                                partial_n.div_(adcRef).mul_(radc_j - 1).round_().div_(radc_j - 1)
                                self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                                partial = partial_p.sub_(partial_n)
                                del partial_n
                                adc_already_applied = True
                            else:
                                partial = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial = torch.einsum(
                                    "bnmjk, mpkl->bnmpjl",
                                    Vin_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial.sub_(neg_partial)
                                del neg_partial
                        else:
                            partial = torch.einsum(
                                "bnmjk, mpkl->bnmpjl",
                                Vin_i,
                                G_shifted[:, :, j, :, :],
                            )
                    else:
                        if self.mode == 2:
                            if differential_input:
                                partial_p = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_p_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial_p = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_p_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial_p.sub_(neg_partial_p)
                                del neg_partial_p
                                partial_n = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_n_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial_n = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_n_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial_n.sub_(neg_partial_n)
                                del neg_partial_n
                                self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                                token = self._profile_start("adc")
                                radc_j = self.radc[j] if self.radc_is_list else self.radc
                                partial_p.div_(adcRef).mul_(radc_j - 1).round_().div_(radc_j - 1)
                                partial_n.div_(adcRef).mul_(radc_j - 1).round_().div_(radc_j - 1)
                                self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                                partial = partial_p.sub_(partial_n)
                                del partial_n
                                adc_already_applied = True
                            else:
                                partial = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_i,
                                    Gp_shifted[:, :, j, :, :],
                                )
                                neg_partial = torch.einsum(
                                    "nmjk, mpkl->nmpjl",
                                    Vin_i,
                                    Gn_shifted[:, :, j, :, :],
                                )
                                partial.sub_(neg_partial)
                                del neg_partial
                        else:
                            partial = torch.einsum(
                                "nmjk, mpkl->nmpjl",
                                Vin_i,
                                G_shifted[:, :, j, :, :],
                            )
                    if not adc_already_applied:
                        self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)

                    if not adc_already_applied:
                        token = self._profile_start("adc")
                        radc_j = self.radc[j] if self.radc_is_list else self.radc
                        partial.div_(adcRef)
                        partial.mul_(radc_j - 1)
                        partial.round_()
                        partial.div_(radc_j - 1)
                        self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                    token = self._profile_start("scale_accumulate")
                    partial.mul_(slice_scale[i, j])

                    if accumulated is None:
                        accumulated = partial
                    else:
                        accumulated.add_(partial)
                        del partial
                    self._profile_stop(token, c0=c0, c1=c1, input_slice=i, weight_slice=j)
                if differential_input:
                    del Vin_p_i, Vin_n_i
                else:
                    del Vin_i

            if self.mode == 2:
                del Gp_shifted, Gn_shifted
            else:
                del G_shifted

            token = self._profile_start("finalize")
            if len(x.shape) == 3:
                out_chunks.append(self._finalize_inference_chunk_3d(accumulated, x, mat, c0, c1))
            else:
                finalized = self._finalize_inference_chunk_2d(accumulated, x, mat, c0, c1)
                emit_chunk(finalized, c0, c1)
                del finalized
            self._profile_stop(token, c0=c0, c1=c1)
            del accumulated

        token = self._profile_start("concat_chunks")
        if direct_output is not None:
            out = direct_output
        elif len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = torch.cat(out_chunks, dim=-1)
        out = out[..., :mat.shape[1]]
        self._profile_stop(token, chunks=len(out_chunks))
        return out


    def _apply_adc_2d(self, out, adcRef):
        """
        Apply ADC quantization to a non-batched intermediate output.
        
        Parameters:
            out (torch.Tensor): Intermediate current tensor.
            adcRef (float or torch.Tensor): ADC reference current.
        
        Returns:
            torch.Tensor: ADC-quantized intermediate output.
        """
        if self.radc_is_list:
            return torch.round(out / adcRef * (self.radc.view(1, 1, 1, 1, -1, 1, 1) - 1)) / \
                   (self.radc.view(1, 1, 1, 1, -1, 1, 1) - 1)
        return torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)

    def _apply_adc_3d(self, out, adcRef):
        """
        Apply ADC quantization to a batched intermediate output.
        
        Parameters:
            out (torch.Tensor): Batched intermediate current tensor.
            adcRef (float or torch.Tensor): ADC reference current.
        
        Returns:
            torch.Tensor: ADC-quantized batched intermediate output.
        """
        if self.radc_is_list:
            return torch.round(out / adcRef * (self.radc.view(1, 1, 1, 1, 1, -1, 1, 1) - 1)) / \
                   (self.radc.view(1, 1, 1, 1, 1, -1, 1, 1) - 1)
        return torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)


    def _reconstruct_2d(self, out, x, mat, adcRef):
        """
        Reconstruct one non-batched output from slice products and scaling metadata.
        
        Parameters:
            out (torch.Tensor): ADC-normalized slice product tensor.
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            adcRef (float or torch.Tensor): ADC reference current used during quantization.
        
        Returns:
            torch.Tensor: Reconstructed 2-D matrix multiplication result.
        """
        QG = self.Q_G
        out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1))
        out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1))
               / QG / self.vread / (self.g_level - 1) * adcRef)
        sw = torch.zeros((len(x), len(mat)), device=x.device)
        for i in range(len(x)):
            sw[i] = x.sliced_weights[i] * mat.sliced_weights
        out = torch.mul(
            out.reshape(out.shape[0], out.shape[1], out.shape[2],
                        -1, out.shape[5], out.shape[6]),
            sw.reshape(1, 1, 1, -1, 1, 1),
        ).sum(dim=3)
        if x.bw_e is None:
            bm = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
            x_qmax = self._sliced_quant_qmax(x)
            mat_qmax = self._sliced_quant_qmax(mat)
            out = (out * bm
                   / x_qmax
                   / mat_qmax)
        else:
            eb = torch.einsum("nmij, mpij->nmpij", 2.0 ** x.e_bias, 2.0 ** mat.e_bias)
            out = out * eb * 2.0 ** (4 - sum(x.slice_method) - sum(mat.slice_method))
        out = out.sum(dim=1).permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
        return out[:x.shape[0], :mat.shape[1]]

    def _reconstruct_3d(self, out, x, mat, adcRef):
        """
        Reconstruct one batched output from slice products and scaling metadata.
        
        Parameters:
            out (torch.Tensor): ADC-normalized slice product tensor.
            x (SlicedDataMultiMode): Sliced input activation data.
            mat (SlicedDataMultiMode): Sliced weight data.
            adcRef (float or torch.Tensor): ADC reference current used during quantization.
        
        Returns:
            torch.Tensor: Reconstructed batched matrix multiplication result.
        """
        QG = self.Q_G
        out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
        out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1))
               / QG / self.vread / (self.g_level - 1) * adcRef)
        sw = torch.zeros((len(x), len(mat)), device=x.device)
        for i in range(len(x)):
            sw[i] = x.sliced_weights[i] * mat.sliced_weights
        out = torch.mul(
            out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3],
                        -1, out.shape[6], out.shape[7]),
            sw.reshape(1, 1, 1, 1, -1, 1, 1),
        ).sum(dim=4)
        if x.bw_e is None:
            bm = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat.max_data)
            x_qmax = self._sliced_quant_qmax(x)
            mat_qmax = self._sliced_quant_qmax(mat)
            out = (out * bm
                   / x_qmax
                   / mat_qmax)
        else:
            eb = torch.einsum("bnmij, mpij->bnmpij",
                               2.0 ** x.e_bias, 2.0 ** mat.e_bias)
            out = out * eb * 2.0 ** (4 - sum(x.slice_method) - sum(mat.slice_method))
        out = out.sum(dim=2).permute(0, 1, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2],
                          out.shape[3] * out.shape[4])
        return out[:out.shape[0], :x.shape[1], :mat.shape[1]]
