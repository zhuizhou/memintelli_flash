"""Optional Triton kernels for MemIntelli Flash v3 inference.

The kernels here are experimental and default-off. They target the profiled
fast inference bottleneck for 2-D Linear layers: current accumulation, optional
Mode-2 branch subtraction, ADC rounding, slice weighting, and weight-slice
reduction for one input slice and one output chunk.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - availability depends on the CUDA/Triton environment
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception as exc:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    TRITON_IMPORT_ERROR = exc
else:
    TRITON_IMPORT_ERROR = None


def is_triton_fast_accumulate_available() -> bool:
    """Return whether the optional Triton fast-accumulate kernel is importable."""
    return triton is not None


def triton_slice_mode0_2d_uniform1(
    x: torch.Tensor,
    *,
    input_slices: int,
    tile_cols: int,
    qmax: int,
    block_r: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-slice a 2-D mode-0 activation tensor for uniform 1-bit slices.

    This narrow helper targets the LLM inference case used in the paper:
    quantization granularity equals the array tile and the activation tile
    row is one token. It avoids materializing the padded/tiled float tensor in
    Python before the analog kernel consumes the slices.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x.dim() != 2:
        raise ValueError("Activation-slice Triton path expects a 2-D tensor.")
    if input_slices <= 0:
        raise ValueError("input_slices must be positive.")
    if tile_cols <= 0:
        raise ValueError("tile_cols must be positive.")
    x0 = x.contiguous()
    rows, cols = x0.shape
    tile_count = triton.cdiv(cols, tile_cols)
    sliced = torch.empty(
        (rows, tile_count, input_slices, 1, tile_cols),
        device=x0.device,
        dtype=torch.uint8,
    )
    max_data = torch.empty((rows, tile_count, 1, 1), device=x0.device, dtype=x0.dtype)
    grid = (triton.cdiv(rows, block_r), tile_count)
    _slice_mode0_2d_uniform1_kernel[grid](
        x0,
        sliced,
        max_data,
        x0.stride(0),
        x0.stride(1),
        sliced.stride(0),
        sliced.stride(1),
        sliced.stride(2),
        sliced.stride(3),
        sliced.stride(4),
        max_data.stride(0),
        max_data.stride(1),
        rows,
        cols,
        tile_count,
        tile_cols,
        input_slices,
        int(qmax),
        x0.dtype is torch.bfloat16,
        int(block_r),
        int(tile_cols),
        num_warps=4,
    )
    return sliced, max_data


def triton_slice_mode2_diff_2d_uniform(
    x: torch.Tensor,
    *,
    slice_bits: tuple[int, ...],
    tile_cols: int,
    quant_cols: int | None = None,
    qmax: int,
    block_r: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-slice a 2-D activation into Mode-2 differential branches.

    This is the Mode-2 counterpart of ``triton_slice_mode0_2d_uniform1`` for
    LLM inference: one-token row tiles, quantization granularity equal to the
    array tile, and unsigned positive/negative branch slices. It keeps the
    analog semantics unchanged; it only fuses activation tiling, abs-max, branch
    quantization, and bit extraction.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x.dim() != 2:
        raise ValueError("Mode-2 differential activation slicer expects a 2-D tensor.")
    if not slice_bits:
        raise ValueError("slice_bits must be non-empty.")
    if any(int(v) <= 0 or int(v) > 8 for v in slice_bits):
        raise ValueError("Mode-2 Triton activation slicer supports 1..8 bits per slice.")
    if tile_cols <= 0:
        raise ValueError("tile_cols must be positive.")
    if quant_cols is None:
        quant_cols = tile_cols
    if quant_cols <= 0 or quant_cols % tile_cols != 0:
        raise ValueError("quant_cols must be a positive multiple of tile_cols.")

    x0 = x.contiguous()
    rows, cols = x0.shape
    quant_count = triton.cdiv(cols, quant_cols)
    tiles_per_quant = int(quant_cols // tile_cols)
    tile_count = int(quant_count * tiles_per_quant)
    num_slices = len(slice_bits)
    sliced_p = torch.zeros(
        (rows, tile_count, num_slices, 1, tile_cols),
        device=x0.device,
        dtype=torch.uint8,
    )
    sliced_n = torch.zeros_like(sliced_p)
    max_data = torch.empty((rows, tile_count, 1, 1), device=x0.device, dtype=x0.dtype)
    bit_offsets = []
    offset = 0
    # data_formats extracts the least-significant slice first by walking
    # slice_method from the end.
    for bits in reversed(tuple(int(v) for v in slice_bits)):
        bit_offsets.append(offset)
        offset += bits
    bit_offsets_t = torch.tensor(bit_offsets, device=x0.device, dtype=torch.int32)
    slice_bits_t = torch.tensor(tuple(reversed(tuple(int(v) for v in slice_bits))), device=x0.device, dtype=torch.int32)

    grid = (triton.cdiv(rows, block_r), quant_count)
    _slice_mode2_diff_2d_uniform_kernel[grid](
        x0,
        sliced_p,
        sliced_n,
        max_data,
        bit_offsets_t,
        slice_bits_t,
        x0.stride(0),
        x0.stride(1),
        sliced_p.stride(0),
        sliced_p.stride(1),
        sliced_p.stride(2),
        sliced_p.stride(3),
        sliced_p.stride(4),
        max_data.stride(0),
        max_data.stride(1),
        rows,
        cols,
        tile_count,
        quant_count,
        tile_cols,
        int(quant_cols),
        tiles_per_quant,
        num_slices,
        int(qmax),
        x0.dtype is torch.bfloat16,
        int(block_r),
        int(quant_cols),
        num_warps=4,
    )
    return sliced_p, sliced_n, max_data


def triton_finalize_2d_tile_reduce(
    out: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    *,
    x_qmax: float,
    mat_qmax: float,
    block_r: int = 16,
    block_c: int = 32,
) -> torch.Tensor:
    """Finalize a 2-D inference chunk by scaling and reducing input tiles.

    The regular Python path materializes a broadcast scale tensor, multiplies
    the VMM output, reduces over input tiles, permutes, and reshapes. This
    helper fuses those steps for the standard quantized 2-D LLM path:

    out:     [N, M, P, J, L]
    x_max:   [N, M, 1, 1]
    mat_max: [M, P, 1, 1]
    result:  [N * J, P * L]
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if out.dim() != 5:
        raise ValueError("Finalize kernel expects out with shape [N, M, P, J, L].")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("Finalize kernel expects x_max/mat_max with rank 4.")
    o0 = out.contiguous()
    xm = x_max.contiguous()
    mm = mat_max.contiguous()
    n, m, p, j, l = o0.shape
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: out={tuple(o0.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: out={tuple(o0.shape)}, mat_max={tuple(mm.shape)}")

    result = torch.empty((n * j, p * l), device=o0.device, dtype=o0.dtype)
    grid = (triton.cdiv(n * j, block_r), triton.cdiv(p * l, block_c))
    _finalize_2d_tile_reduce_kernel[grid](
        o0,
        xm,
        mm,
        result,
        o0.stride(0),
        o0.stride(1),
        o0.stride(2),
        o0.stride(3),
        o0.stride(4),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        result.stride(0),
        result.stride(1),
        n,
        m,
        p,
        j,
        l,
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        int(block_r),
        int(block_c),
        num_warps=4,
    )
    return result


if triton is not None:

    @triton.jit
    def _round_even(x):
        return libdevice.nearbyint(x)


    @triton.jit
    def _slice_mode0_2d_uniform1_kernel(
        x,
        sliced,
        max_data,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        sd_s0: tl.constexpr,
        sd_s1: tl.constexpr,
        sd_s2: tl.constexpr,
        sd_s3: tl.constexpr,
        sd_s4: tl.constexpr,
        md_s0: tl.constexpr,
        md_s1: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        M: tl.constexpr,
        K: tl.constexpr,
        I: tl.constexpr,
        QMAX: tl.constexpr,
        ROUND_TO_BF16: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        cols = pid_m * K + offs_k
        mask = (offs_r[:, None] < R) & (offs_k[None, :] < K) & (cols[None, :] < C)
        vals = tl.load(
            x + offs_r[:, None] * x_s0 + cols[None, :] * x_s1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        abs_vals = tl.abs(vals)
        row_max = tl.max(abs_vals, axis=1)
        safe_max = tl.where(row_max > 0.0, row_max, 1.0)
        tl.store(
            max_data + offs_r * md_s0 + pid_m * md_s1,
            row_max,
            mask=offs_r < R,
        )
        scaled = vals / safe_max[:, None]
        if ROUND_TO_BF16:
            # Match the existing PyTorch BF16 activation-slicing path exactly:
            # mat / max_mat is rounded to BF16, then multiplying by qmax keeps
            # the BF16 arithmetic result before integer quantization.
            scaled = scaled.to(tl.bfloat16)
            scaled = (scaled * QMAX).to(tl.bfloat16).to(tl.float32)
        else:
            scaled = scaled * QMAX
        q = _round_even(scaled).to(tl.int32)
        q = q & ((1 << I) - 1)
        store_mask = (offs_r[:, None] < R) & (offs_k[None, :] < K) & (cols[None, :] < C)
        for i in tl.static_range(0, I):
            bits = (q >> i) & 1
            tl.store(
                sliced
                + offs_r[:, None] * sd_s0
                + pid_m * sd_s1
                + i * sd_s2
                + offs_k[None, :] * sd_s4,
                bits,
                mask=store_mask,
            )


    @triton.jit
    def _slice_mode2_diff_2d_uniform_kernel(
        x,
        sliced_p,
        sliced_n,
        max_data,
        bit_offsets,
        slice_bits,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        sd_s0: tl.constexpr,
        sd_s1: tl.constexpr,
        sd_s2: tl.constexpr,
        sd_s3: tl.constexpr,
        sd_s4: tl.constexpr,
        md_s0: tl.constexpr,
        md_s1: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        M: tl.constexpr,
        Q: tl.constexpr,
        K: tl.constexpr,
        QK: tl.constexpr,
        TILES_PER_Q: tl.constexpr,
        I: tl.constexpr,
        QMAX: tl.constexpr,
        ROUND_TO_BF16: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_q = tl.program_id(1)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        cols = pid_q * QK + offs_k
        mask = (offs_r[:, None] < R) & (offs_k[None, :] < QK) & (cols[None, :] < C)
        vals = tl.load(
            x + offs_r[:, None] * x_s0 + cols[None, :] * x_s1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        abs_vals = tl.abs(vals)
        row_max = tl.max(abs_vals, axis=1)
        safe_max = tl.where(row_max > 0.0, row_max, 1.0)
        for tile_i in tl.static_range(0, TILES_PER_Q):
            tile = pid_q * TILES_PER_Q + tile_i
            tl.store(
                max_data + offs_r * md_s0 + tile * md_s1,
                row_max,
                mask=(offs_r < R) & (tile < M),
            )
        pos = tl.where(vals > 0.0, vals, 0.0)
        neg = tl.where(vals < 0.0, -vals, 0.0)
        pos_scaled = pos / safe_max[:, None]
        neg_scaled = neg / safe_max[:, None]
        if ROUND_TO_BF16:
            pos_scaled = pos_scaled.to(tl.bfloat16)
            neg_scaled = neg_scaled.to(tl.bfloat16)
            pos_scaled = (pos_scaled * QMAX).to(tl.bfloat16).to(tl.float32)
            neg_scaled = (neg_scaled * QMAX).to(tl.bfloat16).to(tl.float32)
        else:
            pos_scaled = pos_scaled * QMAX
            neg_scaled = neg_scaled * QMAX
        q_pos = _round_even(pos_scaled).to(tl.int32)
        q_neg = _round_even(neg_scaled).to(tl.int32)
        q_pos = tl.maximum(0, tl.minimum(q_pos, QMAX))
        q_neg = tl.maximum(0, tl.minimum(q_neg, QMAX))
        store_mask = (offs_r[:, None] < R) & (offs_k[None, :] < QK) & (cols[None, :] < C)
        tile_idx = offs_k // K
        tile_k = offs_k - tile_idx * K
        out_tile = pid_q * TILES_PER_Q + tile_idx
        for i in tl.static_range(0, I):
            offset = tl.load(bit_offsets + i)
            bits = tl.load(slice_bits + i)
            mask_bits = (1 << bits) - 1
            sp = (q_pos >> offset) & mask_bits
            sn = (q_neg >> offset) & mask_bits
            tl.store(
                sliced_p
                + offs_r[:, None] * sd_s0
                + out_tile[None, :] * sd_s1
                + i * sd_s2
                + tile_k[None, :] * sd_s4,
                sp,
                mask=store_mask & (out_tile[None, :] < M),
            )
            tl.store(
                sliced_n
                + offs_r[:, None] * sd_s0
                + out_tile[None, :] * sd_s1
                + i * sd_s2
                + tile_k[None, :] * sd_s4,
                sn,
                mask=store_mask & (out_tile[None, :] < M),
            )


    @triton.jit
    def _finalize_2d_tile_reduce_kernel(
        out,
        x_max,
        mat_max,
        result,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        res_s0: tl.constexpr,
        res_s1: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        L: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        offs_p = offs_c // L
        offs_l = offs_c - offs_p * L
        mask = (offs_r[:, None] < (N * J)) & (offs_c[None, :] < (P * L))

        acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
        for m in tl.range(0, M):
            partial = tl.load(
                out
                + offs_n[:, None] * out_s0
                + m * out_s1
                + offs_p[None, :] * out_s2
                + offs_j[:, None] * out_s3
                + offs_l[None, :] * out_s4,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            xm = tl.load(
                x_max + offs_n * xm_s0 + m * xm_s1,
                mask=offs_r < (N * J),
                other=0.0,
            ).to(tl.float32)
            mm = tl.load(
                mat_max + m * mm_s0 + offs_p * mm_s1,
                mask=offs_c < (P * L),
                other=0.0,
            ).to(tl.float32)
            acc += partial * xm[:, None] * mm[None, :]

        acc *= SCALE
        tl.store(
            result + offs_r[:, None] * res_s0 + offs_c[None, :] * res_s1,
            acc,
            mask=mask,
        )


    @triton.jit
    def _restore_gidx_read_noise_kernel(
        idx,
        out,
        noise_offset_base,
        total: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        idx_f = tl.load(idx + offs, mask=mask, other=0).to(tl.float32)
        g_abs = LGS + idx_f * Q_G
        noise = tl.randn(NOISE_SEED, noise_offset_base + offs)
        shifted = g_abs * tl.exp(noise * READ_SIGMA) - LGS
        tl.store(out + offs, shifted, mask=mask)


    @triton.jit
    def _restore_mode2_gdiff_gidx_read_noise_kernel(
        gp_idx,
        gn_idx,
        out,
        noise_offset_base,
        total: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        gp_f = tl.load(gp_idx + offs, mask=mask, other=0).to(tl.float32)
        gn_f = tl.load(gn_idx + offs, mask=mask, other=0).to(tl.float32)
        if USE_READ_NOISE:
            gp_abs = LGS + gp_f * Q_G
            gn_abs = LGS + gn_f * Q_G
            gp_noise = tl.randn(NOISE_SEED, noise_offset_base + offs)
            gn_noise = tl.randn(NOISE_SEED, noise_offset_base + total + offs)
            gp_shifted = gp_abs * tl.exp(gp_noise * READ_SIGMA) - LGS
            gn_shifted = gn_abs * tl.exp(gn_noise * READ_SIGMA) - LGS
            gdiff = gp_shifted - gn_shifted
        else:
            gdiff = (gp_f - gn_f) * Q_G
        tl.store(out + offs, gdiff, mask=mask)


    @triton.jit
    def _fast_accumulate_2d_kernel(
        vin,
        g0,
        g1,
        scale,
        out,
        vin_s0: tl.constexpr,
        vin_s1: tl.constexpr,
        vin_s2: tl.constexpr,
        vin_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        ADC_REF_UNIT: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        MODE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                v = tl.load(
                    vin
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                w0 = tl.load(
                    g0
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                if MODE == 2:
                    w1 = tl.load(
                        g1
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    w0 = w0 - w1
                cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

            q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += q * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_kernel(
        vin,
        gidx0,
        gidx1,
        scale,
        out,
        vin_s0: tl.constexpr,
        vin_s1: tl.constexpr,
        vin_s2: tl.constexpr,
        vin_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        Q_G: tl.constexpr,
        ADC_REF_UNIT: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        MODE: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                v = tl.load(
                    vin
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                idx0 = tl.load(
                    gidx0
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                w0 = idx0 * Q_G
                if MODE == 2:
                    idx1 = tl.load(
                        gidx1
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    w0 = w0 - idx1 * Q_G
                if DOT_DTYPE == 1:
                    w0 = w0.to(tl.float16)
                elif DOT_DTYPE == 2:
                    w0 = w0.to(tl.bfloat16)
                cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

            q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += q * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_kernel(
        vin,
        gidx0,
        scale,
        out,
        vin_s0: tl.constexpr,
        vin_s1: tl.constexpr,
        vin_s2: tl.constexpr,
        vin_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        Q_G: tl.constexpr,
        ADC_REF: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                v = tl.load(
                    vin
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                idx0 = tl.load(
                    gidx0
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                w0 = idx0 * Q_G
                if DOT_DTYPE == 1:
                    w0 = w0.to(tl.float16)
                elif DOT_DTYPE == 2:
                    w0 = w0.to(tl.bfloat16)
                cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

            q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += q * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_read_noise_kernel(
        vin,
        gidx0,
        scale,
        out,
        vin_s0: tl.constexpr,
        vin_s1: tl.constexpr,
        vin_s2: tl.constexpr,
        vin_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                v = tl.load(
                    vin
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                idx0 = tl.load(
                    gidx0
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                g_abs = LGS + idx0 * Q_G
                noise_offset = (
                    (((pid_m * P + pid_p) * S + s) * K + k[:, None]) * L
                    + offs_l[None, :]
                )
                noise = tl.randn(NOISE_SEED, NOISE_OFFSET_BASE + noise_offset)
                w0 = g_abs * tl.exp(noise * READ_SIGMA) - LGS
                if DOT_DTYPE == 1:
                    w0 = w0.to(tl.float16)
                elif DOT_DTYPE == 2:
                    w0 = w0.to(tl.bfloat16)
                cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

            q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += q * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_input_slices_kernel(
        x_sliced,
        gidx0,
        x_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        Q_G: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    idx0 = tl.load(
                        gidx0
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    w0 = idx0 * Q_G
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                total += q * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_input_slices_read_noise_kernel(
        x_sliced,
        gidx0,
        x_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    idx0 = tl.load(
                        gidx0
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    g_abs = LGS + idx0 * Q_G
                    noise_offset = (
                        (((pid_m * P + pid_p) * S + s) * K + k[:, None]) * L
                        + offs_l[None, :]
                    )
                    noise = tl.randn(NOISE_SEED, NOISE_OFFSET_BASE + noise_offset)
                    w0 = g_abs * tl.exp(noise * READ_SIGMA) - LGS
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                total += q * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_input_slices_direct_final_kernel(
        x_sliced,
        gidx0,
        x_slice_max,
        scale,
        x_max,
        mat_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        OUT_COL_OFFSET,
        OUT_COLS,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        FINAL_SCALE: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + m * xm_s1,
            mask=mask_r,
            other=0.0,
        ).to(tl.float32)
        mat_tile_max = tl.load(
            mat_max + m * mm_s0 + pid_p * mm_s1,
            mask=pid_p < P,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    idx0 = tl.load(
                        gidx0
                        + m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if USE_READ_NOISE:
                        g_abs = LGS + idx0 * Q_G
                        noise_offset = (
                            (((m * P + pid_p) * S + s) * K + k[:, None]) * L
                            + offs_l[None, :]
                        )
                        noise = tl.randn(NOISE_SEED, NOISE_OFFSET_BASE + noise_offset)
                        w0 = g_abs * tl.exp(noise * READ_SIGMA) - LGS
                    else:
                        w0 = idx0 * Q_G
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += q * scale_is

        tile *= x_tile_max[:, None] * mat_tile_max * FINAL_SCALE
        out_cols = OUT_COL_OFFSET + pid_p * L + offs_l
        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            tile,
            mask=mask_r[:, None] & mask_l[None, :] & (out_cols[None, :] < OUT_COLS),
            sem="relaxed",
        )


    @triton.jit
    def _gidx_accumulate_2d_mode0_input_slices_read_noise_reuse_v_kernel(
        x_sliced,
        gidx0,
        x_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_max + i).to(tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                x_raw = tl.load(
                    x_sliced
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + i * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                if DOT_DTYPE == 1:
                    v = v.to(tl.float16)
                elif DOT_DTYPE == 2:
                    v = v.to(tl.bfloat16)
                for s in tl.static_range(0, S):
                    idx0 = tl.load(
                        gidx0
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    g_abs = LGS + idx0 * Q_G
                    noise_offset = (
                        (((pid_m * P + pid_p) * S + s) * K + k[:, None]) * L
                        + offs_l[None, :]
                    )
                    noise = tl.randn(NOISE_SEED, NOISE_OFFSET_BASE + noise_offset)
                    w0 = g_abs * tl.exp(noise * READ_SIGMA) - LGS
                    if DOT_DTYPE == 1:
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        w0 = w0.to(tl.bfloat16)
                    cur = tl.dot(v, w0, input_precision=INPUT_PRECISION)
                    q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                    scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                    total += q * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _fast_accumulate_2d_input_slices_kernel(
        x_sliced,
        g0,
        x_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    w0 = tl.load(
                        g0
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                total += q * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _fast_accumulate_2d_input_slices_direct_final_kernel(
        x_sliced,
        g0,
        x_slice_max,
        scale,
        x_max,
        mat_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        OUT_COL_OFFSET,
        OUT_COLS,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        FINAL_SCALE: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + m * xm_s1,
            mask=mask_r,
            other=0.0,
        ).to(tl.float32)
        mat_tile_max = tl.load(
            mat_max + m * mm_s0 + pid_p * mm_s1,
            mask=pid_p < P,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    w0 = tl.load(
                        g0
                        + m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += q * scale_is

        tile *= x_tile_max[:, None] * mat_tile_max * FINAL_SCALE
        out_cols = OUT_COL_OFFSET + pid_p * L + offs_l
        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            tile,
            mask=mask_r[:, None] & mask_l[None, :] & (out_cols[None, :] < OUT_COLS),
            sem="relaxed",
        )


    @triton.jit
    def _fast_accumulate_2d_input_slices_reuse_v_kernel(
        x_sliced,
        g0,
        x_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        DOT_DTYPE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_max + i).to(tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                x_raw = tl.load(
                    x_sliced
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + i * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                if DOT_DTYPE == 1:
                    v = v.to(tl.float16)
                elif DOT_DTYPE == 2:
                    v = v.to(tl.bfloat16)
                for s in tl.static_range(0, S):
                    w0 = tl.load(
                        g0
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    if DOT_DTYPE == 1:
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        w0 = w0.to(tl.bfloat16)
                    cur = tl.dot(v, w0, input_precision=INPUT_PRECISION)
                    q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                    scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                    total += q * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_kernel(
        vin_p,
        vin_n,
        gp,
        gn,
        scale,
        out,
        vin_s0: tl.constexpr,
        vin_s1: tl.constexpr,
        vin_s2: tl.constexpr,
        vin_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        ADC_REF: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                vp = tl.load(
                    vin_p
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                vn = tl.load(
                    vin_n
                    + offs_n[:, None] * vin_s0
                    + pid_m * vin_s1
                    + offs_j[:, None] * vin_s2
                    + k[None, :] * vin_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                )
                w = tl.load(
                    gp
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                wn = tl.load(
                    gn
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                w = w - wn
                cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

            q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += (q_p - q_n) * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_kernel(
        x_p,
        x_n,
        gp,
        gn,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        XMAX: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                xp = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp = _round_even(xp / XMAX * RDAC_SCALE) * VREAD_SCALE
                vn = _round_even(xn / XMAX * RDAC_SCALE) * VREAD_SCALE
                w = tl.load(
                    gp
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                wn = tl.load(
                    gn
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                w = w - wn
                cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

            q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += (q_p - q_n) * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_gdiff_kernel(
        x_p,
        x_n,
        gdiff,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        XMAX: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                xp = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp = _round_even(xp / XMAX * RDAC_SCALE) * VREAD_SCALE
                vn = _round_even(xn / XMAX * RDAC_SCALE) * VREAD_SCALE
                w = tl.load(
                    gdiff
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

            q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += (q_p - q_n) * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_all_input_slices_kernel(
        x_p,
        x_n,
        gp,
        gn,
        x_slice_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    xp = tl.load(
                        x_p
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    xn = tl.load(
                        x_n
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    vp = _round_even(xp / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    vn = _round_even(xn / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    w = tl.load(
                        gp
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    wn = tl.load(
                        gn
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    w = w - wn
                    cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                    cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

                q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
                q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                total += (q_p - q_n) * scale_is

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_all_input_slices_i3_reuse_g_kernel(
        x_p,
        x_n,
        gp,
        gn,
        x_slice_max,
        scale,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        xmax0 = tl.load(x_slice_max + 0).to(tl.float32)
        xmax1 = tl.load(x_slice_max + 1).to(tl.float32)
        xmax2 = tl.load(x_slice_max + 2).to(tl.float32)
        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)

        for s in tl.static_range(0, S):
            cur_p0 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n0 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_p1 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n1 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_p2 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n2 = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                w = tl.load(
                    gp
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                wn = tl.load(
                    gn
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                )
                w = w - wn

                xp0 = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 0 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn0 = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 0 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp0 = _round_even(xp0 / xmax0 * RDAC_SCALE) * VREAD_SCALE
                vn0 = _round_even(xn0 / xmax0 * RDAC_SCALE) * VREAD_SCALE
                cur_p0 += tl.dot(vp0, w, input_precision=INPUT_PRECISION)
                cur_n0 += tl.dot(vn0, w, input_precision=INPUT_PRECISION)

                xp1 = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 1 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn1 = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 1 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp1 = _round_even(xp1 / xmax1 * RDAC_SCALE) * VREAD_SCALE
                vn1 = _round_even(xn1 / xmax1 * RDAC_SCALE) * VREAD_SCALE
                cur_p1 += tl.dot(vp1, w, input_precision=INPUT_PRECISION)
                cur_n1 += tl.dot(vn1, w, input_precision=INPUT_PRECISION)

                xp2 = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 2 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn2 = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + 2 * x_s2
                    + offs_j[:, None] * x_s3
                    + k[None, :] * x_s4,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp2 = _round_even(xp2 / xmax2 * RDAC_SCALE) * VREAD_SCALE
                vn2 = _round_even(xn2 / xmax2 * RDAC_SCALE) * VREAD_SCALE
                cur_p2 += tl.dot(vp2, w, input_precision=INPUT_PRECISION)
                cur_n2 += tl.dot(vn2, w, input_precision=INPUT_PRECISION)

            q_p0 = _round_even(cur_p0 / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n0 = _round_even(cur_n0 / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_p1 = _round_even(cur_p1 / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n1 = _round_even(cur_n1 / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_p2 = _round_even(cur_p2 / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n2 = _round_even(cur_n2 / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale0 = tl.load(scale + 0 * scale_s0 + s * scale_s1)
            scale1 = tl.load(scale + 1 * scale_s0 + s * scale_s1)
            scale2 = tl.load(scale + 2 * scale_s0 + s * scale_s1)
            total += (q_p0 - q_n0) * scale0
            total += (q_p1 - q_n1) * scale1
            total += (q_p2 - q_n2) * scale2

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_gidx_kernel(
        x_p,
        x_n,
        gp_idx,
        gn_idx,
        scale,
        out,
        noise_offset_base,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        out_s4: tl.constexpr,
        N: tl.constexpr,
        M: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        XMAX: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid_r = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_pl = tl.program_id(2)
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(0, S):
            cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, K, BLOCK_K):
                k = k0 + offs_k
                mask_k = k < K
                xp = tl.load(
                    x_p
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                xn = tl.load(
                    x_n
                    + offs_n[:, None] * x_s0
                    + pid_m * x_s1
                    + offs_j[:, None] * x_s2
                    + k[None, :] * x_s3,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                vp = _round_even(xp / XMAX * RDAC_SCALE) * VREAD_SCALE
                vn = _round_even(xn / XMAX * RDAC_SCALE) * VREAD_SCALE
                gp = tl.load(
                    gp_idx
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                gn = tl.load(
                    gn_idx
                    + pid_m * g_s0
                    + pid_p * g_s1
                    + s * g_s2
                    + k[:, None] * g_s3
                    + offs_l[None, :] * g_s4,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                if USE_READ_NOISE:
                    elem = (((pid_m * P + pid_p) * S + s) * K + k[:, None]) * L + offs_l[None, :]
                    gp_abs = LGS + gp * Q_G
                    gn_abs = LGS + gn * Q_G
                    gp_shift = gp_abs * tl.exp(tl.randn(NOISE_SEED, noise_offset_base + 2 * elem) * READ_SIGMA) - LGS
                    gn_shift = gn_abs * tl.exp(tl.randn(NOISE_SEED, noise_offset_base + 2 * elem + 1) * READ_SIGMA) - LGS
                    w = gp_shift - gn_shift
                else:
                    w = (gp - gn) * Q_G
                cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

            q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
            q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
            scale_s = tl.load(scale + s)
            total += (q_p - q_n) * scale_s

        tl.store(
            out
            + offs_n[:, None] * out_s0
            + pid_m * out_s1
            + pid_p * out_s2
            + offs_j[:, None] * out_s3
            + offs_l[None, :] * out_s4,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_direct_final_kernel(
        x_p,
        x_n,
        gp,
        gn,
        x_slice_max,
        scale,
        x_max,
        mat_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        OUT_COL_OFFSET,
        OUT_COLS,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        FINAL_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + pid_m * xm_s1,
            mask=mask_r,
            other=0.0,
        ).to(tl.float32)
        mat_tile_max = tl.load(
            mat_max + pid_m * mm_s0 + pid_p * mm_s1,
            mask=pid_p < P,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    xp = tl.load(
                        x_p
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    xn = tl.load(
                        x_n
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    vp = _round_even(xp / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    vn = _round_even(xn / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    w = tl.load(
                        gp
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    wn = tl.load(
                        gn
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    w = w - wn
                    cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                    cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

                q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
                q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += (q_p - q_n) * scale_is

        tile *= x_tile_max[:, None] * mat_tile_max * FINAL_SCALE
        out_cols = OUT_COL_OFFSET + pid_p * L + offs_l
        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            tile,
            mask=mask_r[:, None] & mask_l[None, :] & (out_cols[None, :] < OUT_COLS),
            sem="relaxed",
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_gidx_direct_final_kernel(
        x_p,
        x_n,
        gp_idx,
        gn_idx,
        x_slice_max,
        scale,
        x_max,
        mat_max,
        out,
        noise_offset_base,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        OUT_COL_OFFSET,
        OUT_COLS,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        FINAL_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + pid_m * xm_s1,
            mask=mask_r,
            other=0.0,
        ).to(tl.float32)
        mat_tile_max = tl.load(
            mat_max + pid_m * mm_s0 + pid_p * mm_s1,
            mask=pid_p < P,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            safe_xmax_i = tl.where(xmax_i > 0.0, xmax_i, 1.0)
            for s in tl.static_range(0, S):
                cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    xp = tl.load(
                        x_p
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    xn = tl.load(
                        x_n
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    vp = _round_even(xp / safe_xmax_i * RDAC_SCALE) * VREAD_SCALE
                    vn = _round_even(xn / safe_xmax_i * RDAC_SCALE) * VREAD_SCALE
                    gp = tl.load(
                        gp_idx
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    gn = tl.load(
                        gn_idx
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if USE_READ_NOISE:
                        elem = (((pid_m * P + pid_p) * S + s) * K + k[:, None]) * L + offs_l[None, :]
                        gp_abs = LGS + gp * Q_G
                        gn_abs = LGS + gn * Q_G
                        gp_shift = gp_abs * tl.exp(tl.randn(NOISE_SEED, noise_offset_base + 2 * elem) * READ_SIGMA) - LGS
                        gn_shift = gn_abs * tl.exp(tl.randn(NOISE_SEED, noise_offset_base + 2 * elem + 1) * READ_SIGMA) - LGS
                        w = gp_shift - gn_shift
                    else:
                        w = (gp - gn) * Q_G
                    cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                    cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

                q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
                q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += (q_p - q_n) * scale_is

        tile *= x_tile_max[:, None] * mat_tile_max * FINAL_SCALE
        out_cols = OUT_COL_OFFSET + pid_p * L + offs_l
        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            tile,
            mask=mask_r[:, None] & mask_l[None, :] & (out_cols[None, :] < OUT_COLS),
            sem="relaxed",
        )


    @triton.jit
    def _diff_input_accumulate_2d_from_slices_gdiff_direct_final_kernel(
        x_p,
        x_n,
        gdiff,
        x_slice_max,
        scale,
        x_max,
        mat_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        x_s4: tl.constexpr,
        g_s0: tl.constexpr,
        g_s1: tl.constexpr,
        g_s2: tl.constexpr,
        g_s3: tl.constexpr,
        g_s4: tl.constexpr,
        scale_s0: tl.constexpr,
        scale_s1: tl.constexpr,
        xm_s0: tl.constexpr,
        xm_s1: tl.constexpr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        OUT_COL_OFFSET,
        OUT_COLS,
        N: tl.constexpr,
        M: tl.constexpr,
        I: tl.constexpr,
        P: tl.constexpr,
        J: tl.constexpr,
        K: tl.constexpr,
        L: tl.constexpr,
        S: tl.constexpr,
        VREAD_SCALE: tl.constexpr,
        ADC_REF: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        FINAL_SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_mpl = pid // NUM_R_BLOCKS
        pid_m = pid_mpl % M
        pid_pl = pid_mpl // M
        pid_p = pid_pl // NUM_L_BLOCKS
        pid_l = pid_pl - pid_p * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = offs_r // J
        offs_j = offs_r - offs_n * J
        mask_r = offs_r < (N * J)
        mask_l = offs_l < L

        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + pid_m * xm_s1,
            mask=mask_r,
            other=0.0,
        ).to(tl.float32)
        mat_tile_max = tl.load(
            mat_max + pid_m * mm_s0 + pid_p * mm_s1,
            mask=pid_p < P,
            other=0.0,
        ).to(tl.float32)

        for i in tl.static_range(0, I):
            xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            for s in tl.static_range(0, S):
                cur_p = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                cur_n = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    xp = tl.load(
                        x_p
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    xn = tl.load(
                        x_n
                        + offs_n[:, None] * x_s0
                        + pid_m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    vp = _round_even(xp / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    vn = _round_even(xn / xmax_i * RDAC_SCALE) * VREAD_SCALE
                    w = tl.load(
                        gdiff
                        + pid_m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    cur_p += tl.dot(vp, w, input_precision=INPUT_PRECISION)
                    cur_n += tl.dot(vn, w, input_precision=INPUT_PRECISION)

                q_p = _round_even(cur_p / ADC_REF * RADC_SCALE) / RADC_SCALE
                q_n = _round_even(cur_n / ADC_REF * RADC_SCALE) / RADC_SCALE
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += (q_p - q_n) * scale_is

        tile *= x_tile_max[:, None] * mat_tile_max * FINAL_SCALE
        out_cols = OUT_COL_OFFSET + pid_p * L + offs_l
        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            tile,
            mask=mask_r[:, None] & mask_l[None, :] & (out_cols[None, :] < OUT_COLS),
            sem="relaxed",
        )


    @triton.jit
    def _mode1_gidx_direct_final_kernel(
        x,
        gp_idx,
        gn_idx,
        w_scale,
        x_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        gp_s0: tl.constexpr,
        gp_s1: tl.constexpr,
        gn_s0: tl.constexpr,
        gn_s1: tl.constexpr,
        ws_s0: tl.constexpr,
        ws_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        O: tl.constexpr,
        TILE_IN: tl.constexpr,
        TILE_OUT: tl.constexpr,
        OUT_TILE_COUNT: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF_UNIT: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        C2W_DENOM: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
        NUM_IN_TILES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_rest = pid // NUM_R_BLOCKS
        pid_it = pid_rest % NUM_IN_TILES
        pid_ol = pid_rest // NUM_IN_TILES
        pid_ot = pid_ol // NUM_L_BLOCKS
        pid_l = pid_ol - pid_ot * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        out_cols = pid_ot * TILE_OUT + offs_l
        offs_k = tl.arange(0, BLOCK_K)
        mask_r = offs_r < R
        mask_l = out_cols < O

        x_max_value = tl.load(x_max).to(tl.float32)
        safe_xmax = tl.where(x_max_value > 0.0, x_max_value, 1.0)
        r0 = pid_it * TILE_IN
        tile_width = tl.minimum(TILE_IN, C - r0)
        adc_ref_tile = ADC_REF_UNIT * tile_width
        cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        for k0 in tl.static_range(0, TILE_IN, BLOCK_K):
            k_rel = k0 + offs_k
            k_abs = r0 + k_rel
            mask_k = (k_rel < TILE_IN) & (k_abs < C)
            x_raw = tl.load(
                x + offs_r[:, None] * x_s0 + k_abs[None, :] * x_s1,
                mask=mask_r[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            v = _round_even(x_raw / safe_xmax * RDAC_SCALE) * (VREAD / RDAC_SCALE)
            gp = tl.load(
                gp_idx + k_abs[:, None] * gp_s0 + out_cols[None, :] * gp_s1,
                mask=mask_k[:, None] & mask_l[None, :],
                other=0.0,
            ).to(tl.float32)
            gn = tl.load(
                gn_idx + k_abs[:, None] * gn_s0 + out_cols[None, :] * gn_s1,
                mask=mask_k[:, None] & mask_l[None, :],
                other=0.0,
            ).to(tl.float32)
            if USE_READ_NOISE:
                noise_base_p = NOISE_OFFSET_BASE + 2 * (
                    k_abs[:, None] * O + out_cols[None, :]
                )
                noise_base_n = noise_base_p + 1
                gp_abs = LGS + gp * Q_G
                gn_abs = LGS + gn * Q_G
                gp_shift = gp_abs * tl.exp(tl.randn(NOISE_SEED, noise_base_p) * READ_SIGMA) - LGS
                gn_shift = gn_abs * tl.exp(tl.randn(NOISE_SEED, noise_base_n) * READ_SIGMA) - LGS
                w = gp_shift - gn_shift
            else:
                w = (gp - gn) * Q_G
            cur += tl.dot(v, w, input_precision=INPUT_PRECISION)

        q = _round_even(cur / adc_ref_tile * RADC_SCALE) / RADC_SCALE
        scale_val = tl.load(
            w_scale + pid_it * ws_s0 + pid_ot * ws_s1,
            mask=(pid_it < NUM_IN_TILES) & (pid_ot < OUT_TILE_COUNT),
            other=1.0,
        ).to(tl.float32)
        total = q * (adc_ref_tile * scale_val * (safe_xmax / C2W_DENOM))

        tl.atomic_add(
            out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
            total,
            mask=mask_r[:, None] & mask_l[None, :],
            sem="relaxed",
        )


    @triton.jit
    def _mode1_gidx_direct_final_grouped_kernel(
        x,
        gp_idx,
        gn_idx,
        w_scale,
        x_max,
        out,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        gp_s0: tl.constexpr,
        gp_s1: tl.constexpr,
        gn_s0: tl.constexpr,
        gn_s1: tl.constexpr,
        ws_s0: tl.constexpr,
        ws_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
        O: tl.constexpr,
        TILE_IN: tl.constexpr,
        TILE_OUT: tl.constexpr,
        OUT_TILE_COUNT: tl.constexpr,
        LGS: tl.constexpr,
        Q_G: tl.constexpr,
        READ_SIGMA: tl.constexpr,
        NOISE_SEED: tl.constexpr,
        NOISE_OFFSET_BASE,
        ADC_REF_UNIT: tl.constexpr,
        RDAC_SCALE: tl.constexpr,
        RADC_SCALE: tl.constexpr,
        VREAD: tl.constexpr,
        C2W_DENOM: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_READ_NOISE: tl.constexpr,
        USE_ATOMIC: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_R_BLOCKS: tl.constexpr,
        NUM_L_BLOCKS: tl.constexpr,
        NUM_IN_TILES: tl.constexpr,
        NUM_IN_TILE_GROUPS: tl.constexpr,
        GROUP_IN_TILES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_r = pid % NUM_R_BLOCKS
        pid_rest = pid // NUM_R_BLOCKS
        pid_ig = pid_rest % NUM_IN_TILE_GROUPS
        pid_ol = pid_rest // NUM_IN_TILE_GROUPS
        pid_ot = pid_ol // NUM_L_BLOCKS
        pid_l = pid_ol - pid_ot * NUM_L_BLOCKS

        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        out_cols = pid_ot * TILE_OUT + offs_l
        offs_k = tl.arange(0, BLOCK_K)
        mask_r = offs_r < R
        mask_l = out_cols < O

        x_max_value = tl.load(x_max).to(tl.float32)
        safe_xmax = tl.where(x_max_value > 0.0, x_max_value, 1.0)
        tile_total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)

        base_it = pid_ig * GROUP_IN_TILES
        for gi in tl.static_range(0, GROUP_IN_TILES):
            pid_it = base_it + gi
            valid_it = pid_it < NUM_IN_TILES
            r0 = pid_it * TILE_IN
            tile_width = tl.minimum(TILE_IN, C - r0)
            tile_width = tl.where(valid_it & (tile_width > 0), tile_width, 0)
            adc_ref_tile = ADC_REF_UNIT * tile_width
            safe_adc_ref = tl.where(adc_ref_tile > 0.0, adc_ref_tile, 1.0)

            cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
            for k0 in tl.static_range(0, TILE_IN, BLOCK_K):
                k_rel = k0 + offs_k
                k_abs = r0 + k_rel
                mask_k = valid_it & (k_rel < TILE_IN) & (k_abs < C)
                x_raw = tl.load(
                    x + offs_r[:, None] * x_s0 + k_abs[None, :] * x_s1,
                    mask=mask_r[:, None] & mask_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = _round_even(x_raw / safe_xmax * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                gp = tl.load(
                    gp_idx + k_abs[:, None] * gp_s0 + out_cols[None, :] * gp_s1,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                gn = tl.load(
                    gn_idx + k_abs[:, None] * gn_s0 + out_cols[None, :] * gn_s1,
                    mask=mask_k[:, None] & mask_l[None, :],
                    other=0.0,
                ).to(tl.float32)
                if USE_READ_NOISE:
                    noise_base_p = NOISE_OFFSET_BASE + 2 * (
                        k_abs[:, None] * O + out_cols[None, :]
                    )
                    noise_base_n = noise_base_p + 1
                    gp_abs = LGS + gp * Q_G
                    gn_abs = LGS + gn * Q_G
                    gp_shift = gp_abs * tl.exp(tl.randn(NOISE_SEED, noise_base_p) * READ_SIGMA) - LGS
                    gn_shift = gn_abs * tl.exp(tl.randn(NOISE_SEED, noise_base_n) * READ_SIGMA) - LGS
                    w = gp_shift - gn_shift
                else:
                    w = (gp - gn) * Q_G
                cur += tl.dot(v, w, input_precision=INPUT_PRECISION)

            q = _round_even(cur / safe_adc_ref * RADC_SCALE) / RADC_SCALE
            scale_val = tl.load(
                w_scale + pid_it * ws_s0 + pid_ot * ws_s1,
                mask=valid_it & (pid_ot < OUT_TILE_COUNT),
                other=0.0,
            ).to(tl.float32)
            tile_total += q * (adc_ref_tile * scale_val * (safe_xmax / C2W_DENOM))

        if USE_ATOMIC:
            tl.atomic_add(
                out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
                tile_total,
                mask=mask_r[:, None] & mask_l[None, :],
                sem="relaxed",
            )
        else:
            tl.store(
                out + offs_r[:, None] * out_s0 + out_cols[None, :] * out_s1,
                tile_total,
                mask=mask_r[:, None] & mask_l[None, :],
            )


def triton_restore_gidx_read_noise(
    idx: torch.Tensor,
    *,
    lgs: float,
    q_g: float,
    read_sigma: float,
    dtype: torch.dtype,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    block: int = 256,
) -> torch.Tensor:
    """Restore shifted conductance from level indices and apply uniform read variation."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if not idx.is_cuda:
        raise ValueError("Triton G-index restore requires a CUDA tensor.")
    idx_c = idx.contiguous()
    out = torch.empty(idx_c.shape, device=idx_c.device, dtype=dtype)
    total = idx_c.numel()
    grid = (triton.cdiv(total, block),)
    _restore_gidx_read_noise_kernel[grid](
        idx_c,
        out,
        int(noise_offset_base),
        total,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(block),
        num_warps=4,
    )
    return out


def triton_restore_mode2_gdiff_gidx_read_noise(
    gp_idx: torch.Tensor,
    gn_idx: torch.Tensor,
    *,
    lgs: float,
    q_g: float,
    read_sigma: float,
    dtype: torch.dtype,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    block: int = 256,
) -> torch.Tensor:
    """Restore shifted mode-2 Gp-Gn directly from compressed indices."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if not gp_idx.is_cuda or not gn_idx.is_cuda:
        raise ValueError("Triton mode-2 Gdiff restore requires CUDA index tensors.")
    if gp_idx.shape != gn_idx.shape:
        raise ValueError(f"Mode-2 Gdiff restore shape mismatch: gp={tuple(gp_idx.shape)}, gn={tuple(gn_idx.shape)}")
    gp_c = gp_idx.contiguous()
    gn_c = gn_idx.contiguous()
    out = torch.empty(gp_c.shape, device=gp_c.device, dtype=dtype)
    total = gp_c.numel()
    grid = (triton.cdiv(total, block),)
    _restore_mode2_gdiff_gidx_read_noise_kernel[grid](
        gp_c,
        gn_c,
        out,
        int(noise_offset_base),
        total,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        bool(read_sigma and read_sigma > 0.0),
        int(block),
        num_warps=4,
    )
    return out


def triton_fast_accumulate_2d(
    vin: torch.Tensor,
    g_shifted: torch.Tensor | None,
    slice_scale_row: torch.Tensor,
    adc_ref_unit: float,
    radc: int,
    mode: int,
    *,
    gp_shifted: torch.Tensor | None = None,
    gn_shifted: torch.Tensor | None = None,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the optional 2-D Triton fast-accumulate kernel."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if vin.dim() != 4:
        raise ValueError("Triton fast accumulate supports 2-D Linear tensors only.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    vin = vin.contiguous()
    slice_scale_row = slice_scale_row.contiguous()
    if mode == 2:
        if gp_shifted is None or gn_shifted is None:
            raise ValueError("Mode 2 Triton fast accumulate requires gp_shifted and gn_shifted.")
        g0 = gp_shifted.contiguous()
        g1 = gn_shifted.contiguous()
    else:
        if g_shifted is None:
            raise ValueError("Mode 0 Triton fast accumulate requires g_shifted.")
        g0 = g_shifted.contiguous()
        g1 = g0

    n, m, j, k = vin.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: vin={tuple(vin.shape)}, g={tuple(g0.shape)}")

    out = torch.empty((n, m, p, j, l), device=vin.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _fast_accumulate_2d_kernel[grid](
        vin,
        g0,
        g1,
        slice_scale_row,
        out,
        vin.stride(0),
        vin.stride(1),
        vin.stride(2),
        vin.stride(3),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(adc_ref_unit),
        float(radc - 1),
        int(mode),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_gidx_accumulate_2d(
    vin: torch.Tensor,
    gidx: torch.Tensor | None,
    slice_scale_row: torch.Tensor,
    q_g: float,
    adc_ref: float,
    radc: int,
    mode: int,
    *,
    gp_idx: torch.Tensor | None = None,
    gn_idx: torch.Tensor | None = None,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run a 2-D Triton kernel that decodes compressed G indices in-kernel."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if vin.dim() != 4:
        raise ValueError("Triton G-index accumulate supports 2-D Linear tensors only.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    vin = vin.contiguous()
    slice_scale_row = slice_scale_row.contiguous()
    if mode == 2:
        if gp_idx is None or gn_idx is None:
            raise ValueError("Mode 2 Triton G-index accumulate requires gp_idx and gn_idx.")
        g0 = gp_idx.contiguous()
        g1 = gn_idx.contiguous()
    else:
        if gidx is None:
            raise ValueError("Mode 0 Triton G-index accumulate requires gidx.")
        g0 = gidx.contiguous()
        g1 = g0

    n, m, j, k = vin.shape
    gm, p, s, gk, l = g0.shape
    if mode == 2 and g1.shape != g0.shape:
        raise ValueError(f"Conductance-index branch shape mismatch: gp={tuple(g0.shape)}, gn={tuple(g1.shape)}")
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: vin={tuple(vin.shape)}, gidx={tuple(g0.shape)}")

    out = torch.empty((n, m, p, j, l), device=vin.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if vin.dtype is torch.bfloat16 else 1 if vin.dtype is torch.float16 else 0
    if mode == 2:
        _gidx_accumulate_2d_kernel[grid](
            vin,
            g0,
            g1,
            slice_scale_row,
            out,
            vin.stride(0),
            vin.stride(1),
            vin.stride(2),
            vin.stride(3),
            g0.stride(0),
            g0.stride(1),
            g0.stride(2),
            g0.stride(3),
            g0.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            n,
            m,
            p,
            j,
            k,
            l,
            s,
            float(q_g),
            float(adc_ref),
            float(radc - 1),
            int(mode),
            dot_dtype,
            input_precision,
            int(block_r),
            int(block_l),
            int(block_k),
            num_r_blocks,
            num_l_blocks,
            num_warps=4,
        )
    else:
        _gidx_accumulate_2d_mode0_kernel[grid](
            vin,
            g0,
            slice_scale_row,
            out,
            vin.stride(0),
            vin.stride(1),
            vin.stride(2),
            vin.stride(3),
            g0.stride(0),
            g0.stride(1),
            g0.stride(2),
            g0.stride(3),
            g0.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            n,
            m,
            p,
            j,
            k,
            l,
            s,
            float(q_g),
            float(adc_ref),
            float(radc - 1),
            dot_dtype,
            input_precision,
            int(block_r),
            int(block_l),
            int(block_k),
            num_r_blocks,
            num_l_blocks,
            num_warps=4,
        )
    return out


def triton_gidx_accumulate_2d_read_noise(
    vin: torch.Tensor,
    gidx: torch.Tensor,
    slice_scale_row: torch.Tensor,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref_unit: float,
    radc: int,
    *,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run a mode-0 compressed G-index kernel with uniform read variation generated in-kernel."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if vin.dim() != 4:
        raise ValueError("Triton G-index read-noise accumulate supports 2-D Linear tensors only.")
    if gidx is None:
        raise ValueError("Mode 0 Triton G-index read-noise accumulate requires gidx.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    vin = vin.contiguous()
    g0 = gidx.contiguous()
    slice_scale_row = slice_scale_row.contiguous()

    n, m, j, k = vin.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: vin={tuple(vin.shape)}, gidx={tuple(g0.shape)}")

    out = torch.empty((n, m, p, j, l), device=vin.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if vin.dtype is torch.bfloat16 else 1 if vin.dtype is torch.float16 else 0
    _gidx_accumulate_2d_mode0_read_noise_kernel[grid](
        vin,
        g0,
        slice_scale_row,
        out,
        vin.stride(0),
        vin.stride(1),
        vin.stride(2),
        vin.stride(3),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(noise_offset_base),
        float(adc_ref_unit),
        float(radc - 1),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gp_shifted: torch.Tensor,
    gn_shifted: torch.Tensor,
    slice_scale_row: torch.Tensor,
    xmax: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the 2-D differential-input kernel with DAC quantization fused."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 4 or x_n.dim() != 4:
        raise ValueError("Triton fused-Vin differential accumulate supports 2-D Linear tensors only.")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    gp = gp_shifted.contiguous()
    gn = gn_shifted.contiguous()
    slice_scale_row = slice_scale_row.contiguous()

    n, m, j, k = x_p.shape
    gm, p, s, gk, l = gp.shape
    if gn.shape != gp.shape:
        raise ValueError(f"Conductance branch shape mismatch: gp={tuple(gp.shape)}, gn={tuple(gn.shape)}")
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(x_p.shape)}, g={tuple(gp.shape)}")

    out = torch.empty((n, m, p, j, l), device=x_p.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _diff_input_accumulate_2d_from_slices_kernel[grid](
        x_p,
        x_n,
        gp,
        gn,
        slice_scale_row,
        out,
        x_p.stride(0),
        x_p.stride(1),
        x_p.stride(2),
        x_p.stride(3),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(xmax),
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices_gdiff(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gdiff_shifted: torch.Tensor,
    slice_scale_row: torch.Tensor,
    xmax: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run mode-2 differential-input fused-DAC accumulation from pre-subtracted G."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 4 or x_n.dim() != 4:
        raise ValueError("Triton fused-Vin differential accumulate supports 2-D Linear tensors only.")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gdiff_shifted.dim() != 5:
        raise ValueError("gdiff_shifted must have shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    gd = gdiff_shifted.contiguous()
    scale = slice_scale_row.contiguous()
    n, m, j, k = x_p.shape
    gm, p, s, gk, l = gd.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(x_p.shape)}, gdiff={tuple(gd.shape)}")

    out = torch.empty((n, m, p, j, l), device=x_p.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _diff_input_accumulate_2d_from_slices_gdiff_kernel[grid](
        x_p,
        x_n,
        gd,
        scale,
        out,
        x_p.stride(0),
        x_p.stride(1),
        x_p.stride(2),
        x_p.stride(3),
        gd.stride(0),
        gd.stride(1),
        gd.stride(2),
        gd.stride(3),
        gd.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(xmax),
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_all_input_slices(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gp_shifted: torch.Tensor,
    gn_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run mode-2 differential-input accumulation for all input slices in one launch."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 5 or x_n.dim() != 5:
        raise ValueError("Mode-2 all-input-slice kernel expects x_p/x_n with shape [N, M, I, J, K].")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gp_shifted is None or gn_shifted is None or gp_shifted.dim() != 5 or gn_shifted.dim() != 5:
        raise ValueError("Mode-2 all-input-slice kernel expects gp/gn with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    xp = x_p.contiguous()
    xn = x_n.contiguous()
    gp = gp_shifted.contiguous()
    gn = gn_shifted.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = xp.shape
    gm, p, s, gk, l = gp.shape
    if gn.shape != gp.shape:
        raise ValueError(f"Conductance branch shape mismatch: gp={tuple(gp.shape)}, gn={tuple(gn.shape)}")
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(xp.shape)}, g={tuple(gp.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=xp.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _diff_input_accumulate_2d_all_input_slices_kernel[grid](
        xp,
        xn,
        gp,
        gn,
        xmax,
        scale,
        out,
        xp.stride(0),
        xp.stride(1),
        xp.stride(2),
        xp.stride(3),
        xp.stride(4),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices_gidx(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gp_idx: torch.Tensor,
    gn_idx: torch.Tensor,
    slice_scale_row: torch.Tensor,
    xmax: float,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run mode-2 differential-input fused-DAC kernel directly from G indices."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 4 or x_n.dim() != 4:
        raise ValueError("Triton G-index differential accumulate supports 2-D Linear tensors only.")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gp_idx.dim() != 5 or gn_idx.dim() != 5 or gp_idx.shape != gn_idx.shape:
        raise ValueError("Mode-2 G-index differential kernel expects gp/gn with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    xp = x_p.contiguous()
    xn = x_n.contiguous()
    gp = gp_idx.contiguous()
    gn = gn_idx.contiguous()
    scale = slice_scale_row.contiguous()

    n, m, j, k = xp.shape
    gm, p, s, gk, l = gp.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(xp.shape)}, g={tuple(gp.shape)}")
    if scale.numel() != s:
        raise ValueError(f"slice_scale_row must have {s} elements; got {scale.numel()}.")

    out = torch.empty((n, m, p, j, l), device=xp.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _diff_input_accumulate_2d_from_slices_gidx_kernel[grid](
        xp,
        xn,
        gp,
        gn,
        scale,
        out,
        int(noise_offset_base),
        xp.stride(0),
        xp.stride(1),
        xp.stride(2),
        xp.stride(3),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(xmax),
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        input_precision,
        bool(read_sigma and read_sigma > 0.0),
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices_direct_final(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gp_shifted: torch.Tensor,
    gn_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    x_qmax: float,
    mat_qmax: float,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    out: torch.Tensor | None = None,
    out_col_offset: int = 0,
    out_cols: int | None = None,
) -> torch.Tensor:
    """Run mode-2 differential-input accumulation directly into final 2-D output.

    This is the differential-pair counterpart of the mode-0 direct-final path:
    it consumes all input slices in one launch, preserves per input/weight-slice
    ADC semantics, applies tile max scaling, reduces across input tiles, and
    writes/accumulates into the final 2-D Linear output buffer.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 5 or x_n.dim() != 5:
        raise ValueError("Mode-2 direct-final kernel expects x_p/x_n with shape [N, M, I, J, K].")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gp_shifted is None or gn_shifted is None or gp_shifted.dim() != 5 or gn_shifted.dim() != 5:
        raise ValueError("Mode-2 direct-final kernel expects gp/gn with shape [M, P, S, K, L].")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("Mode-2 direct-final kernel expects x_max/mat_max rank 4.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    xp = x_p.contiguous()
    xn = x_n.contiguous()
    gp = gp_shifted.contiguous()
    gn = gn_shifted.contiguous()
    xmax_slice = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()
    xm = x_max.contiguous()
    mm = mat_max.contiguous()

    n, m, i_count, j, k = xp.shape
    gm, p, s, gk, l = gp.shape
    if gn.shape != gp.shape:
        raise ValueError(f"Conductance branch shape mismatch: gp={tuple(gp.shape)}, gn={tuple(gn.shape)}")
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(xp.shape)}, g={tuple(gp.shape)}")
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: x={tuple(xp.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: g={tuple(gp.shape)}, mat_max={tuple(mm.shape)}")
    if xmax_slice.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax_slice.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    if out is None:
        out = torch.zeros((n * j, p * l), device=xp.device, dtype=torch.float32)
        out_col_offset = 0
        out_cols = p * l
    else:
        if out.dim() != 2:
            raise ValueError("External direct-final output buffer must be rank 2.")
        if out.shape[0] != n * j:
            raise ValueError(f"Output row mismatch: expected {n * j}, got {out.shape[0]}.")
        if out.dtype != torch.float32:
            raise ValueError("External direct-final output buffer must be float32.")
        if out.device != xp.device:
            raise ValueError("External direct-final output buffer must be on the same device.")
        if out_cols is None:
            out_cols = int(out.shape[1])
        if int(out_cols) > int(out.shape[1]):
            raise ValueError(f"out_cols={out_cols} exceeds output buffer width={out.shape[1]}.")
        if int(out_col_offset) < 0 or int(out_col_offset) >= int(out_cols):
            raise ValueError(
                f"Output column offset out of bounds: offset={out_col_offset}, cols={out_cols}."
            )

    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    _diff_input_accumulate_2d_from_slices_direct_final_kernel[grid](
        xp,
        xn,
        gp,
        gn,
        xmax_slice,
        scale,
        xm,
        mm,
        out,
        xp.stride(0),
        xp.stride(1),
        xp.stride(2),
        xp.stride(3),
        xp.stride(4),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        scale.stride(0),
        scale.stride(1),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        out.stride(0),
        out.stride(1),
        int(out_col_offset),
        int(out_cols),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices_gdiff_direct_final(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gdiff_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    x_qmax: float,
    mat_qmax: float,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    out: torch.Tensor | None = None,
    out_col_offset: int = 0,
    out_cols: int | None = None,
) -> torch.Tensor:
    """Run mode-2 differential-input direct-final accumulation from pre-subtracted G."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 5 or x_n.dim() != 5:
        raise ValueError("Mode-2 Gdiff direct-final expects x_p/x_n with shape [N, M, I, J, K].")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gdiff_shifted is None or gdiff_shifted.dim() != 5:
        raise ValueError("Mode-2 Gdiff direct-final expects gdiff with shape [M, P, S, K, L].")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("Mode-2 Gdiff direct-final expects x_max/mat_max rank 4.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    xp = x_p.contiguous()
    xn = x_n.contiguous()
    gd = gdiff_shifted.contiguous()
    xmax_slice = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()
    xm = x_max.contiguous()
    mm = mat_max.contiguous()

    n, m, i_count, j, k = xp.shape
    gm, p, s, gk, l = gd.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(xp.shape)}, gdiff={tuple(gd.shape)}")
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: x={tuple(xp.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: gdiff={tuple(gd.shape)}, mat_max={tuple(mm.shape)}")
    if xmax_slice.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax_slice.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    if out is None:
        out = torch.zeros((n * j, p * l), device=xp.device, dtype=torch.float32)
        out_col_offset = 0
        out_cols = p * l
    else:
        if out.dim() != 2:
            raise ValueError("External direct-final output buffer must be rank 2.")
        if out.shape[0] != n * j:
            raise ValueError(f"Output row mismatch: expected {n * j}, got {out.shape[0]}.")
        if out.dtype != torch.float32:
            raise ValueError("External direct-final output buffer must be float32.")
        if out.device != xp.device:
            raise ValueError("External direct-final output buffer must be on the same device.")
        if out_cols is None:
            out_cols = int(out.shape[1])
        if int(out_cols) > int(out.shape[1]):
            raise ValueError(f"out_cols={out_cols} exceeds output buffer width={out.shape[1]}.")
        if int(out_col_offset) < 0 or int(out_col_offset) >= int(out_cols):
            raise ValueError(
                f"Output column offset out of bounds: offset={out_col_offset}, cols={out_cols}."
            )

    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    _diff_input_accumulate_2d_from_slices_gdiff_direct_final_kernel[grid](
        xp,
        xn,
        gd,
        xmax_slice,
        scale,
        xm,
        mm,
        out,
        xp.stride(0),
        xp.stride(1),
        xp.stride(2),
        xp.stride(3),
        xp.stride(4),
        gd.stride(0),
        gd.stride(1),
        gd.stride(2),
        gd.stride(3),
        gd.stride(4),
        scale.stride(0),
        scale.stride(1),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        out.stride(0),
        out.stride(1),
        int(out_col_offset),
        int(out_cols),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d_from_slices_gidx_direct_final(
    x_p: torch.Tensor,
    x_n: torch.Tensor,
    gp_idx: torch.Tensor,
    gn_idx: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    x_qmax: float,
    mat_qmax: float,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    out: torch.Tensor | None = None,
    out_col_offset: int = 0,
    out_cols: int | None = None,
) -> torch.Tensor:
    """Run mode-2 differential-input direct-final accumulation from G indices."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_p.dim() != 5 or x_n.dim() != 5:
        raise ValueError("Mode-2 G-index direct-final expects x_p/x_n with shape [N, M, I, J, K].")
    if x_p.shape != x_n.shape:
        raise ValueError(f"Input phase shape mismatch: x_p={tuple(x_p.shape)}, x_n={tuple(x_n.shape)}")
    if gp_idx is None or gn_idx is None or gp_idx.dim() != 5 or gn_idx.dim() != 5:
        raise ValueError("Mode-2 G-index direct-final expects gp/gn indices with shape [M, P, S, K, L].")
    if gp_idx.shape != gn_idx.shape:
        raise ValueError(f"Conductance-index branch shape mismatch: gp={tuple(gp_idx.shape)}, gn={tuple(gn_idx.shape)}")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("Mode-2 G-index direct-final expects x_max/mat_max rank 4.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    xp = x_p.contiguous()
    xn = x_n.contiguous()
    gp = gp_idx.contiguous()
    gn = gn_idx.contiguous()
    xmax_slice = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()
    xm = x_max.contiguous()
    mm = mat_max.contiguous()

    n, m, i_count, j, k = xp.shape
    gm, p, s, gk, l = gp.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x={tuple(xp.shape)}, gidx={tuple(gp.shape)}")
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: x={tuple(xp.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: gidx={tuple(gp.shape)}, mat_max={tuple(mm.shape)}")
    if xmax_slice.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax_slice.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    if out is None:
        out = torch.zeros((n * j, p * l), device=xp.device, dtype=torch.float32)
        out_col_offset = 0
        out_cols = p * l
    else:
        if out.dim() != 2:
            raise ValueError("External direct-final output buffer must be rank 2.")
        if out.shape[0] != n * j:
            raise ValueError(f"Output row mismatch: expected {n * j}, got {out.shape[0]}.")
        if out.dtype != torch.float32:
            raise ValueError("External direct-final output buffer must be float32.")
        if out.device != xp.device:
            raise ValueError("External direct-final output buffer must be on the same device.")
        if out_cols is None:
            out_cols = int(out.shape[1])
        if int(out_cols) > int(out.shape[1]):
            raise ValueError(f"out_cols={out_cols} exceeds output buffer width={out.shape[1]}.")
        if int(out_col_offset) < 0 or int(out_col_offset) >= int(out_cols):
            raise ValueError(
                f"Output column offset out of bounds: offset={out_col_offset}, cols={out_cols}."
            )

    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    _diff_input_accumulate_2d_from_slices_gidx_direct_final_kernel[grid](
        xp,
        xn,
        gp,
        gn,
        xmax_slice,
        scale,
        xm,
        mm,
        out,
        int(noise_offset_base),
        xp.stride(0),
        xp.stride(1),
        xp.stride(2),
        xp.stride(3),
        xp.stride(4),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        scale.stride(0),
        scale.stride(1),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        out.stride(0),
        out.stride(1),
        int(out_col_offset),
        int(out_cols),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        float(vread / (rdac - 1)),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        input_precision,
        bool(read_sigma and read_sigma > 0.0),
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_mode1_gidx_direct_final(
    x: torch.Tensor,
    gp_idx: torch.Tensor,
    gn_idx: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_max: float | torch.Tensor,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref_unit: float,
    rdac: int,
    radc: int,
    vread: float,
    g_level: int,
    tile_in: int,
    tile_out: int,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    input_tile_group: int = 1,
) -> torch.Tensor:
    """Run mode-1 compressed differential-pair VMM directly into final output.

    Mode 1 stores one positive/negative conductance level per weight, plus a
    tile-local weight scale. This kernel fuses DAC quantization, optional read
    noise, differential current accumulation, ADC, tile-scale reconstruction,
    and the reduction over input tiles.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x.dim() != 2:
        raise ValueError("Mode-1 direct-final kernel expects a 2-D input tensor.")
    if gp_idx.dim() != 2 or gn_idx.dim() != 2 or gp_idx.shape != gn_idx.shape:
        raise ValueError("Mode-1 direct-final kernel expects matching 2-D gp/gn index tensors.")
    if w_scale.dim() != 2:
        raise ValueError("Mode-1 direct-final kernel expects a 2-D tile scale grid.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")
    if tile_in <= 0 or tile_out <= 0:
        raise ValueError("tile_in and tile_out must be positive.")
    if input_tile_group <= 0:
        raise ValueError("input_tile_group must be positive.")

    x0 = x.contiguous()
    gp = gp_idx.contiguous()
    gn = gn_idx.contiguous()
    ws = w_scale.contiguous()
    if torch.is_tensor(x_max):
        xmax = x_max.detach().to(device=x0.device, dtype=torch.float32).reshape(()).contiguous()
    else:
        xmax = torch.tensor(float(x_max), device=x0.device, dtype=torch.float32)
    rows, cols = x0.shape
    w_rows, out_cols = gp.shape
    if w_rows != cols:
        raise ValueError(f"Input/weight shape mismatch: x={tuple(x0.shape)}, gp={tuple(gp.shape)}")
    in_tiles = triton.cdiv(cols, tile_in)
    out_tiles = triton.cdiv(out_cols, tile_out)
    if tuple(ws.shape) != (in_tiles, out_tiles):
        raise ValueError(f"w_scale shape mismatch: expected {(in_tiles, out_tiles)}, got {tuple(ws.shape)}")

    num_r_blocks = triton.cdiv(rows, block_r)
    num_l_blocks = triton.cdiv(tile_out, block_l)
    input_tile_group = max(1, min(int(input_tile_group), int(in_tiles)))
    if input_tile_group > 1:
        in_tile_groups = triton.cdiv(in_tiles, input_tile_group)
        use_atomic = in_tile_groups > 1
        out = (
            torch.zeros((rows, out_cols), device=x0.device, dtype=torch.float32)
            if use_atomic
            else torch.empty((rows, out_cols), device=x0.device, dtype=torch.float32)
        )
        grid = (num_r_blocks * in_tile_groups * out_tiles * num_l_blocks,)
        _mode1_gidx_direct_final_grouped_kernel[grid](
            x0,
            gp,
            gn,
            ws,
            xmax,
            out,
            x0.stride(0),
            x0.stride(1),
            gp.stride(0),
            gp.stride(1),
            gn.stride(0),
            gn.stride(1),
            ws.stride(0),
            ws.stride(1),
            out.stride(0),
            out.stride(1),
            rows,
            cols,
            out_cols,
            int(tile_in),
            int(tile_out),
            out_tiles,
            float(lgs),
            float(q_g),
            float(read_sigma),
            int(noise_seed),
            int(noise_offset_base),
            float(adc_ref_unit),
            float(rdac - 1),
            float(radc - 1),
            float(vread),
            float(float(vread) * float(q_g) * float(g_level - 1)),
            input_precision,
            bool(read_sigma and read_sigma > 0.0),
            bool(use_atomic),
            int(block_r),
            int(block_l),
            int(block_k),
            num_r_blocks,
            num_l_blocks,
            in_tiles,
            in_tile_groups,
            input_tile_group,
            num_warps=4,
        )
        return out

    out = torch.zeros((rows, out_cols), device=x0.device, dtype=torch.float32)
    grid = (num_r_blocks * in_tiles * out_tiles * num_l_blocks,)
    _mode1_gidx_direct_final_kernel[grid](
        x0,
        gp,
        gn,
        ws,
        xmax,
        out,
        x0.stride(0),
        x0.stride(1),
        gp.stride(0),
        gp.stride(1),
        gn.stride(0),
        gn.stride(1),
        ws.stride(0),
        ws.stride(1),
        out.stride(0),
        out.stride(1),
        rows,
        cols,
        out_cols,
        int(tile_in),
        int(tile_out),
        out_tiles,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(noise_offset_base),
        float(adc_ref_unit),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        float(float(vread) * float(q_g) * float(g_level - 1)),
        input_precision,
        bool(read_sigma and read_sigma > 0.0),
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        in_tiles,
        num_warps=4,
    )
    return out


def triton_gidx_accumulate_2d_input_slices(
    x_sliced: torch.Tensor,
    gidx: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    q_g: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run a mode-0 compressed G-index kernel that fuses all input slices.

    This preserves analog semantics by applying ADC per input-slice and
    weight-slice pair before accumulating slice-scaled partial sums.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Input-slice fused G-index kernel expects x_sliced with shape [N, M, I, J, K].")
    if gidx is None or gidx.dim() != 5:
        raise ValueError("Input-slice fused G-index kernel expects gidx with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = gidx.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, gidx={tuple(g0.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=x0.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if x0.dtype is torch.bfloat16 else 1 if x0.dtype is torch.float16 else 0
    _gidx_accumulate_2d_mode0_input_slices_kernel[grid](
        x0,
        g0,
        xmax,
        scale,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(q_g),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_gidx_accumulate_2d_input_slices_read_noise(
    x_sliced: torch.Tensor,
    gidx: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run a compressed G-index input-slice kernel with in-kernel read variation."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Input-slice fused G-index read-noise kernel expects x_sliced with shape [N, M, I, J, K].")
    if gidx is None or gidx.dim() != 5:
        raise ValueError("Input-slice fused G-index read-noise kernel expects gidx with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = gidx.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, gidx={tuple(g0.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=x0.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if x0.dtype is torch.bfloat16 else 1 if x0.dtype is torch.float16 else 0
    _gidx_accumulate_2d_mode0_input_slices_read_noise_kernel[grid](
        x0,
        g0,
        xmax,
        scale,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(noise_offset_base),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_gidx_accumulate_2d_input_slices_direct_final(
    x_sliced: torch.Tensor,
    gidx: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    lgs: float,
    q_g: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    x_qmax: float,
    mat_qmax: float,
    read_sigma: float = 0.0,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    out: torch.Tensor | None = None,
    out_col_offset: int = 0,
    out_cols: int | None = None,
) -> torch.Tensor:
    """Run compressed G-index input-slice accumulation directly into final 2-D output.

    This path combines the existing G-index input-slice fused VMM with the
    direct-final-output writeback. It avoids both restoring the compressed
    conductance tensor and materializing the 5-D MapReduce intermediate.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("G-index direct-final kernel expects x_sliced with shape [N, M, I, J, K].")
    if gidx is None or gidx.dim() != 5:
        raise ValueError("G-index direct-final kernel expects gidx with shape [M, P, S, K, L].")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("G-index direct-final kernel expects x_max/mat_max rank 4.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = gidx
    xmax_slice = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()
    xm = x_max
    mm = mat_max

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, gidx={tuple(g0.shape)}")
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: x={tuple(x0.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: gidx={tuple(g0.shape)}, mat_max={tuple(mm.shape)}")
    if xmax_slice.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax_slice.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    if out is None:
        out = torch.zeros((n * j, p * l), device=x0.device, dtype=torch.float32)
        out_col_offset = 0
        out_cols = p * l
    else:
        if out.dim() != 2:
            raise ValueError("External direct-final output buffer must be rank 2.")
        if out.shape[0] != n * j:
            raise ValueError(f"Output row mismatch: expected {n * j}, got {out.shape[0]}.")
        if out.dtype != torch.float32:
            raise ValueError("External direct-final output buffer must be float32.")
        if out.device != x0.device:
            raise ValueError("External direct-final output buffer must be on the same device.")
        if out_cols is None:
            out_cols = int(out.shape[1])
        if int(out_cols) > int(out.shape[1]):
            raise ValueError(f"out_cols={out_cols} exceeds output buffer width={out.shape[1]}.")
        if int(out_col_offset) < 0 or int(out_col_offset) >= int(out_cols):
            raise ValueError(
                f"Output column offset out of bounds: offset={out_col_offset}, cols={out_cols}."
            )
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if x0.dtype is torch.bfloat16 else 1 if x0.dtype is torch.float16 else 0
    _gidx_accumulate_2d_mode0_input_slices_direct_final_kernel[grid](
        x0,
        g0,
        xmax_slice,
        scale,
        xm,
        mm,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        out.stride(0),
        out.stride(1),
        int(out_col_offset),
        int(out_cols),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(noise_offset_base),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        dot_dtype,
        input_precision,
        bool(read_sigma and read_sigma > 0.0),
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_gidx_accumulate_2d_input_slices_read_noise_reuse_v(
    x_sliced: torch.Tensor,
    gidx: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    lgs: float,
    q_g: float,
    read_sigma: float,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    noise_seed: int = 12345,
    noise_offset_base: int = 0,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the read-noise G-index kernel while reusing quantized input voltage.

    This keeps per input-slice/weight-slice ADC semantics, but avoids reloading
    and requantizing the same input tile for every weight slice.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Input-voltage reuse kernel expects x_sliced with shape [N, M, I, J, K].")
    if gidx is None or gidx.dim() != 5:
        raise ValueError("Input-voltage reuse kernel expects gidx with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = gidx.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, gidx={tuple(g0.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=x0.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if x0.dtype is torch.bfloat16 else 1 if x0.dtype is torch.float16 else 0
    _gidx_accumulate_2d_mode0_input_slices_read_noise_reuse_v_kernel[grid](
        x0,
        g0,
        xmax,
        scale,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(lgs),
        float(q_g),
        float(read_sigma),
        int(noise_seed),
        int(noise_offset_base),
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_fast_accumulate_2d_input_slices(
    x_sliced: torch.Tensor,
    g_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run a mode-0 conductance kernel that fuses all input slices.

    This is the restored-conductance counterpart of the compressed G-index
    input-slice kernel. It preserves per input-slice/weight-slice ADC by
    quantizing each slice pair before the digital slice reduction.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Input-slice fused kernel expects x_sliced with shape [N, M, I, J, K].")
    if g_shifted is None or g_shifted.dim() != 5:
        raise ValueError("Input-slice fused kernel expects conductance with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = g_shifted.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, g={tuple(g0.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=x0.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if g0.dtype is torch.bfloat16 or x0.dtype is torch.bfloat16 else 1 if g0.dtype is torch.float16 or x0.dtype is torch.float16 else 0
    _fast_accumulate_2d_input_slices_kernel[grid](
        x0,
        g0,
        xmax,
        scale,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_fast_accumulate_2d_input_slices_direct_final(
    x_sliced: torch.Tensor,
    g_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    x_max: torch.Tensor,
    mat_max: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    x_qmax: float,
    mat_qmax: float,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
    out: torch.Tensor | None = None,
    out_col_offset: int = 0,
    out_cols: int | None = None,
) -> torch.Tensor:
    """Run restored-conductance input-slice accumulation directly into final 2-D output.

    This experimental path fuses the 5-D MapReduce output materialization and
    the subsequent finalize/reduce step. It preserves the existing per
    input-slice/weight-slice ADC semantics, then applies the tile max scaling
    while reducing over input tiles.
    """
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Direct-final kernel expects x_sliced with shape [N, M, I, J, K].")
    if g_shifted is None or g_shifted.dim() != 5:
        raise ValueError("Direct-final kernel expects conductance with shape [M, P, S, K, L].")
    if x_max.dim() != 4 or mat_max.dim() != 4:
        raise ValueError("Direct-final kernel expects x_max/mat_max rank 4.")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = g_shifted.contiguous()
    xmax_slice = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()
    xm = x_max.contiguous()
    mm = mat_max.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, g={tuple(g0.shape)}")
    if tuple(xm.shape[:2]) != (n, m):
        raise ValueError(f"x_max shape mismatch: x={tuple(x0.shape)}, x_max={tuple(xm.shape)}")
    if tuple(mm.shape[:2]) != (m, p):
        raise ValueError(f"mat_max shape mismatch: g={tuple(g0.shape)}, mat_max={tuple(mm.shape)}")
    if xmax_slice.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax_slice.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    if out is None:
        out = torch.zeros((n * j, p * l), device=x0.device, dtype=torch.float32)
        out_col_offset = 0
        out_cols = p * l
    else:
        if out.dim() != 2:
            raise ValueError("External direct-final output buffer must be rank 2.")
        if out.shape[0] != n * j:
            raise ValueError(f"Output row mismatch: expected {n * j}, got {out.shape[0]}.")
        if out.dtype != torch.float32:
            raise ValueError("External direct-final output buffer must be float32.")
        if out.device != x0.device:
            raise ValueError("External direct-final output buffer must be on the same device.")
        if out_cols is None:
            out_cols = int(out.shape[1])
        if int(out_cols) > int(out.shape[1]):
            raise ValueError(f"out_cols={out_cols} exceeds output buffer width={out.shape[1]}.")
        if int(out_col_offset) < 0 or int(out_col_offset) >= int(out_cols):
            raise ValueError(
                f"Output column offset out of bounds: offset={out_col_offset}, cols={out_cols}."
            )
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if g0.dtype is torch.bfloat16 or x0.dtype is torch.bfloat16 else 1 if g0.dtype is torch.float16 or x0.dtype is torch.float16 else 0
    _fast_accumulate_2d_input_slices_direct_final_kernel[grid](
        x0,
        g0,
        xmax_slice,
        scale,
        xm,
        mm,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        xm.stride(0),
        xm.stride(1),
        mm.stride(0),
        mm.stride(1),
        out.stride(0),
        out.stride(1),
        int(out_col_offset),
        int(out_cols),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        float(1.0 / (float(x_qmax) * float(mat_qmax))),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_fast_accumulate_2d_input_slices_reuse_v(
    x_sliced: torch.Tensor,
    g_shifted: torch.Tensor,
    x_sliced_max: torch.Tensor,
    slice_scale: torch.Tensor,
    adc_ref: float,
    rdac: int,
    radc: int,
    vread: float,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the restored-conductance input-slice kernel with input-voltage reuse."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if x_sliced.dim() != 5:
        raise ValueError("Input-slice reuse kernel expects x_sliced with shape [N, M, I, J, K].")
    if g_shifted is None or g_shifted.dim() != 5:
        raise ValueError("Input-slice reuse kernel expects conductance with shape [M, P, S, K, L].")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")
    if rdac < 2 or radc < 2:
        raise ValueError("rdac and radc must be >= 2.")

    x0 = x_sliced.contiguous()
    g0 = g_shifted.contiguous()
    xmax = x_sliced_max.contiguous()
    scale = slice_scale.contiguous()

    n, m, i_count, j, k = x0.shape
    gm, p, s, gk, l = g0.shape
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: x_sliced={tuple(x0.shape)}, g={tuple(g0.shape)}")
    if xmax.numel() != i_count:
        raise ValueError(f"x_sliced_max must have {i_count} elements; got {xmax.numel()}.")
    if tuple(scale.shape) != (i_count, s):
        raise ValueError(f"slice_scale must have shape {(i_count, s)}; got {tuple(scale.shape)}.")

    out = torch.empty((n, m, p, j, l), device=x0.device, dtype=torch.float32)
    num_r_blocks = triton.cdiv(n * j, block_r)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (num_r_blocks * m * p * num_l_blocks,)
    dot_dtype = 2 if g0.dtype is torch.bfloat16 or x0.dtype is torch.bfloat16 else 1 if g0.dtype is torch.float16 or x0.dtype is torch.float16 else 0
    _fast_accumulate_2d_input_slices_reuse_v_kernel[grid](
        x0,
        g0,
        xmax,
        scale,
        out,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        x0.stride(4),
        g0.stride(0),
        g0.stride(1),
        g0.stride(2),
        g0.stride(3),
        g0.stride(4),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        i_count,
        p,
        j,
        k,
        l,
        s,
        float(adc_ref),
        float(rdac - 1),
        float(radc - 1),
        float(vread),
        dot_dtype,
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_r_blocks,
        num_l_blocks,
        num_warps=4,
    )
    return out


def triton_diff_input_accumulate_2d(
    vin_p: torch.Tensor,
    vin_n: torch.Tensor,
    gp_shifted: torch.Tensor,
    gn_shifted: torch.Tensor,
    slice_scale_row: torch.Tensor,
    adc_ref: float,
    radc: int,
    *,
    input_precision: str = "ieee",
    block_r: int = 32,
    block_l: int = 16,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the 2-D Triton kernel for mode2 differential-input inference."""
    if triton is None:
        raise RuntimeError(f"Triton fast accumulate is unavailable: {TRITON_IMPORT_ERROR}")
    if vin_p.dim() != 4 or vin_n.dim() != 4:
        raise ValueError("Triton differential-input accumulate supports 2-D Linear tensors only.")
    if vin_p.shape != vin_n.shape:
        raise ValueError(f"Input phase shape mismatch: vin_p={tuple(vin_p.shape)}, vin_n={tuple(vin_n.shape)}")
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError("input_precision must be 'tf32', 'tf32x3', or 'ieee'.")

    vin_p = vin_p.contiguous()
    vin_n = vin_n.contiguous()
    gp = gp_shifted.contiguous()
    gn = gn_shifted.contiguous()
    slice_scale_row = slice_scale_row.contiguous()

    n, m, j, k = vin_p.shape
    gm, p, s, gk, l = gp.shape
    if gn.shape != gp.shape:
        raise ValueError(f"Conductance branch shape mismatch: gp={tuple(gp.shape)}, gn={tuple(gn.shape)}")
    if (gm, gk) != (m, k):
        raise ValueError(f"Shape mismatch: vin={tuple(vin_p.shape)}, g={tuple(gp.shape)}")

    out = torch.empty((n, m, p, j, l), device=vin_p.device, dtype=torch.float32)
    num_l_blocks = triton.cdiv(l, block_l)
    grid = (triton.cdiv(n * j, block_r), m, p * num_l_blocks)
    _diff_input_accumulate_2d_kernel[grid](
        vin_p,
        vin_n,
        gp,
        gn,
        slice_scale_row,
        out,
        vin_p.stride(0),
        vin_p.stride(1),
        vin_p.stride(2),
        vin_p.stride(3),
        gp.stride(0),
        gp.stride(1),
        gp.stride(2),
        gp.stride(3),
        gp.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        n,
        m,
        p,
        j,
        k,
        l,
        s,
        float(adc_ref),
        float(radc - 1),
        input_precision,
        int(block_r),
        int(block_l),
        int(block_k),
        num_l_blocks,
        num_warps=4,
    )
    return out
