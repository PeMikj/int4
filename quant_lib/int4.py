import os
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Iterable, Set

# Common set of transformer linear names to quantize by default.
DEFAULT_TARGETS: Set[str] = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

# Allow disabling Triton path (e.g., if GPU/driver is incompatible).
USE_TRITON = os.getenv("INT4_USE_TRITON", "1") != "0"


# ------------------------- Symmetric int4 ------------------------- #

@triton.jit
def _quantize_rowwise_int4_sym(
    w_ptr, packed_ptr, scale_ptr,
    rows, cols,
    stride_wm, stride_wn,
    stride_pm, stride_pn,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    row_w = w_ptr + row_id * stride_wm

    max_abs = 0.0
    for start in range(0, tl.cdiv(cols, BLOCK_SIZE)):
        offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        vals = tl.load(row_w + offs * stride_wn, mask=mask, other=0.0)
        max_abs = tl.maximum(max_abs, tl.max(tl.abs(vals), axis=0))

    scale = tl.where(max_abs > 0, max_abs / 7.0, 1.0)
    tl.store(scale_ptr + row_id, scale.to(tl.float32))

    packed_cols = tl.cdiv(cols, 2)

    for start in range(0, tl.cdiv(packed_cols, BLOCK_SIZE)):
        offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < packed_cols

        even_idx = offs * 2
        odd_idx = even_idx + 1

        even_mask = mask & (even_idx < cols)
        odd_mask = mask & (odd_idx < cols)

        even_vals = tl.load(row_w + even_idx * stride_wn, mask=even_mask, other=0.0)
        odd_vals = tl.load(row_w + odd_idx * stride_wn, mask=odd_mask, other=0.0)

        xe = even_vals / scale
        xe = tl.minimum(tl.maximum(xe, -8.0), 7.0)
        sign_e = tl.where(xe >= 0, 1.0, -1.0)
        even_q = sign_e * tl.floor(tl.abs(xe) + 0.5)

        xo = odd_vals / scale
        xo = tl.minimum(tl.maximum(xo, -8.0), 7.0)
        sign_o = tl.where(xo >= 0, 1.0, -1.0)
        odd_q = sign_o * tl.floor(tl.abs(xo) + 0.5)

        even_u = (even_q + 8).to(tl.uint8)
        odd_u = (odd_q + 8).to(tl.uint8)

        packed_val = even_u | (odd_u << 4)

        col_p = packed_ptr + offs * stride_pm
        tl.store(col_p + row_id * stride_pn, packed_val, mask=mask)


def quantize_rowwise_int4_sym(weight: torch.Tensor):
    weight = weight.contiguous().to(torch.float16)
    rows, cols = weight.shape
    packed_cols = (cols + 1) // 2

    packed = torch.empty((packed_cols, rows), dtype=torch.uint8, device=weight.device)
    scales = torch.empty(rows, dtype=torch.float32, device=weight.device)

    grid = (rows,)
    _quantize_rowwise_int4_sym[grid](
        weight,
        packed,
        scales,
        rows,
        cols,
        weight.stride(0),
        weight.stride(1),
        packed.stride(0),
        packed.stride(1),
        BLOCK_SIZE=128,
    )
    return packed, scales


@triton.jit
def _matmul_fp16_int4_sym(
    a_ptr, b_ptr, scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    packed_cols = tl.cdiv(K, 2)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K

        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        pack_idx = k // 2
        mask_pack = pack_idx < packed_cols

        w_pack = tl.load(
            b_ptr + pack_idx[None, :] * stride_bm + offs_n[:, None] * stride_bn,
            mask=mask_n[:, None] & mask_pack[None, :],
            other=0,
        ).to(tl.uint8)

        low = (w_pack & 0x0F).to(tl.float32) - 8.0
        high = ((w_pack >> 4) & 0x0F).to(tl.float32) - 8.0
        w_tile = tl.where((k % 2 == 1)[None, :], high, low)

        scales = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        w_tile = w_tile * scales[:, None]

        acc += tl.dot(a_tile, tl.trans(w_tile))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def matmul_fp16_int4_sym(a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor):
    a = a.contiguous().to(torch.float16)
    M, K = a.shape
    packed_K, N = b_packed.shape

    assert packed_K >= (K + 1) // 2

    if USE_TRITON:
        try:
            c = torch.empty((M, N), dtype=torch.float16, device=a.device)
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
            _matmul_fp16_int4_sym[grid](
                a,
                b_packed,
                scales,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b_packed.stride(0),
                b_packed.stride(1),
                c.stride(0),
                c.stride(1),
            )
            return c
        except Exception as exc:  # noqa: BLE001
            # Fall back to pure Torch if Triton compilation/runtime fails.
            if os.getenv("INT4_VERBOSE_FALLBACK", "1") != "0":
                print(f"[int4] Symmetric Triton matmul failed, fallback to torch: {exc}")

    # Torch fallback: unpack int4 and multiply.
    device = a.device
    cols = torch.arange(K, device=device)
    pack_idx = cols // 2
    is_odd = (cols % 2 == 1)

    b_low = (b_packed & 0x0F).to(torch.float32) - 8.0
    b_high = ((b_packed >> 4) & 0x0F).to(torch.float32) - 8.0
    weight = torch.where(is_odd[:, None], b_high[pack_idx], b_low[pack_idx])
    weight = (weight * scales).to(torch.float16)

    return torch.matmul(a, weight)


class SymmetricQuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        packed_cols = (in_features + 1) // 2

        self.register_buffer(
            "weight_packed",
            torch.zeros((packed_cols, out_features), dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float32),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_fp16(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        weight = weight.reshape(self.out_features, self.in_features)
        packed, scales = quantize_rowwise_int4_sym(weight)
        self.weight_packed.copy_(packed)
        self.weight_scale.copy_(scales)

        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(torch.float16))

    def forward(self, x: torch.Tensor):
        orig = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        out = matmul_fp16_int4_sym(x, self.weight_packed, self.weight_scale)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig, self.out_features)


# ------------------------- Asymmetric int4 ------------------------- #

@triton.jit
def _quantize_rowwise_int4_asym(
    w_ptr, packed_ptr, scale_ptr, zero_ptr,
    rows, cols,
    stride_wm, stride_wn,
    stride_pm, stride_pn,
    BLOCK_SIZE: tl.constexpr = 64,
):
    row_id = tl.program_id(0)
    if row_id >= rows:
        return

    row_w = w_ptr + row_id * stride_wm

    row_min = 1e9
    row_max = -1e9
    for start in range(0, tl.cdiv(cols, BLOCK_SIZE)):
        offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < cols
        vals = tl.load(row_w + offs * stride_wn, mask=mask, other=0.0).to(tl.float32)
        row_min = tl.minimum(row_min, tl.min(tl.where(mask, vals, 1e9), axis=0))
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, vals, -1e9), axis=0))

    range_val = row_max - row_min
    range_safe = tl.maximum(range_val, 1e-6)
    scale = range_safe / 15.0
    zero_point = tl.extra.cuda.libdevice.rint(-row_min / scale)
    zero_point = tl.minimum(tl.maximum(zero_point, 0.0), 15.0)

    tl.store(scale_ptr + row_id, scale.to(tl.float32))
    tl.store(zero_ptr + row_id, zero_point.to(tl.float32))

    packed_cols = tl.cdiv(cols, 2)

    for start in range(0, tl.cdiv(packed_cols, BLOCK_SIZE)):
        offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < packed_cols

        even_idx = offs * 2
        odd_idx = even_idx + 1

        even_mask = mask & (even_idx < cols)
        odd_mask = mask & (odd_idx < cols)

        even_vals = tl.load(row_w + even_idx * stride_wn, mask=even_mask, other=0.0).to(tl.float32)
        odd_vals = tl.load(row_w + odd_idx * stride_wn, mask=odd_mask, other=0.0).to(tl.float32)

        even_q = tl.extra.cuda.libdevice.rint(even_vals / scale + zero_point)
        odd_q = tl.extra.cuda.libdevice.rint(odd_vals / scale + zero_point)
        even_q = tl.minimum(tl.maximum(even_q, 0.0), 15.0)
        odd_q = tl.minimum(tl.maximum(odd_q, 0.0), 15.0)

        even_u = even_q.to(tl.uint8)
        odd_u = odd_q.to(tl.uint8)
        packed_val = even_u | (odd_u << 4)

        col_p = packed_ptr + offs * stride_pm
        tl.store(col_p + row_id * stride_pn, packed_val, mask=mask)


def quantize_rowwise_int4_asym(weight: torch.Tensor):
    weight = weight.contiguous().to(torch.float16)
    rows, cols = weight.shape
    packed_cols = (cols + 1) // 2

    packed = torch.empty((packed_cols, rows), dtype=torch.uint8, device=weight.device)
    scales = torch.empty(rows, dtype=torch.float32, device=weight.device)
    zeros = torch.empty(rows, dtype=torch.float32, device=weight.device)

    grid = (rows,)
    _quantize_rowwise_int4_asym[grid](
        weight,
        packed,
        scales,
        zeros,
        rows,
        cols,
        weight.stride(0),
        weight.stride(1),
        packed.stride(0),
        packed.stride(1),
    )
    return packed, scales, zeros


@triton.jit
def _matmul_fp16_int4_asym(
    a_ptr, b_ptr, scale_ptr, zero_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
    NUM_WARPS: tl.constexpr = 4,
    NUM_STAGES: tl.constexpr = 3,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    packed_cols = tl.cdiv(K, 2)

    scales = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    zeros = tl.load(zero_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K

        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        pack_idx = k // 2
        mask_pack = pack_idx < packed_cols

        w_pack = tl.load(
            b_ptr + pack_idx[None, :] * stride_bm + offs_n[:, None] * stride_bn,
            mask=mask_n[:, None] & mask_pack[None, :],
            other=0,
        ).to(tl.uint8)

        low = (w_pack & 0x0F).to(tl.float32)
        high = ((w_pack >> 4) & 0x0F).to(tl.float32)
        w_tile = tl.where((k % 2 == 1)[None, :], high, low)

        w_tile = (w_tile - zeros[:, None]) * scales[:, None]

        acc += tl.dot(a_tile, tl.trans(w_tile))

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def matmul_fp16_int4_asym(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
):
    a = a.contiguous().to(torch.float16)
    M, K = a.shape
    packed_K, N = b_packed.shape
    assert packed_K >= (K + 1) // 2

    if USE_TRITON:
        try:
            c = torch.empty((M, N), dtype=torch.float16, device=a.device)

            grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
            _matmul_fp16_int4_asym[grid](
                a,
                b_packed,
                scales,
                zeros,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b_packed.stride(0),
                b_packed.stride(1),
                c.stride(0),
                c.stride(1),
            )
            return c
        except Exception as exc:  # noqa: BLE001
            if os.getenv("INT4_VERBOSE_FALLBACK", "1") != "0":
                print(f"[int4] Asymmetric Triton matmul failed, fallback to torch: {exc}")

    device = a.device
    cols = torch.arange(K, device=device)
    pack_idx = cols // 2
    is_odd = (cols % 2 == 1)

    b_low = (b_packed & 0x0F).to(torch.float32)
    b_high = ((b_packed >> 4) & 0x0F).to(torch.float32)
    weight = torch.where(is_odd[:, None], b_high[pack_idx], b_low[pack_idx])
    weight = (weight - zeros) * scales
    weight = weight.to(torch.float16)

    return torch.matmul(a, weight)


class AsymmetricQuantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        packed_cols = (in_features + 1) // 2

        self.register_buffer(
            "weight_packed",
            torch.zeros((packed_cols, out_features), dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_zero",
            torch.zeros(out_features, dtype=torch.float32),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def quantize_from_fp16(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        weight = weight.reshape(self.out_features, self.in_features)

        packed, scales, zeros = quantize_rowwise_int4_asym(weight)

        self.weight_packed.copy_(packed)
        self.weight_scale.copy_(scales)
        self.weight_zero.copy_(zeros)

        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(torch.float16))

    def forward(self, x: torch.Tensor):
        orig = x.shape[:-1]
        x = x.reshape(-1, self.in_features)

        out = matmul_fp16_int4_asym(
            x,
            self.weight_packed,
            self.weight_scale,
            self.weight_zero,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig, self.out_features)


# ------------------------- Helpers ------------------------- #

def _as_set(targets: Iterable[str] | None) -> Set[str]:
    if targets is None:
        return set(DEFAULT_TARGETS)
    return set(targets)


def replace_linears_with_quant(module: nn.Module, targets: Iterable[str] | None = None, mode: str = "sym"):
    """Recursively swap torch.nn.Linear layers by name.

    targets: iterable of layer names to replace (defaults to DEFAULT_TARGETS)
    mode:    "sym" or "asym" quantization kernels
    """
    targets_set = _as_set(targets)
    mode_norm = mode.lower()
    if mode_norm in {"sym", "symmetric"}:
        quant_cls = SymmetricQuantLinear
    elif mode_norm in {"asym", "asymmetric"}:
        quant_cls = AsymmetricQuantLinear
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    for name, child in list(module.named_children()):
        if name in targets_set and isinstance(child, nn.Linear):
            out_f, in_f = child.weight.shape
            q = quant_cls(in_f, out_f, bias=(child.bias is not None)).to(child.weight.device)
            q.quantize_from_fp16(child.weight, child.bias)
            setattr(module, name, q)
            continue
        replace_linears_with_quant(child, targets_set, mode)


__all__ = [
    "DEFAULT_TARGETS",
    "SymmetricQuantLinear",
    "AsymmetricQuantLinear",
    "replace_linears_with_quant",
    "quantize_rowwise_int4_sym",
    "quantize_rowwise_int4_asym",
    "matmul_fp16_int4_sym",
    "matmul_fp16_int4_asym",
]
