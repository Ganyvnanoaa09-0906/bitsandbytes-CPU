"""3-bit (and experimental 2-bit) blockwise quantisation for CPU training.

This is the productised version of the CPU-forge 2/3-bit research kernels (report
§7.139–§7.150): a frozen base weight is stored as blockwise 3-bit codes (≈11% of
fp32 memory) and can be used for LoRA training or inference. The heavy lifting is done
by the C kernels added to ``csrc/cpu_ops.cpp`` (AVX2 / NEON / scalar fallback):

  * ``cbdequantize_3bit_cpu``  – unpack packed codes into fp32
  * ``cbq3bit_gemm_cpu``       – C[M,N] = A[M,K] @ dequant(B)[N,K]^T without materialising B
  * ``cbq3bit_gemm_t_cpu``     – transposed variant for a custom backward (currently unused
                                  by default: torch's oneDNN matmul wins for small layers)

On machines whose native library does not export those symbols (the current Windows
wheel predates them), every function transparently falls back to a pure-torch
implementation, so the API is portable; only the speed differs.

Layout (identical to the research scripts and the C kernels): blocks of ``block``
elements, one fp32 absmax per block, symmetric codes, block % 8 == 0 (3-bit) or
block % 4 == 0 (2-bit), value(code) = code * absmax/center - absmax with
center = (2^bits - 1) / 2.

Examples
--------
>>> import torch, bitsandbytes as bnb
>>> from bitsandbytes import q3bit
>>> layer = torch.nn.Linear(1024, 1024)
>>> ql = q3bit.QuantLinear3bit.from_linear(layer, lora_r=8)
>>> y = ql(torch.randn(2, 1024))            # frozen 3-bit base + trainable LoRA
>>> packed, absmax = q3bit.quantize_3bit(layer.weight.detach())
>>> w = q3bit.dequantize_3bit(packed, absmax, layer.weight.shape)
>>> logits = q3bit.gemm_3bit(torch.randn(2, 1024), packed, absmax)  # no fp32 materialisation (native)
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "pack_3bit",
    "unpack_3bit",
    "quantize_3bit",
    "dequantize_3bit",
    "quantize_2bit",
    "dequantize_2bit",
    "gemm_3bit",
    "gemm_3bit_transposed",
    "QuantLinear3bit",
    "convert_linears_3bit",
    "native_available",
]

_NATIVE = None
_NATIVE_TRIED = False
_NATIVE_FORCED_OFF = False
_CENTER = {2: 1.5, 3: 3.5}


# ---------------------------------------------------------------------------
# native library access
# ---------------------------------------------------------------------------
def _native():
    """Return the native CDLL if it exports the q3bit symbols, else None."""
    global _NATIVE, _NATIVE_TRIED
    if _NATIVE_FORCED_OFF:
        return None
    if _NATIVE_TRIED:
        return _NATIVE
    _NATIVE_TRIED = True
    if os.environ.get("BNB_Q3BIT_FORCE_TORCH", "").strip() in ("1", "true", "yes"):
        return None
    try:
        from .gdn_cpu import load_native

        lib = load_native()
    except Exception:  # noqa: BLE001 - no native library is a supported configuration
        return None
    try:
        lib.cbdequantize_3bit_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_longlong,
                                              ctypes.c_longlong]
        lib.cbdequantize_3bit_cpu.restype = ctypes.c_longlong
        lib.cbdequantize_2bit_cpu.argtypes = lib.cbdequantize_3bit_cpu.argtypes
        lib.cbdequantize_2bit_cpu.restype = ctypes.c_longlong
        lib.cbq3bit_gemm_cpu.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_longlong,
                                         ctypes.c_longlong, ctypes.c_longlong,
                                         ctypes.c_longlong, ctypes.c_longlong]
        lib.cbq3bit_gemm_cpu.restype = ctypes.c_longlong
        lib.cbq3bit_gemm_t_cpu.argtypes = lib.cbq3bit_gemm_cpu.argtypes
        lib.cbq3bit_gemm_t_cpu.restype = ctypes.c_longlong
    except AttributeError:
        return None
    _NATIVE = lib
    return _NATIVE


def native_available() -> bool:
    """True when the loaded native library exports the 2/3-bit kernels."""
    return _native() is not None


def _set_force_torch(flag: bool) -> None:
    """Testing hook: force the pure-torch fallback path."""
    global _NATIVE_FORCED_OFF, _NATIVE_TRIED, _NATIVE
    _NATIVE_FORCED_OFF = bool(flag)
    _NATIVE_TRIED = False
    _NATIVE = None


# ---------------------------------------------------------------------------
# packing (torch; also used to create the packed weights)
# ---------------------------------------------------------------------------
def _levels(bits: int) -> int:
    return 1 << bits


def _bytes_per_block(bits: int, block: int) -> int:
    if bits == 2:
        return block // 4
    if bits == 3:
        return block // 8 * 3
    raise ValueError("only 2/3-bit supported")


def pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in [0,7] (shape (nblocks, block), block % 8 == 0) into bytes."""
    nb, block = codes.shape
    c = codes.view(nb, block // 8, 8).to(torch.int32)
    v = (c[..., 0] | (c[..., 1] << 3) | (c[..., 2] << 6) | (c[..., 3] << 9)
         | (c[..., 4] << 12) | (c[..., 5] << 15) | (c[..., 6] << 18) | (c[..., 7] << 21))
    b0 = (v & 0xFF).to(torch.uint8)
    b1 = ((v >> 8) & 0xFF).to(torch.uint8)
    b2 = ((v >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(nb, -1)


def unpack_3bit(packed: torch.Tensor, block: int) -> torch.Tensor:
    """Inverse of :func:`pack_3bit` (returns (nblocks, block) uint8 codes).

    ``packed`` may be the per-row 2-D layout (N, blocks_per_row*bytes) as well as the
    flat (nblocks, bytes) layout; both are views of the same block sequence.
    """
    groups = block // 8
    nb = int(packed.numel()) // 3 // groups
    p = packed.reshape(nb, groups, 3).to(torch.int32)
    v = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    return torch.stack([(v >> (3 * i)) & 7 for i in range(8)], dim=-1).reshape(nb, block).to(torch.uint8)


def _pack_2bit(codes: torch.Tensor) -> torch.Tensor:
    nb, block = codes.shape
    c = codes.view(nb, block // 4, 4).to(torch.int32)
    v = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return v.to(torch.uint8).reshape(nb, -1)


def _unpack_2bit(packed: torch.Tensor, block: int) -> torch.Tensor:
    groups = block // 4
    nb = int(packed.numel()) // groups
    v = packed.reshape(nb, groups).to(torch.int32)
    return torch.stack([(v >> (2 * i)) & 3 for i in range(4)], dim=-1).reshape(nb, block).to(torch.uint8)


# ---------------------------------------------------------------------------
# quantise / dequantise
# ---------------------------------------------------------------------------
def _quantize(w: torch.Tensor, bits: int, block: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, int]:
    if block % (8 if bits == 3 else 4) != 0 or block <= 0:
        raise ValueError("block must be a positive multiple of 8 for 3-bit / 4 for 2-bit")
    x = w.detach().float().contiguous()
    shape = x.shape
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (block - n % block) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    nb = flat.numel() // block
    blk = flat.view(nb, block)
    absmax = blk.abs().amax(dim=1)
    center = _CENTER[bits]
    scale = (absmax / center).clamp_min(1e-12)
    q = torch.round(blk / scale.unsqueeze(1) + center).clamp(0, _levels(bits) - 1).to(torch.uint8)
    packed = pack_3bit(q) if bits == 3 else _pack_2bit(q)
    absmax = absmax.float()
    # For a 2-D weight whose last dimension is block-aligned, keep the per-row block
    # layout (N, nblocks_per_row * bytes) so the fused GEMM can index rows directly.
    if len(shape) == 2 and shape[1] % block == 0:
        nb_per_row = shape[1] // block
        bytes_per_block = packed.shape[1]
        packed = packed.reshape(shape[0], nb_per_row * bytes_per_block)
        absmax = absmax.reshape(shape[0], nb_per_row)
    return packed, absmax, shape, n


def quantize_3bit(w: torch.Tensor, block: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockwise symmetric 3-bit quantisation.

    Returns ``(packed_uint8, absmax_float32)``; the byte layout matches the C kernels
    (3 bytes per 8 weights, one fp32 absmax per ``block`` weights).
    """
    packed, absmax, _, _ = _quantize(w, 3, block)
    return packed, absmax


def quantize_2bit(w: torch.Tensor, block: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Experimental 2-bit variant (higher error, smaller memory; same layout rules)."""
    packed, absmax, _, _ = _quantize(w, 2, block)
    return packed, absmax


def _dequantize_torch(packed: torch.Tensor, absmax: torch.Tensor, shape, n: int,
                      bits: int, block: int) -> torch.Tensor:
    # flatten: the 2-D per-row layout is just a view of the same block sequence
    absmax_f = absmax.reshape(-1).to(torch.float32)
    packed_f = packed.reshape(-1, packed.shape[-1])
    codes = unpack_3bit(packed_f, block) if bits == 3 else _unpack_2bit(packed_f, block)
    center = _CENTER[bits]
    scale = absmax_f / center
    vals = codes.float() * scale.unsqueeze(1) - absmax_f.unsqueeze(1)
    return vals.reshape(-1)[:n].reshape(shape)


def dequantize_3bit(packed: torch.Tensor, absmax: torch.Tensor, shape,
                    block: int = 64, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Unpack 3-bit codes back to a tensor of ``shape``.

    The native kernel is used when available (exact same arithmetic); otherwise the
    pure-torch unpack is used. ``packed``/``absmax`` are never modified.
    """
    n = 1
    for s in shape:
        n *= int(s)
    packed = packed.contiguous()
    absmax = absmax.contiguous()
    nblocks = int(absmax.numel())
    if native_available() and packed.is_contiguous() and absmax.dtype in (torch.float32,):
        lib = _native()
        # the kernel writes nblocks*block floats (the last block is padded)
        out = torch.empty(nblocks * int(block), dtype=torch.float32)
        rc = lib.cbdequantize_3bit_cpu(packed.data_ptr(), absmax.data_ptr(),
                                       out.data_ptr(), nblocks, int(block))
        if rc == 0:
            return out[:n].reshape(shape).to(dtype or torch.float32)
    out = _dequantize_torch(packed, absmax, shape, n, 3, block)
    return out.to(dtype or torch.float32)


def dequantize_2bit(packed: torch.Tensor, absmax: torch.Tensor, shape,
                    block: int = 64, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    packed = packed.contiguous()
    absmax = absmax.contiguous()
    nblocks = int(absmax.numel())
    if native_available():
        lib = _native()
        out = torch.empty(nblocks * int(block), dtype=torch.float32)
        rc = lib.cbdequantize_2bit_cpu(packed.data_ptr(), absmax.data_ptr(),
                                       out.data_ptr(), nblocks, int(block))
        if rc == 0:
            return out[:n].reshape(shape).to(dtype or torch.float32)
    out = _dequantize_torch(packed, absmax, shape, n, 2, block)
    return out.to(dtype or torch.float32)


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------
def _check_gemm_args(x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor,
                     block: int) -> Tuple[int, int, torch.Tensor]:
    """Validate args and normalise ``packed`` to the per-row (N, blocks*bytes) layout."""
    if x.dim() != 2:
        raise ValueError("x must be 2-D (M, K)")
    if packed.dim() != 2:
        raise ValueError("packed weight must be 2-D")
    K = x.shape[1]
    if K % block != 0:
        raise ValueError("K (%d) must be divisible by block (%d)" % (K, block))
    nb_per_row = K // block
    bpb = _bytes_per_block(3, block)
    bytes_per_row = nb_per_row * bpb
    if packed.shape[1] == bytes_per_row:
        pass                                    # already per-row layout
    elif packed.shape[1] == bpb:
        # flat (nblocks, bytes) layout from an older/simple quantizer call
        if packed.shape[0] % nb_per_row != 0:
            raise ValueError("packed block count is not a multiple of blocks-per-row")
        packed = packed.reshape(-1, bytes_per_row)
    else:
        raise ValueError("packed row width %d does not match K/block (=%d)"
                         % (packed.shape[1], bytes_per_row))
    if int(absmax.numel()) != packed.shape[0] * nb_per_row:
        raise ValueError("absmax count %%d does not match packed blocks")
    if packed.shape[0] == 0:
        raise ValueError("empty packed weight")
    return packed.shape[0], K, packed


def gemm_3bit(x: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor,
              block: int = 64, threads: int = 0, ntile: int = 64) -> torch.Tensor:
    """``x @ dequant(packed).T`` without materialising the fp32 weight (native path).

    ``x`` is (M, K), the packed weight is (N, K) row-blockwise 3-bit. The native kernel
    is used when available (it never writes the fp32 weight); otherwise this falls back
    to ``x @ dequantize_3bit(...).T`` in torch.
    """
    N, K, packed = _check_gemm_args(x, packed, absmax, block)
    if native_available():
        x = x.contiguous().float()
        packed = packed.contiguous()
        absmax = absmax.contiguous()
        out = torch.empty(x.shape[0], N, dtype=torch.float32)
        lib = _native()
        rc = lib.cbq3bit_gemm_cpu(x.data_ptr(), int(x.shape[0]), packed.data_ptr(),
                                  absmax.data_ptr(), out.data_ptr(), N, K, int(block),
                                  int(threads), int(ntile))
        if rc == 0:
            return out
    w = dequantize_3bit(packed, absmax, (N, K), block=block)
    return x.float() @ w.t()


def _check_gemm_t_args(dy: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor,
                       block: int) -> Tuple[int, int, int, torch.Tensor]:
    """Validate the transposed GEMM: here dy is (M, N) and K comes from the packed rows."""
    if dy.dim() != 2:
        raise ValueError("dy must be 2-D (M, N)")
    if packed.dim() != 2:
        raise ValueError("packed weight must be 2-D")
    bpb = _bytes_per_block(3, block)
    if packed.shape[1] % bpb != 0:
        raise ValueError("packed row width is not a multiple of bytes-per-block")
    nb_per_row = packed.shape[1] // bpb
    K = nb_per_row * block
    N = packed.shape[0]
    if dy.shape[1] != N:
        raise ValueError("dy.shape[1] (%d) != packed rows (%d)" % (dy.shape[1], N))
    if int(absmax.numel()) != N * nb_per_row:
        raise ValueError("absmax count does not match packed blocks")
    return dy.shape[0], N, K, packed


def gemm_3bit_transposed(dy: torch.Tensor, packed: torch.Tensor, absmax: torch.Tensor,
                         block: int = 64, threads: int = 0, ntile: int = 64) -> torch.Tensor:
    """``dy @ dequant(packed)`` (i.e. the transposed GEMM used by the backward pass)."""
    M, N, K, packed = _check_gemm_t_args(dy, packed, absmax, block)
    if native_available():
        dy = dy.contiguous().float()
        packed = packed.contiguous()
        absmax = absmax.contiguous()
        out = torch.empty(M, K, dtype=torch.float32)
        lib = _native()
        rc = lib.cbq3bit_gemm_t_cpu(dy.data_ptr(), int(M), packed.data_ptr(),
                                    absmax.data_ptr(), out.data_ptr(), N, K, int(block),
                                    int(threads), int(ntile))
        if rc == 0:
            return out
    w = dequantize_3bit(packed, absmax, (N, K), block=block)
    return dy.float() @ w


# ---------------------------------------------------------------------------
# module: frozen 3-bit base + trainable LoRA
# ---------------------------------------------------------------------------
class QuantLinear3bit(nn.Module):
    """A frozen 3-bit ``nn.Linear`` replacement with trainable LoRA adapters.

    The base matrix is stored as blockwise 3-bit codes (≈11% of fp32 memory) and is
    frozen; only ``lora_A``/``lora_B`` are trainable. In grad mode the forward
    dequantises the weight (native kernel when available) and runs a regular
    matmul/``F.linear`` so autograd handles the input gradient and the LoRA
    gradients; in ``no_grad`` mode on a native build it calls the fused 3-bit GEMM
    and never materialises the fp32 weight.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 block: int = 64, lora_r: int = 8, lora_alpha: int = 16,
                 dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        if block % 8 != 0:
            raise ValueError("block must be a multiple of 8")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block = int(block)
        self.bits = 3
        # the buffers are filled by from_linear() / load_from_linear()
        self.register_buffer("packed", None)
        self.register_buffer("absmax", None)
        self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype,
                                             device=device), requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(lora_r, self.in_features, dtype=dtype,
                                               device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_r, dtype=dtype,
                                               device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = lora_alpha / lora_r

    @classmethod
    def from_linear(cls, linear: nn.Linear, block: int = 64, lora_r: int = 8,
                    lora_alpha: int = 16) -> "QuantLinear3bit":
        """Build a :class:`QuantLinear3bit` from an existing ``nn.Linear``."""
        m = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                block=block, lora_r=lora_r, lora_alpha=lora_alpha,
                dtype=linear.weight.dtype, device=linear.weight.device)
        m.load_from_linear(linear)
        return m

    def load_from_linear(self, linear: nn.Linear) -> None:
        """Quantise ``linear.weight`` into this module and copy/freeze the bias."""
        w = linear.weight.detach()
        packed, absmax = quantize_3bit(w, self.block)
        self.packed = packed
        self.absmax = absmax
        if linear.bias is not None:
            with torch.no_grad():
                self.bias.copy_(linear.bias.detach())

    def weight(self) -> torch.Tensor:
        """Dequantise the base weight (fp32) — allocates N*K floats."""
        if self.packed is None:
            raise RuntimeError("QuantLinear3bit has no packed weight; use from_linear()")
        return dequantize_3bit(self.packed, self.absmax,
                               (self.out_features, self.in_features), block=self.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.packed is None:
            raise RuntimeError("QuantLinear3bit has no packed weight; use from_linear()")
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        base: torch.Tensor
        if torch.is_grad_enabled():
            # dequantise once (native kernel when available) then let autograd compute
            # the input gradient through the frozen weight.
            w = self.weight().to(x.dtype)
            base = F.linear(x2, w, self.bias)
        elif self.in_features % self.block == 0:
            base = gemm_3bit(x2, self.packed, self.absmax, block=self.block)
            if self.bias is not None:
                base = base + self.bias
        else:
            w = self.weight().to(x.dtype)
            base = F.linear(x2, w, self.bias)
        lora = (x2 @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return (base + lora).reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return ("in_features=%d, out_features=%d, bits=3, block=%d, lora_r=%d"
                % (self.in_features, self.out_features, self.block, self.lora_A.shape[0]))


def convert_linears_3bit(model: nn.Module, block: int = 64, lora_r: int = 8,
                         lora_alpha: int = 16, skip_names: Tuple[str, ...] = ("lm_head",),
                         name_prefix: str = "") -> int:
    """Replace every ``nn.Linear`` child with :class:`QuantLinear3bit` (in place).

    Layers whose *leaf* name is in ``skip_names`` are kept in fp32 (``lm_head`` by
    default, because logits need the full unquantised matrix on CPU). Returns the
    number of replaced layers; the base weights become frozen and only the LoRA
    adapters are trainable.
    """
    n = 0
    for name, child in list(model.named_children()):
        full = f"{name_prefix}.{name}" if name_prefix else name
        if isinstance(child, nn.Linear):
            if name in skip_names:
                continue
            setattr(model, name, QuantLinear3bit.from_linear(
                child, block=block, lora_r=lora_r, lora_alpha=lora_alpha))
            n += 1
        else:
            n += convert_linears_3bit(child, block=block, lora_r=lora_r,
                                      lora_alpha=lora_alpha, skip_names=skip_names,
                                      name_prefix=full)
    return n
