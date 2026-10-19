"""AdamW4bit — CPU-only 4-bit blockwise AdamW (restored in 0.50.2.dev8).

The dev2 release shipped this optimizer, but the kernel was dropped when the repository
vendored upstream bitsandbytes 0.50.2 (dev4) while the README/CLI kept documenting it.
This module restores the Python API and the scalar C kernel (``csrc/cpu_ops.cpp``,
``coptimizer_update_4bit_blockwise_cpu``); it falls back to a pure-torch implementation
when the loaded native library does not export that symbol (the Windows wheel).

State cost: two 4-bit codes per byte for m and v plus one fp32 absmax per 256 elements
=> ~1.03 bytes/param instead of 8 for fp32 AdamW (~7.8x less). The scalar kernel is
~4x slower than the AVX2 8-bit kernel per element; the optimizer is a small fraction of
a CPU training step, so this trades speed for memory. (The dev2 AVX2 path for 4-bit was
known to produce wrong codes and stays disabled; only the verified scalar path runs.)

Storage contract (must match csrc): state1/state2 are ceil(n/2) uint8 each, two 4-bit
codes per byte, HIGH nibble = even index; absmax1/absmax2 have ceil(n/256) fp32 each;
qmap is a 256-entry float table of which only the first 16 codes are used
(qmap[c] = c*2/15 - 1, i.e. 16 symmetric linear levels; code 8 is the closest to zero).
"""
from __future__ import annotations

import ctypes
import os

import torch

__all__ = ["AdamW4bit", "native_available", "BLOCKSIZE"]

BLOCKSIZE = 256          # must equal kOptBlockSize in csrc/cpu_ops.cpp
OPT_ADAM = 0

_NATIVE = None
_NATIVE_TRIED = False
_FORCED_OFF = False


def _native():
    """Return (lib) if it exports coptimizer_update_4bit_blockwise_cpu, else None."""
    global _NATIVE, _NATIVE_TRIED
    if _FORCED_OFF:
        return None
    if _NATIVE_TRIED:
        return _NATIVE
    _NATIVE_TRIED = True
    if os.environ.get("BNB_ADAMW4BIT_FORCE_TORCH", "").strip() in ("1", "true", "yes"):
        return None
    try:
        from ..gdn_cpu import load_native

        lib = load_native()
        lib.coptimizer_update_4bit_blockwise_cpu
    except Exception:  # noqa: BLE001 - no native library is a supported configuration
        return None
    _F = ctypes.c_float
    _U8P = ctypes.POINTER(ctypes.c_ubyte)
    _FP = ctypes.POINTER(_F)
    fn = lib.coptimizer_update_4bit_blockwise_cpu
    fn.restype = None
    fn.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, _U8P, _U8P,
        _F, _F, _F, _F, _F, ctypes.c_int, _F, _FP, _FP, _FP, _FP,
        _F, _F, ctypes.c_int, ctypes.c_longlong, ctypes.c_int,
    ]
    _NATIVE = lib
    return _NATIVE


def native_available() -> bool:
    return _native() is not None


def _set_force_torch(flag: bool) -> None:
    """Testing hook: force the pure-torch implementation."""
    global _FORCED_OFF, _NATIVE_TRIED, _NATIVE
    _FORCED_OFF = bool(flag)
    _NATIVE_TRIED = False
    _NATIVE = None


def _qmap() -> torch.Tensor:
    q = torch.empty(256, dtype=torch.float32)
    for i in range(256):
        q[i] = (i & 0x0F) * (2.0 / 15.0) - 1.0
    return q


_QMAP = _qmap()


def _unpack4(state: torch.Tensor, n: int) -> torch.Tensor:
    """uint8 packed nibbles -> int codes (length n)."""
    b = state[: (n + 1) // 2]
    codes = torch.empty(n, dtype=torch.uint8)
    codes[0::2] = b[: (n + 1) // 2] >> 4
    codes[1::2] = b[: n // 2] & 0x0F
    return codes.to(torch.long)   # uint8 indices are treated as deprecated bool masks


def _pack4(codes: torch.Tensor) -> torch.Tensor:
    """int codes (length n) -> uint8 packed nibbles (ceil(n/2)); high nibble = even."""
    n = codes.numel()
    even = codes[0::2].to(torch.uint8)
    odd = codes[1::2].to(torch.uint8)
    out = torch.empty((n + 1) // 2, dtype=torch.uint8)
    out[: even.numel()] = (even << 4) | odd
    return out


def _fallback_step(opt: "AdamW4bit", group: dict, p: torch.Tensor) -> None:
    """Pure-torch equivalent of the scalar C kernel (same arithmetic, same packing)."""
    g = p.grad
    n = p.numel()
    blocks = (n + BLOCKSIZE - 1) // BLOCKSIZE
    st = opt.state[p]
    if "state1" not in st:
        st["state1"] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
        st["state2"] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
        st["absmax1"] = torch.zeros(blocks, dtype=torch.float32)
        st["absmax2"] = torch.zeros(blocks, dtype=torch.float32)
        st["step"] = 0
    st["step"] += 1
    step = st["step"]
    beta1, beta2 = group["betas"]
    eps, lr, wd = group["eps"], group["lr"], group["weight_decay"]
    pad = blocks * BLOCKSIZE - n
    gf = g.detach().float().reshape(-1)
    pf = p.detach().float().reshape(-1)
    if pad:
        gf = torch.cat([gf, torch.zeros(pad)])
        pf = torch.cat([pf, torch.zeros(pad)])
    s1 = _QMAP[_unpack4(st["state1"], n)].to(torch.float32)
    s2 = _QMAP[_unpack4(st["state2"], n)].to(torch.float32)
    am1 = st["absmax1"].repeat_interleave(BLOCKSIZE)[:n]
    am2 = st["absmax2"].repeat_interleave(BLOCKSIZE)[:n]
    if pad:
        s1 = torch.cat([s1, torch.zeros(pad)])
        s2 = torch.cat([s2, torch.zeros(pad)])
        am1 = torch.cat([am1, torch.zeros(pad)])
        am2 = torch.cat([am2, torch.zeros(pad)])
    s1 = (s1 * am1).view(blocks, BLOCKSIZE)
    s2 = (s2 * am2).view(blocks, BLOCKSIZE)
    gv = gf.view(blocks, BLOCKSIZE)
    pv = pf.view(blocks, BLOCKSIZE)
    s2 = s2 * beta2 + (1.0 - beta2) * gv * gv
    s1 = s1 * beta1 + (1.0 - beta1) * gv
    c1 = 1.0 - beta1 ** step
    c2 = (1.0 - beta2 ** step) ** 0.5
    step_size = -lr * c2 / c1
    pn = pv + step_size * s1 / (torch.sqrt(s2) + c2 * eps)
    if wd > 0:
        pn = pn * (1.0 - lr * wd)
    # new block absmax + requantise (zero block -> code 8, exactly like the kernel)
    n1 = s1.abs().amax(dim=1)
    n2 = s2.abs().amax(dim=1)
    inv1 = torch.where(n1 > 0, 1.0 / n1, torch.zeros_like(n1))
    inv2 = torch.where(n2 > 0, 1.0 / n2, torch.zeros_like(n2))
    code1 = torch.round((s1 * inv1[:, None] + 1.0) * 7.5).clamp(0, 15)
    code2 = torch.round((s2 * inv2[:, None] + 1.0) * 7.5).clamp(0, 15)
    zero1 = (n1 == 0)[:, None]
    zero2 = (n2 == 0)[:, None]
    code1 = torch.where(zero1, torch.full_like(code1, 8), code1)
    code2 = torch.where(zero2, torch.full_like(code2, 8), code2)
    st["absmax1"] = n1.to(torch.float32)
    st["absmax2"] = n2.to(torch.float32)
    st["state1"] = _pack4(code1.reshape(-1).to(torch.uint8))
    st["state2"] = _pack4(code2.reshape(-1).to(torch.uint8))
    with torch.no_grad():
        p.copy_(pn.reshape(-1)[:n].reshape(p.shape).to(p.dtype))


class AdamW4bit(torch.optim.Optimizer):
    """CPU 4-bit blockwise AdamW; drop-in for ``torch.optim.AdamW`` on fp32 params.

    >>> opt = AdamW4bit(model.parameters(), lr=1e-4, weight_decay=0.01)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, blocksize=BLOCKSIZE):
        if blocksize != BLOCKSIZE:
            raise ValueError("blocksize must be %d (compiled into cpu_ops.cpp)" % BLOCKSIZE)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self._qmap = _QMAP.clone()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lib = _native()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32:
                    raise TypeError("AdamW4bit supports fp32 params only; got %s" % p.dtype)
                if not p.is_contiguous():
                    p.data = p.data.contiguous()
                g = p.grad
                if not g.is_contiguous():
                    g = g.contiguous()
                if lib is None:
                    _fallback_step(self, group, p)
                    continue
                n = p.numel()
                blocks = (n + BLOCKSIZE - 1) // BLOCKSIZE
                st = self.state[p]
                if "state1" not in st:
                    st["state1"] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
                    st["state2"] = torch.zeros((n + 1) // 2, dtype=torch.uint8)
                    st["absmax1"] = torch.zeros(blocks, dtype=torch.float32)
                    st["absmax2"] = torch.zeros(blocks, dtype=torch.float32)
                    st["step"] = 0
                st["step"] += 1
                b1, b2 = group["betas"]
                lib.coptimizer_update_4bit_blockwise_cpu(
                    OPT_ADAM,
                    g.data_ptr(), p.data_ptr(),
                    ctypes.cast(st["state1"].data_ptr(), ctypes.POINTER(ctypes.c_ubyte)),
                    ctypes.cast(st["state2"].data_ptr(), ctypes.POINTER(ctypes.c_ubyte)),
                    b1, b2, 0.0, 0.0, group["eps"],
                    st["step"], group["lr"],
                    ctypes.cast(self._qmap.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(self._qmap.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(st["absmax1"].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(st["absmax2"].data_ptr(), ctypes.POINTER(ctypes.c_float)),
                    group["weight_decay"], 1.0, 0, n, 0,
                )
        return loss

    def state_bytes(self) -> int:
        """Actual optimizer-state bytes held (the metric this optimizer exists for)."""
        tot = 0
        for st in self.state.values():
            for k in ("state1", "state2", "absmax1", "absmax2"):
                if k in st:
                    tot += st[k].numel() * st[k].element_size()
        return tot
