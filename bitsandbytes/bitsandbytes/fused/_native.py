"""ctypes loader for the zig-compiled AVX2 fused kernels."""
from __future__ import annotations

import ctypes
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLL = os.path.join(_HERE, 'bnb_fused.dll')
_LIB = None
_TRIED = False


def _lib():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if not os.path.exists(_DLL):
        return None
    try:
        lib = ctypes.CDLL(_DLL)
        lib.rms_norm_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.rms_norm_f32.restype = None
        lib.silu_mul_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_longlong]
        lib.silu_mul_f32.restype = None
        _LIB = lib
    except Exception:  # noqa: BLE001 - missing/incompatible lib means torch fallback
        _LIB = None
    return _LIB


def native_available() -> bool:
    return _lib() is not None


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().float()


def rms_norm_native(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lib = _lib()
    if lib is None:
        raise RuntimeError('native fused DLL not available')
    if x.dim() < 2:
        raise ValueError('x must have at least 2 dims')
    D = x.shape[-1]
    x2 = _f32(x).reshape(-1, D)
    w = _f32(weight).reshape(D)
    out = torch.empty_like(x2)
    lib.rms_norm_f32(x2.data_ptr(), w.data_ptr(), out.data_ptr(), x2.shape[0], D, float(eps))
    return out.reshape(x.shape)


def silu_mul_native(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    lib = _lib()
    if lib is None:
        raise RuntimeError('native fused DLL not available')
    g = _f32(gate).reshape(-1)
    u = _f32(up).reshape(-1)
    out = torch.empty_like(g)
    lib.silu_mul_f32(g.data_ptr(), u.data_ptr(), out.data_ptr(), g.numel())
    return out.reshape(gate.shape)
