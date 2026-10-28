# -*- coding: utf-8 -*-
"""fused_cpu.py — bnb 融合 elementwise 算子的 Python 封装（含 autograd）

背景（实测）:
  训练步 2.819 s = GEMM 63.6% / elementwise 20.8% / attention 7.6% / 其他
  GEMM 已达同形状纯 GEMM 基准的 92% ⇒ 唯一空间是 elementwise
  RMSNorm 在 eager 下展开成 6 个 aten 算子（pow/mean/add/rsqrt/mul/mul）

本模块提供:
  fused_rmsnorm(x, weight, res=None, eps=1e-6)   autograd 版
  fused_swiglu(gate, up)                          autograd 版
  fused_add_scale(x, y, scale)                    autograd 版
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import ctypes
import os

import torch

_LIB = None
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATHS = [
    # repo root 是 <here>/bitsandbytes，package 是 <here>/bitsandbytes/bitsandbytes
    os.path.join(_HERE, 'bitsandbytes', 'bitsandbytes', 'libbitsandbytes_cpu.dll'),
    os.path.join(_HERE, 'bitsandbytes', 'libbitsandbytes_cpu.dll'),
    os.path.join(_HERE, 'libbitsandbytes_cpu.dll'),
]


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    # 直接 ctypes 载入优先 —— 因为 cwd 里有个叫 bitsandbytes 的 repo 目录，
    # `import bitsandbytes` 会解析成命名空间包（__file__ = None）而拿不到 cextension。
    for p in _LIB_PATHS:
        if os.path.exists(p):
            lib = ctypes.CDLL(p)
            if hasattr(lib, 'cfused_rmsnorm_fwd_cpu'):
                _LIB = lib
                return _LIB
    # 退路：走 bitsandbytes 包自己的加载器
    try:
        from bitsandbytes import cextension
        if cextension.lib is not None and hasattr(cextension.lib, 'cfused_rmsnorm_fwd_cpu'):
            _LIB = cextension.lib
            return _LIB
    except Exception:
        pass
    raise RuntimeError('找不到含融合核的 libbitsandbytes_cpu.dll（需要重新编译 bnb）')


def _bind(lib):
    fp = ctypes.POINTER(ctypes.c_float)
    fp_ = ctypes.POINTER(ctypes.c_float)
    ll = ctypes.c_longlong
    if not hasattr(lib, 'cfused_rmsnorm_fwd_cpu'):
        raise RuntimeError('DLL 里没有融合核 —— 需要重新编译 bnb')
    lib.cfused_rmsnorm_fwd_cpu.argtypes = [fp, fp, fp, fp, fp, ll, ll, ctypes.c_float]
    lib.cfused_rmsnorm_fwd_cpu.restype = ll
    lib.cfused_rmsnorm_bwd_cpu.argtypes = [fp, fp, fp, fp, fp, fp, ll, ll]
    lib.cfused_rmsnorm_bwd_cpu.restype = ll
    lib.cfused_swiglu_fwd_cpu.argtypes = [fp, fp, fp, ll]
    lib.cfused_swiglu_fwd_cpu.restype = ll
    lib.cfused_swiglu_bwd_cpu.argtypes = [fp, fp, fp, fp, fp, ll]
    lib.cfused_swiglu_bwd_cpu.restype = ll
    lib.cfused_add_scale_cpu.argtypes = [fp, fp, fp, fp, ll, ll]
    lib.cfused_add_scale_cpu.restype = ll
    # v2
    lib.cfused_swiglu_strided_fwd_cpu.argtypes = [fp, ll, ll, fp, ll, ll, fp, ll, ll, ll]
    lib.cfused_swiglu_strided_fwd_cpu.restype = ll
    lib.cfused_swiglu_strided_bwd_cpu.argtypes = [
        fp, ll, ll, fp, ll, ll, fp, ll, ll, fp, ll, ll, fp, ll, ll, ll, ll]
    lib.cfused_swiglu_strided_bwd_cpu.restype = ll
    lib.cfused_add_scale_inplace_cpu.argtypes = [fp, fp, fp, ll, ll]
    lib.cfused_add_scale_inplace_cpu.restype = ll
    lib.cfused_add_scale_rmsnorm_cpu.argtypes = [fp, fp, fp, fp, fp, fp, ll, ll, ctypes.c_float]
    lib.cfused_add_scale_rmsnorm_cpu.restype = ll
    return lib


def available():
    try:
        lib = _load()
        _bind(lib)
        return True
    except Exception:
        return False


def _p(t):
    return ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_float))


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, res, eps):
        # 显式算 h = x + res：反向公式需要 h 而不是 x（∂h/∂x = ∂h/∂res = 1）
        h = x + res if res is not None else x
        M = h.numel() // h.shape[-1]
        D = h.shape[-1]
        h2 = h.reshape(M, D).contiguous()
        w = weight.contiguous() if weight is not None else None
        out = torch.empty(M, D, dtype=h.dtype)
        xs = torch.empty(M, dtype=torch.float32)
        lib = _bind(_load())
        lib.cfused_rmsnorm_fwd_cpu(
            _p(h2), None, _p(w) if w is not None else None, _p(out), _p(xs),
            M, D, float(eps))
        ctx.has_w = w is not None
        ctx.shape = x.shape
        ctx.res = res is not None
        ctx.save_for_backward(h2, w if w is not None else torch.empty(0), xs)
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, dout):
        h2, w, xs = ctx.saved_tensors
        M, D = h2.shape
        has_w = ctx.has_w
        d2 = dout.reshape(M, D).contiguous()
        dx = torch.empty(M, D)
        dw = torch.empty(D) if has_w else None
        lib = _bind(_load())
        lib.cfused_rmsnorm_bwd_cpu(
            _p(h2), _p(w) if has_w else None, _p(d2), _p(xs),
            _p(dx), _p(dw) if has_w else None, M, D)
        dxr = dx.reshape(ctx.shape)
        # h = x + res  =>  dx = dh, dres = dh
        return dxr, (dw if has_w else None), (dxr if ctx.res else None), None


def fused_rmsnorm(x, weight=None, res=None, eps=1e-6):
    """out = (x + res) / sqrt(mean((x+res)^2)+eps) * weight"""
    return _RMSNormFn.apply(x, weight, res, eps)


# --------------------------------------------------------------------------
# SwiGLU（v2：支持 stride，零拷贝直接吃 chunk 出来的视图）
# --------------------------------------------------------------------------
class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        # 支持任意前导维：展平成 (M, D) 视图（对 chunk 出来的切片是零拷贝）
        D = gate.shape[-1]
        g2 = gate.reshape(-1, D)
        u2 = up.reshape(-1, D)
        M = g2.shape[0]
        gs0, gs1 = g2.stride()
        us0, us1 = u2.stride()
        out = torch.empty(M, D)
        lib = _bind(_load())
        lib.cfused_swiglu_strided_fwd_cpu(
            _p(g2), gs0, gs1, _p(u2), us0, us1, _p(out), D, M, D)
        ctx.gs = (gs0, gs1)
        ctx.us = (us0, us1)
        ctx.shape = gate.shape
        ctx.saved = (g2, u2)
        return out.reshape(gate.shape)

    @staticmethod
    def backward(ctx, dout):
        g2, u2 = ctx.saved
        gs0, gs1 = ctx.gs
        us0, us1 = ctx.us
        M, D = g2.shape
        d2 = dout.reshape(-1, D)
        if not d2.is_contiguous():
            d2 = d2.contiguous()
        dg = torch.empty(M, D)
        du = torch.empty(M, D)
        lib = _bind(_load())
        lib.cfused_swiglu_strided_bwd_cpu(
            _p(g2), gs0, gs1, _p(u2), us0, us1,
            _p(d2), D, 1, _p(dg), D, 1, _p(du), D, 1, M, D)
        return dg.reshape(ctx.shape), du.reshape(ctx.shape)


def fused_swiglu(gate, up):
    """out = silu(gate) * up   （gate/up 可以是 chunk 出来的非连续视图，零拷贝）"""
    return _SwiGLUFn.apply(gate, up)


def fused_add_scale_rmsnorm(x, y, scale, weight, eps=1e-6):
    """h = x + scale*y ; out = h / sqrt(mean(h^2)+eps) * weight

    ⚠️ 就地修改 x（把它变成 h）。返回 (out, xs)。8 个算子一条链，融合才划算。
    """
    M = x.numel() // x.shape[-1]
    D = x.shape[-1]
    x2 = x.reshape(M, D)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y2 = y.reshape(M, D)
    if not y2.is_contiguous():
        y2 = y2.contiguous()
    s = scale.contiguous() if scale is not None else None
    w = weight.contiguous() if weight is not None else None
    out = torch.empty(M, D)
    xs = torch.empty(M, dtype=torch.float32)
    lib = _bind(_load())
    lib.cfused_add_scale_rmsnorm_cpu(
        _p(x2), _p(y2), _p(s) if s is not None else None,
        _p(w) if w is not None else None, _p(out), _p(xs), M, D, float(eps))
    return out, xs


# --------------------------------------------------------------------------
# add_scale
# --------------------------------------------------------------------------
class _AddScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, scale):
        M = x.numel() // x.shape[-1]
        D = x.shape[-1]
        x2 = x.reshape(M, D).contiguous()
        y2 = y.reshape(M, D).contiguous()
        s = scale.contiguous() if scale is not None else None
        out = torch.empty(M, D)
        lib = _bind(_load())
        lib.cfused_add_scale_cpu(_p(x2), _p(y2), _p(s) if s is not None else None,
                                 _p(out), M, D)
        ctx.save_for_backward(y2, s if s is not None else torch.empty(0))
        ctx.has_s = s is not None
        ctx.shape = x.shape
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, dout):
        y2, s = ctx.saved_tensors
        d = dout.reshape_as(y2)
        dx = d.clone()
        if ctx.has_s:
            dy = d * s.unsqueeze(0)
            ds = (d * y2).sum(0)
        else:
            dy = d.clone()
            ds = None
        return dx.reshape(ctx.shape), dy.reshape(ctx.shape), ds


def fused_add_scale(x, y, scale=None):
    """out = x + scale * y"""
    return _AddScaleFn.apply(x, y, scale)
