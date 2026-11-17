# -*- coding: utf-8 -*-
"""igpu_ref.py — pure torch CPU references for the iGPU alignment tests.

Signatures follow LEAD's IGPU_OPS_CONTRACT_LEAD.md:

    ref_linear(x, w, dy=None, bias=None)         -> (y, dx, dw, db)
    ref_rmsnorm(x, w, dy=None, eps=1e-6)         -> (y, dx, dw)
    ref_attention(q, k, v, do=None, causal=True) -> (o, dq, dk, dv)
    ref_conv2d(x, w, b, dy=None, stride=1, padding=0) -> (y, dx, dw, db)

All inputs/outputs are torch tensors; q/k/v are (B*nh, T, hd). Conv2d is a reference only
(the device kernel is still LEAD's side); it uses autograd for the gradients.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _t(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def ref_linear(x, w, dy=None, bias=None):
    """y = x @ w.T (+bias); analytic dx/dw/db.  x:(M,K), w:(N,K), dy:(M,N)."""
    x = _t(x).float()
    w = _t(w).float()
    y = x @ w.t()
    if bias is not None:
        y = y + _t(bias).float()
    if dy is None:
        return y, None, None, None
    dy = _t(dy).float()
    dx = dy @ w
    dw = dy.t() @ x
    db = dy.sum(dim=0) if bias is not None else None
    return y, dx, dw, db


def ref_rmsnorm(x, w, dy=None, eps=1e-6):
    """y = x * rsqrt(mean(x^2)+eps) * w; returns (y, dx, dw)."""
    x = _t(x).float().requires_grad_(True)
    w = _t(w).float().clone().requires_grad_(True)
    rs = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rs * w
    if dy is None:
        return y, None, None
    y.backward(_t(dy).float())
    return y, x.grad, w.grad


def ref_attention(q, k, v, do=None, causal=True, scale=None):
    """q/k/v/do: (B*nh, T, hd); returns (o, dq, dk, dv)."""
    q = _t(q).float().requires_grad_(True)
    k = _t(k).float().requires_grad_(True)
    v = _t(v).float().requires_grad_(True)
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    # q/k/v are (BH, T, hd), matching the device Attention layout.
    s = torch.einsum("btd,bud->btu", q, k) * scale
    if causal:
        t = q.shape[1]
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("btu,bud->btd", p, v)
    if do is None:
        return o, None, None, None
    o.backward(_t(do).float())
    return o, q.grad, k.grad, v.grad


def ref_conv2d(x, w, b, dy=None, stride=1, padding=0):
    """Reference Conv2d fwd + grads via autograd; returns (y, dx, dw, db)."""
    x = _t(x).float().requires_grad_(True)
    w = _t(w).float().clone().requires_grad_(True)
    b = _t(b).float().clone().requires_grad_(True)
    y = F.conv2d(x, w, b, stride=stride, padding=padding)
    if dy is None:
        return y, None, None, None
    y.backward(_t(dy).float())
    return y, x.grad, w.grad, b.grad
