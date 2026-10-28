"""Monarch linear layer, second attempt, with the math written down before the code.

The first attempt failed its own check with an error of 6.26, and the diagnosis recorded then was a
reversed permutation: forward applied x[:, perm], which as a matrix is Q with Q[arange, perm] = 1,
while effective_weight built a different matrix. So this version defines Q once and uses that single
definition in both places, which makes the two consistent by construction rather than by luck.

    forward   y = Lm( S( Q( Rm(x) ) ) )
    weight    W = Lm @ S @ Q @ Rm           shape (m, n)

    Rm   n x n block diagonal, block b_r, from self.R
    Q    n x n, Q[arange(n), perm] = 1, so that Q @ v == v[perm]
    S    m x n selection, S[i, i] = 1 for i < min(m, n), zero elsewhere -- padding when m > n,
         truncation when m < n
    Lm   m x m block diagonal, block b_l, from self.L

Rectangular shapes are required, not optional: the model's MLP is 512 -> 2048 (via w12 at width
2736) and 2048 -> 512 (w3), and the attention projections are 512 -> 512. A layer that only handles
squares cannot replace any of them.

The check is unchanged and comes before any timing: the layer's output must equal x @ W.T to within
1e-5. If it does not, the layer is computing something other than its own weight matrix and any
speed number would be meaningless.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）



def _largest_divisor_leq(n, target):
    b = max(1, min(n, target))
    while n % b:
        b -= 1
    return b


class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, b_r=None, b_l=None, bias=True):
        super().__init__()
        n, m = in_features, out_features
        self.n, self.m = n, m
        self.b_r = _largest_divisor_leq(n, b_r or int(round(n ** 0.5)))
        self.b_l = _largest_divisor_leq(m, b_l or int(round(m ** 0.5)))
        self.nb_r = n // self.b_r
        self.nb_l = m // self.b_l
        self.R = nn.Parameter(torch.randn(self.nb_r, self.b_r, self.b_r) / self.b_r ** 0.5)
        self.L = nn.Parameter(torch.randn(self.nb_l, self.b_l, self.b_l) / self.b_l ** 0.5)
        # Q is defined once, here, and both forward and effective_weight use this exact object
        self.register_buffer('perm', self._make_perm(n), persistent=False)
        self.bias = nn.Parameter(torch.zeros(m)) if bias else None

    def _make_perm(self, n):
        """perm such that Q[arange, perm] = 1 gives Q @ v == v[perm]."""
        nb = n // self.b_r
        return torch.arange(n).view(self.b_r, nb).t().reshape(-1)

    # ---- the single definition of Q
    def _Q(self):
        n = self.n
        Q = torch.zeros(n, n, dtype=self.R.dtype)
        Q[torch.arange(n), self.perm] = 1.0
        return Q

    def _blockdiag(self, P, nb, b, size):
        M = torch.zeros(size, size, dtype=P.dtype)
        for i in range(nb):
            s = i * b
            M[s:s + b, s:s + b] = P[i]
        return M

    def _S(self):
        m, n = self.m, self.n
        S = torch.zeros(m, n, dtype=self.R.dtype)
        k = min(m, n)
        S[torch.arange(k), torch.arange(k)] = 1.0
        return S

    def effective_weight(self):
        """W = Lm @ S @ Q @ Rm, shape (m, n)."""
        Rm = self._blockdiag(self.R, self.nb_r, self.b_r, self.n)
        Lm = self._blockdiag(self.L, self.nb_l, self.b_l, self.m)
        return Lm @ self._S() @ self._Q() @ Rm

    def forward(self, x):
        lead = x.shape[:-1]
        xf = x.reshape(-1, self.n)
        # Rm @ x, blockwise
        xr = torch.einsum('abi,bji->abj', xf.view(-1, self.nb_r, self.b_r), self.R)
        xr = xr.reshape(-1, self.n)
        # Q @ xr  ==  xr[:, perm]
        xp = xr[:, self.perm]
        # S @ xp, i.e. pad when m > n, truncate when m < n
        if self.m > self.n:
            xl = torch.nn.functional.pad(xp, (0, self.m - self.n))
        elif self.m < self.n:
            xl = xp[:, :self.m]
        else:
            xl = xp
        # Lm @ xl, blockwise
        xo = torch.einsum('abi,bji->abj', xl.reshape(-1, self.nb_l, self.b_l), self.L)
        out = xo.reshape(-1, self.m)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.m)


def main():
    torch.manual_seed(0)
    print()
    print('  === Monarch 层 v2：先自检，误差 < 1e-5 才谈速度 ===')
    print()
    print('  %-14s %10s %11s %11s %9s %12s' %
          ('形状 n->m', 'b_r/b_l', '参数', 'dense', '倍数', '自检误差'))
    print('  ' + '-' * 74)
    ok_all = True
    cases = [(512, 512), (512, 2048), (2048, 512), (2048, 2736), (2736, 512)]
    for (n, m) in cases:
        lay = MonarchLinear(n, m)
        W = lay.effective_weight()
        x = torch.randn(4, n)
        with torch.no_grad():
            y = lay(x)
            y_ref = x @ W.t() + (lay.bias if lay.bias is not None else 0)
        err = float((y - y_ref).abs().max())
        p_m = lay.R.numel() + lay.L.numel()
        p_d = n * m
        flag = '✓' if err < 1e-5 else '✗'
        if err >= 1e-5:
            ok_all = False
        print('  %-14s %10s %11d %11d %8.1fx %10.2e %s'
              % ('%d->%d' % (n, m), '%d/%d' % (lay.b_r, lay.b_l), p_m, p_d, p_d / p_m, err, flag))
    print()
    print('  判据: 误差 < 1e-5 ⇒ 层算的是它自己的权重矩阵 ⇒ 才能开始计时')
    print('  ⇒ %s' % ('【全部通过，可以进入计时】✓' if ok_all else '【仍有不通过，先修】✗'))


if __name__ == '__main__':
    main()
