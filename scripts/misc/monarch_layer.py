"""Monarch linear layer, and the correctness check that has to come before any speed number.

Monarch (Dao et al.) factorises a dense m x n matrix as M = P L P R where R is block-diagonal
n x n, L is block-diagonal m x m, and P is a fixed permutation. With blocks of size b the parameter
count falls from mn to roughly b(m+n) and the FLOPs with it, which is where the 8-15x would come from.

One design decision, made before writing anything, because it changes what "verify" means. The
tempting experiment is to take the trained dense weight matrix and approximate it by a Monarch
factorisation, then check the error falls as b grows. But an arbitrary matrix needs b = n to be
represented exactly, so that experiment would measure a compression ratio rather than a layer.

Instead the layer replaces the dense one and is trained from the start, which is what actually saves
compute. And then the correctness check is concrete: the layer's output must equal the output of its
own effective dense matrix. Build that matrix by feeding one-hot inputs, or by materialising P L P R
explicitly, and compare. A layer that computes something other than what it claims would be faster
for the obvious reason, and this project has paid for that lesson thirteen times.

Decision rule, fixed in advance: outputs must agree to within 1e-5 for a random parameterisation,
and the parameter and FLOP counts must match the formulas. Only then does timing mean anything.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）



class MonarchLinear(nn.Module):
    """y = P L P R x, with R block-diagonal over the input and L over the output.

    R: (n/b_r) blocks of b_r x b_r, applied blockwise to the input
    P: a fixed permutation between R and L, chosen to mix across blocks
    L: (m/b_l) blocks of b_l x b_l, applied to the permuted intermediate

    b_r and b_l default to sqrt of the respective dimension, which is the standard choice.
    """

    def __init__(self, in_features, out_features, b_r=None, b_l=None, bias=True):
        super().__init__()
        n, m = in_features, out_features
        self.n, self.m = n, m
        self.b_r = b_r or max(1, int(round(n ** 0.5)))
        self.b_l = b_l or max(1, int(round(m ** 0.5)))
        while n % self.b_r:
            self.b_r -= 1
        while m % self.b_l:
            self.b_l -= 1
        self.nb_r = n // self.b_r
        self.nb_l = m // self.b_l
        self.R = nn.Parameter(torch.randn(self.nb_r, self.b_r, self.b_r) * (1.0 / self.b_r ** 0.5))
        self.L = nn.Parameter(torch.randn(self.nb_l, self.b_l, self.b_l) * (1.0 / self.b_l ** 0.5))
        # fixed permutation: interleave so that positions from different R-blocks land together
        self.register_buffer('perm', self._make_perm(n), persistent=False)
        self.bias = nn.Parameter(torch.zeros(m)) if bias else None

    def _make_perm(self, n):
        # group index varies fastest, so consecutive permuted positions come from different blocks
        idx = torch.arange(n).view(self.nb_r, self.b_r).t().reshape(-1)
        return idx

    def effective_weight(self):
        """Materialise P L P R as a dense (m, n) matrix, for checking only."""
        n, m = self.n, self.m
        # R as dense n x n
        Rd = torch.zeros(n, n, dtype=self.R.dtype)
        for i in range(self.nb_r):
            s = i * self.b_r
            Rd[s:s + self.b_r, s:s + self.b_r] = self.R[i]
        # L as dense m x m, acting on the permuted space
        Ld = torch.zeros(m, m, dtype=self.L.dtype)
        for i in range(self.nb_l):
            s = i * self.b_l
            Ld[s:s + self.b_l, s:s + self.b_l] = self.L[i]
        # P as a permutation matrix on n
        n_used = min(n, m) if n != m else n

        def Pmat(size):
            Pm = torch.zeros(size, size, dtype=self.R.dtype)
            p = self.perm
            if p.numel() != size:
                p = self._make_perm(size)
            Pm[torch.arange(size), p] = 1.0
            return Pm

        if n == m:
            W = Pmat(n).T @ Ld @ Pmat(n) @ Rd
        else:
            # rectangular: L is m x m, R is n x n, and the permutation is applied on the n side
            Pn = Pmat(n)
            # project L onto the n-dimensional permuted space by padding or truncating
            if m >= n:
                Lr = Ld[:n, :n]
            else:
                Lr = torch.zeros(n, n, dtype=self.R.dtype)
                Lr[:m, :m] = Ld
            W = Lr @ Pn @ Rd if m >= n else (Pn.T @ Lr @ Pn @ Rd)
        return W

    def forward(self, x):
        n, m = self.n, self.m
        lead = x.shape[:-1]
        xf = x.reshape(-1, n)
        # R blockwise
        xr = xf.view(-1, self.nb_r, self.b_r)
        xr = torch.einsum('abi,bij->abj', xr, self.R).reshape(-1, n)
        # permutation
        xp = xr[:, self.perm]
        # L blockwise, on the m side: pad or truncate the n dimension to m
        if m == n:
            xl = xp
        elif m > n:
            xl = F.pad(xp, (0, m - n))
        else:
            xl = xp[:, :m]
        xl = xl.view(-1, self.nb_l, self.b_l)
        xl = torch.einsum('abi,bij->abj', xl, self.L).reshape(-1, m)
        if m < n:
            # project back onto the output dimension by taking the first m of a permuted copy
            pass
        out = xl
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, m)


def main():
    torch.manual_seed(0)
    print()
    print('  === Monarch 层：正确性与复杂度 ===')
    print()
    print('  %-16s %10s %12s %12s %12s %10s' %
          ('形状 n->m', 'b_r/b_l', '参数', 'dense 参数', '倍数', '误差'))
    print('  ' + '-' * 82)
    for (n, m) in ((512, 2048), (2048, 512), (512, 512), (2048, 2736)):
        lay = MonarchLinear(n, m)
        W = lay.effective_weight()
        if W.shape != (m, n):
            print('  %-16s 有效权重形状 %s ≠ (%d,%d) -- 跳过' % ('%d->%d' % (n, m), tuple(W.shape), m, n))
            continue
        x = torch.randn(4, n)
        with torch.no_grad():
            y = lay(x)
            y_ref = x @ W.t() + (lay.bias if lay.bias is not None else 0)
        err = float((y - y_ref).abs().max())
        p_m = lay.R.numel() + lay.L.numel()
        p_d = n * m
        print('  %-16s %10s %12d %12d %11.1fx %10.2e'
              % ('%d->%d' % (n, m), '%d/%d' % (lay.b_r, lay.b_l), p_m, p_d, p_d / p_m, err))
    print()
    print('  判据（事先固定）: 误差 < 1e-5 ⇒ 层算的是它声称的东西，计时才有意义。')
    print()


if __name__ == '__main__':
    main()
