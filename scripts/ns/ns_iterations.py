"""ns_iterations.py -- is 5 Newton-Schulz iterations necessary, or would 3 do?

NS is 2.42 s of Muon's 6.42 s step (7.27.3) and the optimisation attempts -- batching, full
device residency, true async -- all made it slower (7.29.2). The remaining lever is doing
less of it. Muon orthogonalises each 2-D weight with five quintic iterations; four or three
might reach the same orthogonality, which would cut the dominant cost proportionally.

Checked numerically first because it is a minute instead of three hours: orthogonalise
random matrices at the shapes the model uses and measure how close X^T X comes to the
identity. If three iterations land as close as five, a training A/B is worth running; if not,
the question is closed without spending the time.

Residual is reported as ||X^T X - I||_F / sqrt(m), which is scale-free across shapes, plus
the largest singular-value deviation from 1 -- the quantity orthogonalisation exists to
control.
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

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

torch.set_num_threads(5)


def ns(G, steps, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


def quality(X):
    """How far X is from having orthonormal rows."""
    m = X.shape[0]
    G = X @ X.T
    I = torch.eye(m)
    resid = float((G - I).norm()) / (m ** 0.5)
    sv = torch.linalg.svdvals(X)
    return resid, float((sv - 1).abs().max())


SHAPES = [(512, 512), (512, 256), (2048, 512)]
print('%-12s %-8s %12s %12s' % ('形状', '迭代', '残差', '奇异值偏离'))
print('-' * 50)
for shape in SHAPES:
    # average over a few matrices, since the answer depends on the spectrum
    acc = {}
    for _ in range(3):
        G = torch.randn(*shape)
        for steps in (3, 4, 5, 6):
            r, s = quality(ns(G, steps))
            a = acc.setdefault(steps, [0.0, 0.0])
            a[0] += r / 3
            a[1] += s / 3
    base = acc[5][0]
    for steps in (3, 4, 5, 6):
        r, s = acc[steps]
        print('%-12s %-8d %12.3e %12.3e   %s'
              % ('%dx%d' % shape, steps, r, s,
                 '' if steps == 5 else '%.2fx vs 5 次' % (r / base)))
    print()

print('判据: 3 次的残差若与 5 次同量级（例如 2 倍以内）⇒ 值得跑训练 A/B；')
print('      若差一个数量级 ⇒ NS 的迭代数不能减，问题关闭。')
