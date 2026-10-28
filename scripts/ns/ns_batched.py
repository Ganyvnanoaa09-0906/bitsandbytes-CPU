"""ns_batched.py -- run Muon's orthogonalisation as batched matmuls.

Newton-Schulz is 2.42 s of a 6.42 s Muon step, measured clean, i.e. 38% of it -- the single
largest removable cost in the optimiser. The reason it is that expensive is visible in the
loop: 74 matrices, five iterations each, three matmuls per iteration, all done one matrix
at a time. Small matmuls on the CPU run well below the rate large ones achieve (218 GFLOPS
on the model's real shapes against 315 on a bigger single one), and the per-call overhead
is paid 1110 times per step.

The matrices are not all different. For this model the 2-D weights fall into three shape
families -- 512x512, 512x256, 2048x512 -- repeated across ten layers. Stacking each family
into a 3-D tensor turns 74 small matmuls into 3 batched ones per iteration, which is exactly
what torch.bmm exists for and where the CPU is efficient.

Correctness is checked against the existing per-matrix implementation, elementwise, because
a batched rewrite that agrees only in aggregate would be easy to get subtly wrong. Two
things need care:

  * the transpose. The loop transposes any matrix with rows > cols before iterating; the
    batched version has to do the same per group.
  * the scaling. The loop multiplies each update by sqrt(max(1, rows/cols)) after
    orthogonalising; with a group this depends on the group's shape, not the matrix's.

Timed on the shapes the model actually uses. If batching does not help, that is worth
knowing too -- it would mean the cost is in the matmul arithmetic rather than the call
overhead, and the answer is fewer iterations instead.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


torch.set_num_threads(6)

# Muon's shapes for this model: per layer Q(512,512) K(512,256) V(512,256) O(512,512)
# and FFN gate/up(512,2048) down(2048,512), ten layers
SHAPES = []
for _ in range(10):
    SHAPES += [(512, 512), (512, 256), (512, 256), (512, 512),
               (512, 2048), (512, 2048), (2048, 512)]
SHAPES = SHAPES[:74]


def ns_one(G, steps=5, eps=1e-7):
    """The existing per-matrix implementation, from bench_muon.py."""
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


def ns_batched(weights, steps=5, eps=1e-7):
    """Same maths, grouped by shape and run as batched matmuls."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    groups = defaultdict(list)
    order = []
    for i, w in enumerate(weights):
        groups[tuple(w.shape)].append(i)
        order.append(tuple(w.shape))

    out = [None] * len(weights)
    for shape, idxs in groups.items():
        X = torch.stack([weights[i] for i in idxs]).float()        # (B, m0, n0)
        m0, n0 = shape
        tr = m0 > n0
        if tr:
            X = X.transpose(1, 2).contiguous()
        m, n = X.shape[1], X.shape[2]
        # normalise each matrix by its own norm, as the loop does per matrix
        norms = X.reshape(len(idxs), -1).norm(dim=1).clamp_min(eps)
        X = X / norms.view(-1, 1, 1)
        for _ in range(steps):
            A = X @ X.transpose(1, 2)                              # (B, m, m)
            Bm = b * A + c * (A @ A)
            X = a * X + Bm @ X
        if tr:
            X = X.transpose(1, 2)
        # the loop scales by sqrt(max(1, rows/cols)) per matrix, which for a group is
        # the group's shape
        scale = max(1.0, m / n) ** 0.5
        X = X * scale
        for k, i in enumerate(idxs):
            out[i] = X[k]
    return out


weights = [torch.randn(*s) for s in SHAPES]
print('矩阵 %d 个，%.1fM 参数，形状族 %d 个'
      % (len(SHAPES), sum(m * n for m, n in SHAPES) / 1e6,
         len({tuple(s) for s in SHAPES})))
fams = defaultdict(int)
for s in SHAPES:
    fams[s] += 1
for s, k in sorted(fams.items()):
    print('  %-14s × %d' % ('%dx%d' % s, k))
print()

# ---- correctness first
ref = [ns_one(w) for w in weights]
got = ns_batched(weights)
worst = 0.0
for r, g in zip(ref, got):
    d = float((g - r).abs().max()) / (float(r.abs().max()) or 1.0)
    worst = max(worst, d)
print('批量版与逐矩阵版的【最大相对差: %.3e  %s'
      % (worst, '✓ 一致' if worst < 1e-4 else '✗ 不一致'))
print()


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


t_one = timeit(lambda: [ns_one(w) for w in weights])
t_bat = timeit(lambda: ns_batched(weights))
print('  逐矩阵版（现状）   %7.3f s' % t_one)
print('  批量版            %7.3f s   ⇒ %.2fx' % (t_bat, t_one / t_bat))
print()
print('判据: 若批量版显著更快且结果一致 ⇒ 替换进 Muon。')
print('      若几乎相同 ⇒ 开销在算术而非调用，应减少迭代次数（见 ns_fewer_steps）。')
