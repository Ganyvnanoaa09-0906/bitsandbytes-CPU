# -*- coding: utf-8 -*-
"""bench_qkv.py — QKV 投影融合值不值得改？

现状：三个独立 Linear(512->512), (512->256), (512->256)
融合：一个 Linear(512->1024) 再切片

融合的好处：
  1) 3 次 kernel 启动 -> 1 次（Python/dispatch 开销降 2/3）
  2) GEMM 的 N 维度 512/256/256 -> 1024，效率更高

代价：多一次切片（split 是 view，几乎免费）

同时测：RMSNorm 的开销，以及整个 block 的构成。
"""
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
import time
import torch
import torch.nn as nn

torch.set_num_threads(6)
D, NH, NKV, HD = 512, 8, 4, 64
KVH = HD


def bench(fn, iters=60, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


q = nn.Linear(D, NH * HD, bias=False)
k = nn.Linear(D, NKV * KVH, bias=False)
v = nn.Linear(D, NKV * KVH, bias=False)
fused = nn.Linear(D, NH * HD + 2 * NKV * KVH, bias=False)


def sep(x):
    return q(x), k(x), v(x)


def fus(x):
    y = fused(x)
    return y.split([NH * HD, NKV * KVH, NKV * KVH], dim=-1)


print('=' * 62, flush=True)
print('QKV 融合微基准 (d_model=%d, threads=6)' % D, flush=True)
print('=' * 62, flush=True)
print('  %-12s %12s %12s %10s' % ('batch*T', '分开 (ms)', '融合 (ms)', '提升'), flush=True)

for bt in (256, 512, 1024, 2048, 4096):
    x = torch.randn(bt, D)
    ta = bench(lambda: sep(x))
    tb = bench(lambda: fus(x))
    print('  %-12d %12.4f %12.4f %9.1f%%'
          % (bt, ta * 1e3, tb * 1e3, 100 * (ta - tb) / ta), flush=True)

print()
print('--- 单次调用的固定开销（batch 很小时暴露 dispatch 成本）---', flush=True)
x1 = torch.randn(1, D)
print('  分开 3 次调用: %.4f ms' % (bench(lambda: sep(x1), iters=200) * 1e3), flush=True)
print('  融合 1 次调用: %.4f ms' % (bench(lambda: fus(x1), iters=200) * 1e3), flush=True)

print()
print('--- RMSNorm 开销（每层 2 次，都是 hd=64 的小张量）---', flush=True)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


rn = RMSNorm(HD)
for bt in (256, 1024, 4096):
    x = torch.randn(bt, NH, HD)
    t = bench(lambda: rn(x))
    print('  b*T=%-5d %.4f ms' % (bt, t * 1e3), flush=True)

print()
print('DONE', flush=True)
