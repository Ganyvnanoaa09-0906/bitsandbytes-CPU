# -*- coding: utf-8 -*-
"""measure_fwd_bwd2.py — 反向 GEMM 为什么只有前向的一半？

[1] 已发现：反向的 dx/dW GEMM 只有 95~111 GFLOPS，前向有 187~236。

本轮区分两个原因：
  A) autograd 引擎开销（不是 GEMM 本身慢）
  B) GEMM 形状本身效率低（dx = dout @ W 的 K 维是 N=2048，dW = dout^T @ x）

方法：纯 GEMM 直接测同形状，不经 autograd。
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
import sys, time
import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 84)
print('反向 GEMM 为什么只有前向的一半')
print('=' * 84)

M = 2048
cases = [('FFN gate', 512, 2048), ('FFN down', 2048, 512), ('QKV', 512, 512),
         ('Out head', 512, 1024)]


def bench(fn, n=5):
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print()
print('[1] 纯 GEMM（不经 autograd）的三种形状')
print('    前向   y  = x @ W^T     x(M,K)  W(N,K)  -> x @ W.t()')
print('    dx     dx = dout @ W    dout(M,N) W(N,K) -> dout @ W')
print('    dW     dW = dout^T @ x  dout^T(N,M) x(M,K) -> dout.t() @ x')
print()
print('    %-12s %10s %10s %10s %10s' % ('形状 K->N', 'fwd', 'dx', 'dW', 'dW(连续)'))
print('    ' + '-' * 58)
for name, K, N in cases:
    x = torch.randn(M, K)
    W = torch.randn(N, K)
    Wt = W.t().contiguous()
    dout = torch.randn(M, N)
    dout_t = dout.t().contiguous()
    fl = 2 * M * K * N

    t_fwd = bench(lambda: x @ Wt)
    t_dx = bench(lambda: dout @ W)
    t_dW_t = bench(lambda: dout.t() @ x)
    t_dW_c = bench(lambda: dout_t @ x)
    print('    %-12s %10.1f %10.1f %10.1f %10.1f' % (
        '%d->%d' % (K, N), fl / t_fwd / 1e9, fl / t_dx / 1e9,
        fl / t_dW_t / 1e9, fl / t_dW_c / 1e9))

print()
print('    ⇒ 若纯 GEMM 的 dx/dW 也慢，则是【形状问题】；若纯 GEMM 快，则是【autograd 开销】')

print()
print('[2] autograd 开销量化（同形状，加减 autograd）')
print('    %-12s %14s %14s %14s' % ('形状', '纯GEMM(ms)', 'autograd(ms)', '开销 ms'))
print('    ' + '-' * 58)
for name, K, N in cases[:3]:
    lin = nn.Linear(K, N, bias=False)
    x = torch.randn(M, K, requires_grad=True)
    fl = 2 * M * K * N

    def pure():
        with torch.no_grad():
            return lin(x)

    t_pure = bench(pure)

    def with_grad():
        y = lin(x)
        return y

    t_grad = bench(with_grad)
    print('    %-12s %14.3f %14.3f %14.3f' % (
        '%d->%d' % (K, N), t_pure * 1e3, t_grad * 1e3, (t_grad - t_pure) * 1e3))

print()
print('=' * 84)
print('DONE')
