# -*- coding: utf-8 -*-
"""profile_weight_stream.py — 权重流式读取对 GEMM 吞吐的影响

谜团：同形状单次 GEMM 382 GFLOPS，模型里同规模只有 205 GFLOPS。

假设：模型里 62 个 Linear 的权重共 122 MB，L3 只有 4 MB，
      每个 GEMM 都必须从内存流式读权重 ⇒ 吞吐掉一半。

验证：固定单个 GEMM 的形状，改变【权重集合的总大小】，看吞吐变化。
      以及：反复用同一份权重 vs 每次用新权重（模拟模型里的流式读取）。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time
import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 82)
print('权重流式读取对 GEMM 吞吐的影响')
print('=' * 82)

M = 2048
print()
print('[1] 同一份权重反复用 vs 每次换新权重')
print('    形状 (M=2048, K=512, N=512)，fp32 = 1 MB/矩阵')
print('    %-34s %10s %12s' % ('模式', 'ms', 'GFLOPS'))
print('    ' + '-' * 58)

K = N = 512
a = torch.randn(M, K)

# a) 同一权重反复用（缓存热）
w = torch.randn(K, N)
for _ in range(3):
    _ = a @ w
t0 = time.perf_counter()
for _ in range(20):
    _ = a @ w
t_hot = (time.perf_counter() - t0) / 20
print('    %-34s %10.3f %12.1f' % ('同一权重 x20（缓存热）', t_hot * 1e3, 2 * M * K * N / t_hot / 1e9))

# b) 每次换新权重（冷）
ws = [torch.randn(K, N) for _ in range(20)]
for i in range(3):
    _ = a @ ws[i]
t0 = time.perf_counter()
for i in range(20):
    _ = a @ ws[i]
t_cold = (time.perf_counter() - t0) / 20
print('    %-34s %10.3f %12.1f' % ('每次新权重 x20（冷）', t_cold * 1e3, 2 * M * K * N / t_cold / 1e9))
print('    ⇒ 冷/热 = %.2fx' % (t_cold / t_hot))

# ---- [2] 权重集合大小的影响 ----
print()
print('[2] 连续遍历 N 个不同权重矩阵（模拟模型的 62 个 Linear）')
print('    每个矩阵 (512,512) = 1 MB')
print('    %-14s %14s %14s %16s' % ('矩阵数', '总权重 MB', 'ms/矩阵', 'GFLOPS'))
print('    ' + '-' * 62)
K2 = N2 = 512
pool_sizes = (1, 4, 16, 64, 128, 256)
results = []
for npool in pool_sizes:
    ws = [torch.randn(K2, N2) for _ in range(npool)]
    # 预热整个池
    for w_ in ws[:min(4, npool)]:
        _ = a @ w_
    t0 = time.perf_counter()
    reps = 2 if npool <= 64 else 1
    for _ in range(reps):
        for w_ in ws:
            _ = a @ w_
    t = (time.perf_counter() - t0) / (reps * npool)
    mb = npool * K2 * N2 * 4 / 1e6
    gf = 2 * M * K2 * N2 / t / 1e9
    results.append((npool, mb, t, gf))
    print('    %-14d %14.1f %14.3f %16.1f' % (npool, mb, t * 1e3, gf))

print()
print('    ⇒ 若 GFLOPS 随矩阵数（总权重）下降，则【权重流式读取是真凶】')

# ---- [3] 模型真实负载模拟 ----
print()
print('[3] 模拟真实模型：62 个 Linear 的连续遍历（每次前向都过一遍）')
layers = []
for i in range(10):
    layers += [nn.Linear(512, 512, bias=False),      # q
               nn.Linear(512, 256, bias=False),      # k
               nn.Linear(512, 256, bias=False),      # v
               nn.Linear(512, 512, bias=False),      # o
               nn.Linear(512, 2048, bias=False),     # ffn gate
               nn.Linear(2048, 512, bias=False)]     # ffn down
# 补齐到 62
while len(layers) < 62:
    layers.append(nn.Linear(512, 512, bias=False))
npar = sum(L.weight.numel() for L in layers)
print('    Linear 数 %d，权重合计 %.1f MB' % (len(layers), npar * 4 / 1e6))
for L in layers:
    L.eval()
x = torch.randn(M, 512)
with torch.no_grad():
    for _ in range(1):
        for L in layers:
            if L.in_features == 512:
                _ = L(x)
    t0 = time.perf_counter()
    n = 3
    for _ in range(n):
        for L in layers:
            if L.in_features == 512:
                _ = L(x)
    t = (time.perf_counter() - t0) / n
print('    单次遍历 %.3f ms' % (t * 1e3))
tot_flops = sum(2 * M * L.in_features * L.out_features for L in layers)
print('    FLOPs %.1f GFLOP ⇒ 【%.1f GFLOPS】' % (tot_flops / 1e9, tot_flops / t / 1e9))

# ---- [4] 大 M 的影响 ----
print()
print('[4] 同一批权重，改变 M（batch 维）看吞吐是否饱和')
print('    %-12s %14s %14s' % ('M', 'ms', 'GFLOPS'))
print('    ' + '-' * 42)
for Mx in (256, 512, 1024, 2048, 4096, 8192):
    xx = torch.randn(Mx, 512)
    with torch.no_grad():
        for _ in range(1):
            for L in layers[:12]:
                if L.in_features == 512:
                    _ = L(xx)
        t0 = time.perf_counter()
        for _ in range(2):
            for L in layers[:12]:
                if L.in_features == 512:
                    _ = L(xx)
        t = (time.perf_counter() - t0) / 2
    f = sum(2 * Mx * L.in_features * L.out_features for L in layers[:12])
    print('    %-12d %14.3f %14.1f' % (Mx, t * 1e3, f / t / 1e9))

print()
print('=' * 82)
print('DONE')
