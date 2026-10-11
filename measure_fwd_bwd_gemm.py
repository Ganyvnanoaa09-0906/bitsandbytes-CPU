# -*- coding: utf-8 -*-
"""measure_fwd_bwd_gemm.py — 前向 vs 反向 GEMM 的效率差

假设：真实模型 GEMM 只有 194 GFLOPS（纯前向 Linear 序列能到 300.8），
      是因为【反向的 GEMM 形状更差】。

反向的三种 GEMM：
  dx = dout @ W^T      形状 (M,N)@(N,K)  —— W^T 是转置视图，访问不连续
  dW = dout^T @ x      形状 (N,M)@(M,K)  —— 需要累加
"""
import sys, time
import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 84)
print('前向 vs 反向 GEMM 的效率差')
print('=' * 84)

M = 2048
cases = [
    ('FFN gate  d=512 -> h=2048', 512, 2048),
    ('FFN down  h=2048 -> d=512', 2048, 512),
    ('Attn QKV  d=512 -> 512', 512, 512),
    ('Attn O    d=512 -> 512', 512, 512),
    ('Out head  d=512 -> 1024', 512, 1024),
]

print()
print('[1] 前向 Linear  vs  反向（dx / dW），单位 GFLOPS')
print('    %-30s %10s %10s %10s %10s' % ('形状', 'fwd', 'bwd(dx)', 'bwd(dW)', 'bwd总'))
print('    ' + '-' * 74)

for name, K, N in cases:
    lin = nn.Linear(K, N, bias=False)
    x = torch.randn(M, K, requires_grad=True)
    fl = 2 * M * K * N

    def fwd_only():
        return lin(x)

    for _ in range(2):
        _ = fwd_only()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = fwd_only()
    t_fwd = (time.perf_counter() - t0) / 5

    def dx_only():
        x.grad = None
        y = lin(x)
        torch.autograd.backward(y, torch.ones_like(y), inputs=[x])

    for _ in range(2):
        dx_only()
    t0 = time.perf_counter()
    for _ in range(5):
        dx_only()
    t_dx = (time.perf_counter() - t0) / 5

    xd = x.detach()

    def dw_only():
        lin.weight.grad = None
        y = lin(xd)
        torch.autograd.backward(y, torch.ones_like(y), inputs=[lin.weight])

    for _ in range(2):
        dw_only()
    t0 = time.perf_counter()
    for _ in range(5):
        dw_only()
    t_dW = (time.perf_counter() - t0) / 5

    print('    %-30s %10.1f %10.1f %10.1f %10.1f' % (
        name, fl / t_fwd / 1e9, fl / t_dx / 1e9, fl / t_dW / 1e9,
        fl / (t_dx + t_dW) / 1e9))

print()
print('    ⇒ 若 bwd 明显低于 fwd，则【反向 GEMM 是主攻方向】')

print()
print('[2] dx = dout @ W^T ：转置访问 vs 预存转置权重')
print('    %-30s %16s %16s %12s' % ('形状', 'W^T(转置视图)', 'Wt(预存contiguous)', '加速'))
print('    ' + '-' * 80)
for name, K, N in cases[:3]:
    W = torch.randn(N, K)   # nn.Linear 布局 (out= N, in=K)
    Wt = W.t().contiguous()   # (K,N)
    dout = torch.randn(M, N)
    for _ in range(2):
        _ = dout @ W
    t0 = time.perf_counter()
    for _ in range(5):
        _ = dout @ W
    t_t = (time.perf_counter() - t0) / 5
    for _ in range(2):
        _ = dout @ Wt
    t0 = time.perf_counter()
    for _ in range(5):
        _ = dout @ Wt
    t_c = (time.perf_counter() - t0) / 5
    fl = 2 * M * K * N
    print('    %-30s %11.1f GF   %12.1f GF   %11.2fx' % (name, fl / t_t / 1e9, fl / t_c / 1e9, t_t / t_c))

print()
print('[3] 真实规模完整前后向（62 个 Linear，117 MB 权重）')
layers = []
for i in range(10):
    layers += [nn.Linear(512, 512, bias=False), nn.Linear(512, 256, bias=False),
               nn.Linear(512, 256, bias=False), nn.Linear(512, 512, bias=False),
               nn.Linear(512, 2048, bias=False), nn.Linear(2048, 512, bias=False)]
while len(layers) < 62:
    layers.append(nn.Linear(512, 512, bias=False))
tot_fl = sum(2 * M * L.in_features * L.out_features for L in layers)
x = torch.randn(M, 512, requires_grad=True)


def run_fwd():
    h = x
    for L in layers:
        if L.in_features == 512:
            h = L(h)
    return h


def run_fb():
    h = run_fwd()
    torch.autograd.backward(h, torch.ones_like(h))
    x.grad = None
    for L in layers:
        L.weight.grad = None


for _ in range(1):
    h = run_fwd()
t0 = time.perf_counter()
for _ in range(3):
    h = run_fwd()
t_fwd = (time.perf_counter() - t0) / 3
print('    前向        %8.3f ms   %8.1f GFLOPS' % (t_fwd * 1e3, tot_fl / t_fwd / 1e9))

for _ in range(1):
    run_fb()
t0 = time.perf_counter()
for _ in range(3):
    run_fb()
t_fb = (time.perf_counter() - t0) / 3
print('    前向+反向   %8.3f ms' % (t_fb * 1e3))
print('    其中反向    %8.3f ms   %8.1f GFLOPS（按 2x FLOPs 算）'
      % ((t_fb - t_fwd) * 1e3, 2 * tot_fl / (t_fb - t_fwd) / 1e9))

print()
print('=' * 84)
print('DONE')
