"""CPU step versus device-resident step on the same layer and batch. The last empty cell.

Everything else in this thread is verified: forward, backward and Adam all produce results matching the
CPU, and device residency moved per-step host traffic from O(NK) to O(BK+BN). What was never measured is
whether any of it is faster, and 7.99 already showed a case where the iGPU path was 3.1x slower because
weights were re-uploaded every step.

That case is exactly the one device residency removes, so this is the measurement that decides whether
the work was worth it. Same layer, same batch, same optimizer, same number of steps. The CPU arm uses
torch throughout; the device arm uses the OpenCL GEMM and the verified Adam kernel with weights, moments
and dW all resident.

Reported honestly rather than as a ratio: the mean, the median, and the spread, plus the per-step host
traffic each arm causes. A ratio alone would hide that the two arms move different amounts of data and
that this machine's timing resolution is around 20%.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pyopencl as cl  # noqa: E402

exec(open(r'_os.path.join(_root, 'scripts\\igpu\\igpu_resident_step.py')', encoding='utf-8')
     .read().split('# ---- 配置')[0])

torch.manual_seed(0)
N = K = 512
B = 1024
lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
STEPS = 12

W_true = torch.randn(N, K) * 0.1
Xt = torch.randn(4096, K)
Yt = Xt @ W_true.t()

print()
print('  === CPU 每步（torch 全流程）===')
torch.set_num_threads(5)
pc = torch.randn(N, K) * 0.05
mc = torch.zeros(N, K)
vc = torch.zeros(N, K)
ct = []
for step in range(1, STEPS + 1):
    t0 = time.perf_counter()
    idx = torch.randint(0, Xt.shape[0], (B,))
    xb, yb = Xt[idx], Yt[idx]
    y = xb @ pc.t()
    loss = ((y - yb) ** 2).mean()
    g = 2.0 * (y - yb) / (B * N)
    dWc = g.t() @ xb
    mc = b1 * mc + (1 - b1) * dWc
    vc = b2 * vc + (1 - b2) * dWc * dWc
    c1, c2 = 1 - b1 ** step, 1 - b2 ** step
    pc = pc - lr * (mc / c1) / ((vc / c2).sqrt() + eps)
    ct.append(time.perf_counter() - t0)
print('    中位 %.2f ms · 均值 %.2f ms · 离散 %.1f%%'
      % (statistics.median(ct) * 1e3, statistics.mean(ct) * 1e3,
         statistics.pstdev(ct) / statistics.mean(ct) * 100))

print()
print('  === 设备常驻每步（GEMM + Adam 全在核显，权重/m/v/dW 常驻）===')
W0 = torch.randn(N, K) * 0.05
w = buf(N * K, ro=False); wT = buf(K * N, ro=False)
mt = buf(N * K, ro=False); vt = buf(N * K, ro=False)
dW = buf(N * K, ro=False)
xa = buf(B * K, ro=False); ya = buf(B * N, ro=False); gya = buf(B * N, ro=False)
up(w, W0.numpy()); up(wT, np.ascontiguousarray(W0.numpy().T))
up(mt, np.zeros((N, K))); up(vt, np.zeros((N, K)))

STAT['to_dev'] = STAT['to_host'] = 0
dt = []
for step in range(1, STEPS + 1):
    t0 = time.perf_counter()
    idx = torch.randint(0, Xt.shape[0], (B,))
    xb, yb = Xt[idx], Yt[idx]
    up(xa, xb.numpy())
    gemm(B, N, K, xa, wT, ya)
    yn = down(ya, (B, N))
    gy = 2.0 * (torch.from_numpy(yn) - yb) / (B * N)
    gyaT = buf(N * B, ro=False)
    up(gyaT, np.ascontiguousarray(gy.numpy().T))
    gemm(N, K, B, gyaT, xa, dW)
    c1, c2 = 1 - b1 ** step, 1 - b2 ** step
    k_adam(_q, ((N * K + 255) // 256 * 256,), None,
           np.int32(N), np.int32(K), w, wT, mt, vt, dW,
           np.float32(lr), np.float32(b1), np.float32(b2),
           np.float32(c1), np.float32(c2), np.float32(eps))
    _q.finish()
    dt.append(time.perf_counter() - t0)
print('    中位 %.2f ms · 均值 %.2f ms · 离散 %.1f%%'
      % (statistics.median(dt) * 1e3, statistics.mean(dt) * 1e3,
         statistics.pstdev(dt) / statistics.mean(dt) * 100))
print('    主机流量 %.2f MB/步（上 %.2f · 下 %.2f）'
      % ((STAT['to_dev'] + STAT['to_host']) / STEPS / 1e6,
         STAT['to_dev'] / STEPS / 1e6, STAT['to_host'] / STEPS / 1e6))

mc_, md_ = statistics.median(ct), statistics.median(dt)
print()
print('  === 对照 ===')
print('    CPU    %.2f ms/步' % (mc_ * 1e3))
print('    设备   %.2f ms/步' % (md_ * 1e3))
print('    ⇒ %s' % ('★【设备快 %.2f×】' % (mc_ / md_) if md_ < mc_ else 'CPU 快 %.2f×' % (md_ / mc_)))
print()
print('  ★ 而这一层是 512×512、batch 1024 ⇒ 参数只有 1 MB、每步的算术强度很低。')
print('    ⇒ ★【而 §7.97 的换手结论说:核显要【大 N、大 M】才赢 —— 所以这个结果只对这一个形状成立。】')
print()
