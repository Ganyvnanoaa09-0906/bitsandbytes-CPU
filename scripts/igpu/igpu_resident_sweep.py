"""Sweep the layer size and batch to find where device residency actually wins.

7.103 found the CPU 1.91x faster on a 512x512 layer at batch 1024, and 7.97 had already predicted why:
the iGPU needs large N and large M before its parallelism is fed. So the question is not whether the
device path is fast, it is where the crossover is, and that is a two-dimensional question because N
sets the parameter count (what residency saves moving) and B sets the batch (what it still has to move).

Four points along each axis, on the same two step functions used in 7.103 so the comparison is the one
already validated -- CPU torch against the OpenCL GEMM plus the verified Adam kernel with everything
resident. Nothing else changes.

Reported as a table rather than a summary ratio, because a single number here would be exactly the
mistake this thread has been avoiding: the answer is a region, not a value, and the region's boundary is
what matters.

Criterion, fixed in advance: device residency wins when its median step time is below the CPU's by more
than the machine's 20% measurement resolution. Below that the two are equivalent and the CPU should be
used, since it involves no host traffic at all.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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

exec(open(r'D:\work\bitsandbytes-CPU\igpu_resident_step.py', encoding='utf-8')
     .read().split('# ---- 配置')[0])

torch.manual_seed(0)
lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8
STEPS = 8
torch.set_num_threads(5)


def cpu_arm(N, K, B):
    Xt = torch.randn(B * 4, K)
    Yt = Xt @ (torch.randn(N, K) * 0.1).t()
    p = torch.randn(N, K) * 0.05
    m = torch.zeros(N, K)
    v = torch.zeros(N, K)
    ts = []
    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, Xt.shape[0], (B,))
        xb, yb = Xt[idx], Yt[idx]
        y = xb @ p.t()
        g = 2.0 * (y - yb) / (B * N)
        dw = g.t() @ xb
        m = b1 * m + (1 - b1) * dw
        v = b2 * v + (1 - b2) * dw * dw
        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
        p = p - lr * (m / c1) / ((v / c2).sqrt() + eps)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def dev_arm(N, K, B):
    Xt = torch.randn(B * 4, K)
    Yt = Xt @ (torch.randn(N, K) * 0.1).t()
    W0 = torch.randn(N, K) * 0.05
    w = buf(N * K, ro=False); wT = buf(K * N, ro=False)
    mt = buf(N * K, ro=False); vt = buf(N * K, ro=False)
    dw = buf(N * K, ro=False)
    xa = buf(B * K, ro=False); ya = buf(B * N, ro=False); ga = buf(N * B, ro=False)
    up(w, W0.numpy()); up(wT, np.ascontiguousarray(W0.numpy().T))
    up(mt, np.zeros((N, K))); up(vt, np.zeros((N, K)))
    STAT['to_dev'] = STAT['to_host'] = 0
    ts = []
    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()
        idx = torch.randint(0, Xt.shape[0], (B,))
        xb, yb = Xt[idx], Yt[idx]
        up(xa, xb.numpy())
        gemm(B, N, K, xa, wT, ya)
        yn = down(ya, (B, N))
        gy = 2.0 * (torch.from_numpy(yn) - yb) / (B * N)
        up(ga, np.ascontiguousarray(gy.numpy().T))
        gemm(N, K, B, ga, xa, dw)
        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
        k_adam(_q, ((N * K + 255) // 256 * 256,), None,
               np.int32(N), np.int32(K), w, wT, mt, vt, dw,
               np.float32(lr), np.float32(b1), np.float32(b2),
               np.float32(c1), np.float32(c2), np.float32(eps))
        _q.finish()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, (STAT['to_dev'] + STAT['to_host']) / STEPS / 1e6


CONFIGS = [
    (512, 512, 1024),
    (1024, 1024, 1024),
    (2048, 2048, 1024),
    (2048, 2048, 4096),
    (4096, 1024, 4096),
]
print()
print('  %-18s %10s %12s %12s %10s %12s' %
      ('N × K × B', '参数 MB', 'CPU ms', '设备 ms', '设备/CPU', '设备流量 MB'))
print('  ' + '-' * 82)
wins = []
for (N, K, B) in CONFIGS:
    try:
        c = cpu_arm(N, K, B)
        d, mb = dev_arm(N, K, B)
        ratio = d / c
        flag = '★ 设备赢 %.2f×' % (c / d) if d < c * 0.8 else ('平' if ratio < 1.2 else 'CPU 赢 %.2f×' % ratio)
        if d < c * 0.8:
            wins.append((N, K, B, c / d))
        print('  %-18s %10.1f %12.2f %12.2f %10.2f  %s'
              % ('%dx%dx%d' % (N, K, B), N * K * 4 / 1e6, c, d, ratio, flag))
    except Exception as e:
        print('  %-18s %10.1f %12s %12s %10s  %s: %s'
              % ('%dx%dx%d' % (N, K, B), N * K * 4 / 1e6, 'FAIL', '-', '-',
                 type(e).__name__, str(e)[:34]))
print()
if wins:
    print('  ⇒ ★【设备常驻赢的区间: %s】' % ', '.join('%dx%dx%d (%.2f×)' % w for w in wins))
else:
    print('  ⇒ ✗ 这五个点里设备都没有超过 CPU 20%')
print()
print('  ⇒ 判据（事先固定）: 设备 < CPU × 0.8 ⇒ 设备赢（本机分辨力约 20%）')
print('  ⇒ ★ 而 §7.97 的 M 扫描说核显要【大 M、大 N】才赢 —— 所以这张表是在补它缺的那两个维度')
print()
