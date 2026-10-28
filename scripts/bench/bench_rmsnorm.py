# -*- coding: utf-8 -*-
"""bench_rmsnorm.py — 手写 RMSNorm vs PyTorch 融合版 F.rms_norm。

实测手写版每次调用要 ~2ms（对 100 万元素张量，正常应 ~0.1ms）。
原因：6 个分离算子，每个都是一次 kernel 启动 + 6 线程同步。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(6)


def bench(fn, iters=200, warm=30):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


class HandRMS(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MulRMS(nn.Module):
    """用 x*x 代替 pow(2)，少一层临时量"""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.mul(x).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.w


print('=' * 64, flush=True)
print('RMSNorm 实现对比 (threads=6)', flush=True)
print('=' * 64, flush=True)
print('  F.rms_norm 可用:', hasattr(F, 'rms_norm'), flush=True)

for d in (64, 512):
    print()
    print('--- normalized dim = %d ---' % d, flush=True)
    hand = HandRMS(d)
    mulv = MulRMS(d)
    print('  %-22s %10s %10s %10s' % ('形状', '手写(ms)', 'x*x版(ms)', 'Fused(ms)'), flush=True)
    for shape in ((8, 256, 8, d), (8, 256, d), (2048, d), (8, 262144)):
        if shape[-1] != d:
            continue
        x = torch.randn(*shape)
        t_hand = bench(lambda: hand(x))
        t_mul = bench(lambda: mulv(x))
        t_fused = float('nan')
        if hasattr(F, 'rms_norm'):
            try:
                t_fused = bench(lambda: F.rms_norm(x, (d,), hand.w, 1e-6))
            except Exception as e:
                t_fused = float('nan')
                print('    fused 失败:', str(e)[:50], flush=True)
        print('  %-22s %10.4f %10.4f %10.4f'
              % (str(shape), t_hand * 1e3, t_mul * 1e3, t_fused * 1e3), flush=True)

print()
print('--- 正确性核对 ---', flush=True)
d = 512
hand = HandRMS(d); mulv = MulRMS(d)
x = torch.randn(4, 32, d)
a = hand(x); b = mulv(x)
print('  手写 vs x*x 版 最大差: %.2e' % float((a - b).abs().max()), flush=True)
if hasattr(F, 'rms_norm'):
    c = F.rms_norm(x, (d,), hand.w, 1e-6)
    print('  手写 vs Fused  最大差: %.2e' % float((a - c).abs().max()), flush=True)
print()
print('DONE', flush=True)
