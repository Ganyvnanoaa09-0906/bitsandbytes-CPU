# -*- coding: utf-8 -*-
"""bench_avx2.py — 本地 R5-4500U 的 GEMM 压榨空间测量（轻量版，防卡死）"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time
import torch
import torch.nn as nn

PEAK = 672.0   # 6核 × 32 FLOP/周期 × 3.5GHz 粗估


def bench(fn, iters=15, warm=3):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def say(*a):
    print(*a, flush=True)


say('=' * 66)
say('torch', torch.__version__)
say('=' * 66)

say('--- 后端能力 ---')
for name, fn in (
    ('mkldnn', lambda: torch.backends.mkldnn.is_available()),
    ('mkl', lambda: torch.backends.mkl.is_available()),
    ('openmp', lambda: torch.backends.openmp.is_available()),
):
    try:
        say('  %-8s %s' % (name, fn()))
    except Exception as e:
        say('  %-8s 查询失败 %s' % (name, str(e)[:60]))

say()
say('--- sgemm GFLOPS 线程扫描 (n=1024, 理论峰值约 %.0f) ---' % PEAK)
A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)
best = (0, None)
for nth in (1, 2, 3, 4, 5, 6):
    torch.set_num_threads(nth)
    dt = bench(lambda: A @ B, iters=20, warm=4)
    gf = 2 * 1024 ** 3 / dt / 1e9
    if gf > best[0]:
        best = (gf, nth)
    say('  threads=%d   %8.2f GFLOP/s   %5.1f%% 峰值' % (nth, gf, 100 * gf / PEAK))
say('  => 最佳 %d 线程 %.2f GFLOP/s' % (best[1], best[0]))

say()
say('--- 不同尺寸 ---')
torch.set_num_threads(best[1])
for n in (256, 512, 1024, 2048):
    a = torch.randn(n, n); b = torch.randn(n, n)
    dt = bench(lambda: a @ b, iters=12, warm=3)
    say('  n=%-5d %8.2f GFLOP/s  (%.2f ms)' % (n, 2 * n ** 3 / dt / 1e9, dt * 1e3))

say()
say('--- mkldnn 开关 (n=1024) ---')
a = torch.randn(1024, 1024); b = torch.randn(1024, 1024)
for flag in (True, False):
    torch.backends.mkldnn.enabled = flag
    dt = bench(lambda: a @ b, iters=20, warm=4)
    say('  mkldnn=%-5s %8.2f GFLOP/s' % (flag, 2 * 1024 ** 3 / dt / 1e9))
torch.backends.mkldnn.enabled = True

say()
say('--- 真实训练形状的 Linear ---')
for (bs, din, dout) in ((8, 512, 2048), (8, 512, 512), (64, 256, 1024), (1, 512, 4096), (16, 512, 512)):
    lin = nn.Linear(din, dout)
    x = torch.randn(bs, din)
    dt = bench(lambda: lin(x), iters=40, warm=5)
    fl = 2 * bs * din * dout
    say('  b%-3d %4d->%4d  %8.2f GFLOP/s   %7.3f ms' % (bs, din, dout, fl / dt / 1e9, dt * 1e3))

say()
say('--- 精度对比 (n=1024) ---')
for nm, cast in (('fp32', torch.float32), ('bf16', torch.bfloat16), ('fp16', torch.float16)):
    try:
        a = torch.randn(1024, 1024).to(cast)
        b = torch.randn(1024, 1024).to(cast)
        t = bench(lambda: a @ b, iters=12, warm=3)
        say('  %-5s %8.2f GFLOP/s' % (nm, 2 * 1024 ** 3 / t / 1e9))
    except Exception as e:
        say('  %-5s 失败 %s' % (nm, str(e)[:70]))

say()
say('DONE')
