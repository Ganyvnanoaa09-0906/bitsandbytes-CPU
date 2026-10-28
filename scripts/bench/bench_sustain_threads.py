# -*- coding: utf-8 -*-
"""bench_sustain_threads.py — 持续负载下的最优线程数（不是爆发下的）。

背景：
  bench_avx2.py 用短时爆发测得「6 线程 255 > 5 线程 212 GFLOPS」（+20%）
  但 bench_r5.py 测出持续 90 秒会降频 19%（131.7 → 106.9）
  ⇒ 两个结论都只在各自条件下成立，训练是【持续负载】，必须用持续条件重测

方法：每个线程数连续跑 60 秒 sgemm，只统计【后 30 秒】的稳态吞吐。
线程之间留 20 秒冷却，避免前一组的热量影响后一组。

这是直接可执行的结论：改训练脚本的 --threads 就行。
"""
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
import time
import torch

print('=' * 70)
print('持续负载下的最优线程数（只统计后 30 秒稳态）')
print('=' * 70)
print('  对照: 短时爆发 6线程 255、5线程 212 GFLOPS')
print('        持续 90 秒单线程配置会降频 19%')
print()

n = 1024
A = torch.randn(n, n)
B = torch.randn(n, n)

results = {}
for nth in (4, 5, 6):
    torch.set_num_threads(nth)
    A @ B                                       # 预热
    print('  测 %d 线程...' % nth, flush=True)

    # 阶段 1：0-30 秒，先让它热起来（这段数据丢弃）
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30.0:
        A @ B

    # 阶段 2：后 30 秒，统计稳态
    reps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < 30.0:
        A @ B
        reps += 1
    el = time.perf_counter() - t1
    gf = reps * 2 * n ** 3 / el / 1e9
    results[nth] = gf
    print('    %d 线程 稳态 %8.2f GFLOPS' % (nth, gf), flush=True)

    # 阶段 3：冷却
    time.sleep(20)

print()
print('  %-10s %14s' % ('线程数', '稳态 GFLOPS'))
print('  ' + '-' * 27)
for k in sorted(results):
    print('  %-10d %14.2f' % (k, results[k]))
best = max(results, key=results.get)
print()
print('  => 持续负载下最优: %d 线程 (%.2f GFLOPS)' % (best, results[best]))
if 6 in results and 5 in results:
    print('     6 vs 5 线程: %+.1f%%' % (100 * (results[6] - results[5]) / results[5]))
if 6 in results and 4 in results:
    print('     6 vs 4 线程: %+.1f%%' % (100 * (results[6] - results[4]) / results[4]))
print()
print('DONE')
