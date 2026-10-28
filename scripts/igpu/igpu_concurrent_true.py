"""Measure the CPU's ACTUAL concurrent throughput, not its throughput with the GPU idle.

The previous run computed "combined = cpu_alone_tiled + gpu_concurrent", where cpu_alone_tiled was
measured with the GPU doing nothing. That assumes the CPU keeps its tiled rate while the GPU runs, and
it was never checked -- even though the run counted the CPU's iterations during the concurrent phase
and then ignored the count.

That assumption is exactly where the answer lives. If the CPU holds 189 GFLOPS while the iGPU delivers
213, the combined figure is 402 and concurrency is worth 2.07x over the CPU alone, which would be the
first genuinely new optimisation found in some time. If instead the CPU collapses to 60 while the iGPU
runs, the combined figure is 273 and concurrency loses to the CPU alone.

Also worth fixing: the earlier criterion compared against the iGPU alone, but the iGPU alone is not what
training currently uses. Training uses the CPU alone, so that is the number to beat.

Method: for each tiling, run the CPU job in a thread and the iGPU GEMM in the main thread, and count
CPU iterations across the whole concurrent window. Divide the work done by the elapsed time for the
CPU's true concurrent rate, and add the iGPU's measured rate. Both terms are then measured under the
same conditions.

Criterion, fixed in advance and corrected for the use case: combined must exceed the CPU alone
(194.7 GFLOPS measured on the same shape at five threads) by more than the measurement resolution,
which on this machine is about 20%. Anything less is not a win.
"""
from __future__ import annotations
import os, sys
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
import threading
import time

import numpy
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl
from scripts.lib.measure_guard import assert_idle, stamp

exec(open(r'D:\work\bitsandbytes-CPU\igpu_blocking_optimum.py', encoding='utf-8').read()
     .split('assert_idle()')[0])          # reuse the kernel and setup verbatim

assert_idle()
stamp('并发时 CPU 的【实际】吞吐：合计到底是多少')
print()


def cpu_only_rate(once, reps=3):
    for _ in range(2):
        once()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        ts.append(time.perf_counter() - t0)
    return CPU_FLOP / statistics.median(ts) / 1e9, statistics.median(ts)


def concurrent(once, windows=4):
    """CPU in a thread, iGPU in the main thread; report both rates over the same window."""
    stop = [False]
    count = [0]

    def worker():
        while not stop[0]:
            once()
            count[0] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(1.5)
    count[0] = 0
    t0 = time.perf_counter()
    gts = []
    for _ in range(windows):
        tg = time.perf_counter()
        gpu_call()
        gts.append(time.perf_counter() - tg)
    wall = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=5)
    time.sleep(1.5)
    cpu_rate = count[0] * CPU_FLOP / wall / 1e9
    gpu_rate = GPU_FLOP / statistics.median(gts) / 1e9
    return cpu_rate, gpu_rate


idle_gpu = gpu_steady()
gpu_alone = GPU_FLOP / idle_gpu / 1e9
print('  核显单独: %.1f GFLOPS' % gpu_alone)

once_full, _ = make_cpu_job(None, None, 5)
cpu_alone, _ = cpu_only_rate(once_full)
print('  CPU 单独（全量,5 线程）: %.1f GFLOPS' % cpu_alone)
print()
print('  %-16s %9s %11s %11s %11s %11s' %
      ('分块', '工作集', 'CPU 单独', 'CPU 并发', '核显 并发', '合计并发'))
print('  ' + '-' * 78)

configs = [
    ('全量', None, None, 32.0),
    ('M 512', 512, N, 12.2),
    ('MxN 512x512', 512, 512, 2.6),
    ('MxN 256x256', 256, 256, 1.3),
    ('MxN 128x128', 128, 128, 0.59),
]
rows = []
for name, tm, tn, ws in configs:
    once, _ = make_cpu_job(tm, tn, 5)
    co, _ = cpu_only_rate(once)
    cc, gc = concurrent(once)
    rows.append((name, ws, co, cc, gc, cc + gc))
    print('  %-16s %7.1f MB %11.1f %11.1f %11.1f %11.1f'
          % (name, ws, co, cc, gc, cc + gc))

print()
best = max(rows, key=lambda r: r[5])
print('  最好: %s ⇒ 合计 【%.1f GFLOPS】（并发时 CPU %.1f + 核显 %.1f）'
      % (best[0], best[5], best[3], best[4]))
print()
print('  判据（事先固定,基线改成【CPU 单独】%.1f,因为训练现在用的是它）:' % cpu_alone)
if best[5] > cpu_alone * 1.20:
    print('  ⇒ ★【成立】合计 %.1f 对 CPU 单独 %.1f ⇒ 【%.2f×】✓'
          % (best[5], cpu_alone, best[5] / cpu_alone))
    print('  ⇒ ★【这是今天第一条【真的新】的加速路径】')
else:
    print('  ⇒ ✗【不成立】合计 %.1f 对 CPU 单独 %.1f ⇒ 仅 %.2f×（低于本机 ~20%% 的分辨力）'
          % (best[5], cpu_alone, best[5] / cpu_alone))
    print('  ⇒ 结论与 7.62 一致，但理由从"功耗墙"改成【共享缓存预算】')
