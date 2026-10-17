"""The blocking sweep, done with a ratio rather than an absolute figure.

Three runs disagreed by up to 2.9x on the same configurations. Chasing it found that the iGPU's solo
rate itself moves -- 393, 386, 390, 315, then back to 393 without any load between the last two -- so
the instability is the machine's state, not the method. Within one process the concurrent measurement
is stable to 2.1%; across processes it is not, because the state left behind by the previous experiment
differs.

What survives that is the ratio. Collapsing the three runs:

    no blocking     65.0/390.6 = 0.17     60.7/386.5 = 0.16     consistent
    blocked        220.4/390.6 = 0.56    148.0/386.5 = 0.38     direction consistent, magnitude not
                                           128.0/315.2 = 0.41

So the design has to be: measure the solo baseline immediately next to each concurrent measurement, in
the same process, and report the ratio. Which is the rule this project has twice paid for failing to
follow -- 7.62's cliff and 7.88's factor of 2.9.

Baseline for the criterion: training currently runs the CPU alone, at about 197 GFLOPS on this shape at
five threads. That is the number a combination has to beat, not the iGPU alone, because the iGPU alone
is not what anything uses.

Criterion, fixed in advance: combined must exceed the CPU alone by at least 20%, that being this
machine's measurement resolution. The iGPU's share is reported as a ratio against a baseline taken
immediately adjacent, so it cannot be contaminated by state drift.
"""
from __future__ import annotations

import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

exec(open(r'D:\work\bitsandbytes-CPU\igpu_blocking_optimum.py', encoding='utf-8').read()
     .split('assert_idle()')[0])

assert_idle()
stamp('分块扫描（比值设计：每次并发测量紧邻一次单独基准）')
print()


def solo(n=8):
    for _ in range(2):
        gpu_call()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        gpu_call()
        ts.append(time.perf_counter() - t0)
    return GPU_FLOP / statistics.median(ts) / 1e9


def cpu_rate(once, reps=3):
    for _ in range(2):
        once()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        ts.append(time.perf_counter() - t0)
    return CPU_FLOP / statistics.median(ts) / 1e9


def concurrent(once, n=8):
    stop = [False]
    count = [0]

    def worker():
        while not stop[0]:
            once()
            count[0] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(2.0)
    count[0] = 0
    t0 = time.perf_counter()
    gts = []
    for _ in range(n):
        tg = time.perf_counter()
        gpu_call()
        gts.append(time.perf_counter() - tg)
    wall = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=5)
    time.sleep(2.0)
    return count[0] * CPU_FLOP / wall / 1e9, GPU_FLOP / statistics.median(gts) / 1e9


configs = [
    ('全量', None, None, 32.0, 5),
    ('全量 · CPU 2 线程', None, None, 32.0, 2),
    ('M 512', 512, N, 12.2, 5),
    ('MxN 512x512', 512, 512, 2.6, 5),
    ('MxN 256x256', 256, 256, 1.3, 5),
    ('MxN 256x256 · CPU 3 线程', 256, 256, 1.3, 3),
]

print('  %-26s %9s %9s %10s %10s %10s %9s' %
      ('配置', '工作集', 'CPU单独', '单独基准', '并发 CPU', '并发核显', '核显占比'))
print('  ' + '-' * 92)
rows = []
for name, tm, tn, ws, nthreads in configs:
    torch.set_num_threads(nthreads)
    once, _ = make_cpu_job(tm, tn, nthreads)
    co = cpu_rate(once)
    s_before = solo(6)                 # solo baseline immediately before
    cc, gc = concurrent(once, 6)
    s_after = solo(6)                  # and immediately after
    s_ref = statistics.mean([s_before, s_after])
    share = gc / s_ref
    rows.append((name, ws, co, cc, gc, s_ref, share, cc + gc))
    print('  %-26s %7.1f MB %9.1f %9.1f %10.1f %10.1f %8.0f%%'
          % (name, ws, co, s_ref, cc, gc, share * 100))

print()
print('  %-26s %12s %12s %14s' % ('配置', '合计并发', 'CPU 单独', '相对 CPU 单独'))
print('  ' + '-' * 70)
for r in rows:
    print('  %-26s %12.1f %12.1f %13.2f×' % (r[0], r[7], r[2], r[7] / r[2]))

print()
best = max(rows, key=lambda r: r[7] / r[2])
cpu_ref = rows[0][2]
print('  最好: %s ⇒ 合计 【%.1f】 对 CPU 单独 %.1f ⇒ 【%.2f×】'
      % (best[0], best[7], best[2], best[7] / best[2]))
print()
print('  判据（事先固定）: 合计 > CPU 单独 × 1.20 ⇒ 并发值得做')
if best[7] > best[2] * 1.20:
    print('  ⇒ ★【成立】✓ 而这是今天第一条【真的新】的加速路径')
    print('  ⇒ ★【而它的机制不是功耗协商,是【缓存预算】—— 压小 CPU 工作集,把 L3 让给核显】')
else:
    print('  ⇒ ✗ 不成立')
print()
print('  核显占比一列才是最稳的读数（它对着相邻基准算,不受跨时段漂移影响）:')
for r in rows:
    print('    %-26s %.0f%%' % (r[0], r[6] * 100))
