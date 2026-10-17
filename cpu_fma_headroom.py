"""cpu_fma_headroom.py -- how much of the CPU's FMA capacity is actually reached?

The numbers so far, none of which answer this directly:

    torch sgemm peak measured here        255 GFLOPS
    what training actually sustains       198.5 GFLOPS
    iGPU best (7.22)                      300 GFLOPS

255 is 78% of the training figure's implied ceiling and nobody has established what the
ceiling is. If the CPU could be fed properly and it is substantially above 255, that is
worth far more than the iGPU, because the CPU needs no separate memory path -- and section
7.25 just established that the iGPU cannot help alongside it.

The theoretical figure depends on microarchitecture details I should not assert from
memory: whether a Zen2 core issues two 256-bit FMAs per cycle or four 128-bit ones, and
what the sustained all-core clock is under AVX2 load. Both are measurable.

  * Clock: read from the OS while the machine is under load, not at idle.
  * FMA throughput: matmuls small enough to stay in cache, swept over size, so the result
    reflects arithmetic rather than memory. Squeezing the working set into L1/L2 removes
    the memory system from the question entirely.
  * The ratio of the two gives how close the machine gets, and the shape of the size sweep
    shows whether the limit is arithmetic or data supply.

Reported per size: GFLOPS, implied FLOPs per cycle at the measured clock, and the fraction
of a candidate peak. Both candidate models are shown rather than picking one in advance.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import threading
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

torch.set_num_threads(6)

print('=== 实测全核频率（加载时读取，不是空闲值）===')


def read_clock():
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "(Get-CimInstance Win32_Processor).CurrentClockSpeed"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().split()[0])
    except Exception:
        return None


idle = read_clock()
print('  空闲时: %s MHz' % idle)

# load all six cores, then read the clock while they are busy
a = torch.randn(1024, 1024)
stop = threading.Event()
res = {}


def spin():
    while not stop.is_set():
        a @ a


ths = [threading.Thread(target=spin, daemon=True) for _ in range(4)]
for t in ths:
    t.start()
time.sleep(1.5)
busy = [read_clock() for _ in range(3)]
stop.set()
for t in ths:
    t.join(timeout=2)
busy = [b for b in busy if b]
bw = statistics.median(busy) if busy else None
print('  加载时: %s MHz  %s' % (bw, '（未降频）' if bw and idle and bw >= idle * 0.95
                               else '（有降频）'))
print()

print('=== cache 内 matmul：算术吞吐，不含内存系统 ===')
print()
print('  %-14s %10s %10s %12s' % ('尺寸', '毫秒', 'GFLOPS', 'FLOP/cycle/核'))

FREQ = (bw or idle or 2800) / 1000.0     # GHz


def bench(n, reps=None):
    x = torch.randn(n, n)
    y = torch.randn(n, n)
    z = torch.empty(n, n)
    torch.mm(x, y, out=z)
    if reps is None:
        reps = max(20, int(6e8 / (n ** 3)))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            torch.mm(x, y, out=z)
        ts.append((time.perf_counter() - t0) / reps)
    dt = statistics.median(ts)
    gf = 2.0 * n ** 3 / dt / 1e9
    per_core_cycle = gf * 1e9 / (FREQ * 1e9 * 6) / 2.0   # FMA per cycle per core
    return dt * 1e3, gf, per_core_cycle


sizes = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
best = (0, 0)
for n in sizes:
    ms, gf, fpc = bench(n)
    if gf > best[1]:
        best = (n, gf)
    kb = 3 * n * n * 4 / 1024
    print('  %-14s %10.3f %10.1f %12.2f   （工作集 %.0f KB）'
          % ('%dx%d' % (n, n), ms, gf, fpc, kb))
print()
print('  峰值出现在 n=%d，%.1f GFLOPS' % best)

print()
print('=== 与几种候选峰值模型对比（频率 %.2f GHz，6 核）===' % FREQ)
for label, fma_per_cycle in (('2x256bit FMA/周期/核', 2.0),
                             ('1x256bit FMA/周期/核', 1.0),
                             ('4x128bit FMA/周期/核', 2.0)):
    flop = fma_per_cycle * 8 * 2 * FREQ * 6      # 8 floats per 256-bit op
    if '128' in label:
        flop = fma_per_cycle * 4 * 2 * FREQ * 6
    print('  %-24s %7.0f GFLOPS   实测/模型 %.2f' % (label, flop, best[1] / flop))
print()
print('=== 与训练实际值对比 ===')
print('  训练中实测（7.22）: 198.5 GFLOPS ⇒ 为本次峰值的 %.2f' % (198.5 / best[1]))
print('  核显最佳（7.22）  : 300.0 GFLOPS ⇒ 为本次峰值的 %.2f' % (300.0 / best[1]))
