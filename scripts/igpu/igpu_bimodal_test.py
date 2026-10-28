"""Is the iGPU's concurrent throughput bimodal? Same configuration, repeated.

Two runs of the blocking sweep disagreed by a factor of 2.9 on identical configurations: the iGPU under
CPU load came out at 213, 203 and 220 GFLOPS in one run and 74, 74 and 75 in the other, with the
"best" configuration flipping from MxN 256x256 to M 512. The difference between runs is larger than the
effect being measured, so neither run supports a conclusion.

The pattern -- about 200 or about 75, nothing between -- suggests a bimodal state rather than noise. If
so, it is the same problem the earliest iGPU measurements had: a quantity that varies severalfold with
conditions that are not being controlled, measured as if it were stable.

This repeats one configuration several times inside a single process and reports the spread. One
configuration, one process, identical code path: any variation is state, not method.

    MxN 256x256 tiling, five CPU threads, iGPU GEMM 2048x2736x512
    eight independent concurrent measurements, warmup discarded

Criterion, fixed in advance: if the spread across the eight is under 15%, the earlier disagreement came
from between-process state and a single careful run would settle it. If the spread approaches the
2.9x seen between runs, the quantity is bimodal, and no sweep of it can be trusted until the state that
selects the mode is identified -- which would be the same class of error the report has already
recorded twice.
"""
from __future__ import annotations
import os, sys
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

exec(open(r'_os.path.join(_root, 'scripts\\igpu\\igpu_blocking_optimum.py')', encoding='utf-8').read()
     .split('assert_idle()')[0])

assert_idle()
stamp('核显并发吞吐是不是双峰的')
print()

once, ws = make_cpu_job(256, 256, 5)
print('  配置: MxN 256x256 分块（工作集 %.1f MB）· CPU 5 线程' % ws)
print()


def one_shot(label):
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
    for _ in range(6):
        tg = time.perf_counter()
        gpu_call()
        gts.append(time.perf_counter() - tg)
    wall = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=5)
    time.sleep(2.0)
    gt = statistics.median(gts)
    return (count[0] * CPU_FLOP / wall / 1e9, GPU_FLOP / gt / 1e9, gt * 1e3)


# idle reference at the start
idle = gpu_steady()
print('  核显单独基准: %.3f ms ⇒ %.1f GFLOPS' % (idle * 1e3, GPU_FLOP / idle / 1e9))
print()
print('  %-8s %12s %14s %14s' % ('第几次', 'CPU 并发', '核显 ms', '核显 GFLOPS'))
print('  ' + '-' * 52)
shots = []
for i in range(8):
    c, g, gt = one_shot(i)
    shots.append((c, g))
    print('  %-8d %12.1f %14.3f %14.1f' % (i + 1, c, gt, g))

print()
gs = [s[1] for s in shots]
cs = [s[0] for s in shots]
print('  核显并发: 中位 %.1f · 最小 %.1f · 最大 %.1f · 【离散 %.1f%%】'
      % (statistics.median(gs), min(gs), max(gs),
         statistics.pstdev(gs) / statistics.mean(gs) * 100))
print('  CPU 并发: 中位 %.1f · 最小 %.1f · 最大 %.1f'
      % (statistics.median(cs), min(cs), max(cs)))
print()
print('  判据（事先固定）: 八次离散 <15%% ⇒ 是进程间状态;接近 2.9× ⇒ 双峰,不能用')
sp = statistics.pstdev(gs) / statistics.mean(gs) * 100
if sp < 15:
    print('  ⇒ 【稳定】✓ 离散 %.1f%% —— 那么两次跑的分歧来自进程间状态' % sp)
elif max(gs) / max(min(gs), 1e-9) > 2.0:
    print('  ⇒ ★【双峰】✗ 最大/最小 = %.1f× —— 同样的代码,同一个进程,两个模式'
          % (max(gs) / max(min(gs), 1e-9)))
    print('  ⇒ ★【所以任何对"核显并发吞吐"的扫描都不可信,直到找出【选择模式的那个状态】】')
else:
    print('  ⇒ 离散 %.1f%%,介于两者之间' % sp)
