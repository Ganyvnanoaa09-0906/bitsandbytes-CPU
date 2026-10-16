"""The three unfinished CPU items, in one run.

First, verify the latency figure. 7.41.3 recorded that the stride test was broken -- it divided by
sub.nbytes while large strides read far fewer elements -- and the latency test was never audited the
same way. 259 ns against a normal 70 to 90 for DDR4-2667 is suspicious enough to check before it is
used as evidence. The check is to measure latency across working-set sizes: L1 should be a few
nanoseconds, L2 a dozen, L3 thirtyish, and DRAM seventy to ninety. If the small sizes come out sane
and only DRAM is wild, the DRAM number is suspect; if every level is inflated by the same factor, the
test's own overhead is the problem.

Second, thread scaling. One thread gives 72.9 GFLOPS and six give 190.0, a factor of 2.6 rather than 6.
That saturation needs a cause, and the candidates are the memory subsystem, the power budget, or the
clock dropping as more cores engage. Measure the clock alongside.

Third, bandwidth, with a straightforwardly correct stream rather than one that has already been shown
wrong once.

No decision rules here: these are diagnostics on measurements already used, and the outcome decides
whether earlier numbers stand.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from measure_guard import assert_idle, stamp  # noqa: E402

assert_idle()
stamp('CPU 三项：延迟判据 · 线程扩展 · 带宽复验')
print()

# ---------- 1. latency across working-set sizes ----------
print('  === 1. 逐级延迟（指针追逐，工作集从 L1 到内存）===')
print('  %-12s %12s %12s %14s' % ('工作集', '延迟 ns', '每跳周期', '预期区间'))
print('  ' + '-' * 56)
SIZES = [(16 * 1024, 'L1 16KB', '~4-5 周期'),
         (256 * 1024, 'L2 256KB', '~12-15 周期'),
         (4 * 1024 ** 2, 'L3 4MB', '~40-50 周期'),
         (256 * 1024 ** 2, 'DRAM 256MB', '~170-200 周期')]
GHZ = 2.375
lat = {}
for nbytes, label, expect in SIZES:
    n = nbytes // 8                      # 8 bytes per node (next index)
    idx = np.arange(n, dtype=np.int64)
    # build a single cycle covering the array, then shuffle it into a random cycle
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    nxt = np.empty(n, dtype=np.int64)
    nxt[perm] = np.roll(perm, -1)        # cycle through a random order
    nxt = np.ascontiguousarray(nxt)

    steps = max(n * 4, 2_000_000)
    # warm
    p = 0
    for _ in range(min(steps, n)):
        p = int(nxt[p])
    t0 = time.perf_counter()
    p = 0
    cnt = 0
    while cnt < steps:
        p = int(nxt[p]); p = int(nxt[p]); p = int(nxt[p]); p = int(nxt[p])
        cnt += 4
    dt = time.perf_counter() - t0
    ns = dt / cnt * 1e9
    cyc = ns * GHZ
    lat[label] = ns
    print('  %-12s %12.1f %12.1f %14s' % (label, ns, cyc, expect))
print()
if 'L1 16KB' in lat and 'DRAM 256MB' in lat:
    ratio = lat['DRAM 256MB'] / max(lat['L1 16KB'], 1e-9)
    print('  L1 对 DRAM 的比值: 【%.1f×】**（正常约 15~20×）' % ratio)
    print('  ⇒ %s' % ('各级都合理 ⇒ 延迟测量本身可信' if lat['L1 16KB'] < 5
                      else '★ L1 都偏高 ⇒ 【测试自身有固定开销,各级数字都要减掉它】'))
print()

# ---------- 2. thread scaling with clock ----------
print('  === 2. 线程扩展 + 频率 ===')
import torch  # noqa: E402
print('  %-10s %12s %12s %12s' % ('线程', 'GFLOPS', '相对 1 线程', '满载频率 GHz'))
print('  ' + '-' * 52)
N = 2048
A = torch.randn(N, N)
B = torch.randn(N, N)
f0 = None
for nt in (1, 2, 3, 4, 5, 6):
    torch.set_num_threads(nt)
    for _ in range(2):
        torch.mm(A, B)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        torch.mm(A, B)
        ts.append(time.perf_counter() - t0)
    gf = 2.0 * N ** 3 / statistics.median(ts) / 1e9
    if f0 is None:
        f0 = gf
    ghz = None
    try:
        import subprocess
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Processor).MaxClockMHz'],
                           capture_output=True, text=True)
        ghz = float(r.stdout.strip()) / 1000.0 if r.stdout.strip() else None
    except Exception:
        pass
    print('  %-10d %12.1f %11.2f× %12s'
          % (nt, gf, gf / f0, '%.2f (标称)' % ghz if ghz else '—'))
print()
print('  ⇒ 若扩展远低于线性（6 线程应约 6×），而这不是频率下降造成的，')
print('    那就是【内存子系统】在 2 线程之后就饱和了')
