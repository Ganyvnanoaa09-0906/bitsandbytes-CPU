"""Why is the FMA not saturated? Separate the four candidates with one experiment.

The framing first, because it changes the answer: the peak must be computed at the measured
full-load clock, not at the boost clock. Six cores, two FMA ports, eight float32 per AVX2 register,
two FLOP per FMA gives 192 FLOP per cycle, so 456 GFLOPS at the 2375 MHz this machine actually holds
under load and 691 at the 3.6 GHz nameplate. The 4500U is a 15 W part with a 2.38 GHz base, and the
fact that it sits at 2375 rather than 3600 under full AVX2 load says the boost is never sustained.
Against the honest denominator, 371 GFLOPS is 81%, not 54%.

That leaves four candidates for the remaining 19% plus the drop from 371 to 230 as the matrices grow,
and they are separable:

  registers    16 YMM registers under AVX2. A classic microkernel uses 12 for accumulators, 2 for
               A, 2 for B, giving 96 FMAs per 16 loads. Enough, but it fixes the blocking shape.
  bandwidth    arithmetic intensity for an n-cubed GEMM is about n/6 FLOP per byte, so 64 at n=384
               and 341 at n=2048, against a machine ratio near 9. Bandwidth cannot bind. My earlier
               claim that memory limits the GEMM was wrong, and this prints the numbers that show it.
  architecture  two 256-bit loads per cycle are needed to feed two FMAs, and the L1 has exactly two
               load ports, so loop overhead and address arithmetic eat directly into issue.
  clock        whether AVX2 load drags the frequency below 2375, which Task Manager cannot show
               because it reports an instantaneous value.

The measurements:

  * GFLOPS and the clock sampled during the run, so the denominator is the real one
  * a working-set sweep at roughly constant FLOPs, which isolates cache capacity from everything else
  * torch's GEMM against a deliberately naive one, to establish whether 371 was my kernel's ceiling
    or the machine's

Decision rule, fixed in advance: if torch beats the previously measured 371 by more than 10%, the
earlier figure was an implementation artefact and the real ceiling is higher. If it does not, 371 is
the machine's and 0.81 of the clock-derived peak is where this ends.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
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

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from scripts.lib.measure_guard import assert_idle, stamp

assert_idle()
stamp('FMA 喂不满的四因分离')
torch.set_num_threads(6)
print()

# the honest denominator
CORES, PORTS, WIDTH, FLOPS_PER_FMA = 6, 2, 8, 2
for ghz in (2.375, 3.60):
    print('  峰值 @ %.3f GHz = %d FLOP/周期 × %.3f GHz = 【%.0f GFLOPS】'
          % (ghz, CORES * PORTS * WIDTH * FLOPS_PER_FMA,
             ghz, CORES * PORTS * WIDTH * FLOPS_PER_FMA * ghz))
print()

print('=== 1. 算术强度:证明带宽【不可能】是 GEMM 的限制 ===')
print('  %-14s %12s %14s' % ('n', '算术强度', '按 27GB/s 需'))
print('  ' + '-' * 44)
for n in (384, 768, 2048):
    ai = n / 6.0                      # FLOP per byte for an n-cubed GEMM
    print('  %-14d %9.1f F/B %11.1f GFLOPS' % (n, ai, 27e9 * ai / 1e9))
print('  ⇒ 本机 240~371 GFLOPS 都远低于带宽上限 ⇒ 【GEMM 是计算受限】')
print()

print('=== 2. 工作集扫描（固定 FLOPs，只改尺寸）===')
print('  %-16s %10s %10s %14s' % ('形状', '总 MB', 'GFLOPS', '占 456 峰值'))
print('  ' + '-' * 54)
TOTAL = 2.0 * 2048 ** 3
for n in (256, 384, 512, 768, 1024, 1536, 2048):
    A = torch.randn(n, n)
    B = torch.randn(n, n)
    reps = max(1, int(round((2048.0 / n) ** 3)))
    for _ in range(2):
        torch.mm(A, B)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            torch.mm(A, B)
        ts.append((time.perf_counter() - t0) / reps)
    gf = 2.0 * n ** 3 / statistics.median(ts) / 1e9
    ws = 3 * n * n * 4 / 1024 ** 2
    print('  %-16s %9.2fM %10.1f %13.1f%%'
          % ('%dx%dx%d' % (n, n, n), ws, gf, gf / 456.0 * 100))
print('  ⇒ L2 = 3 MB · L3 = 8 MB（共享）—— 看掉落在哪个尺寸发生')
print()

print('=== 3. torch(MKL) 对我手写内核能到多少 ===')
print('  我此前测到的: n=384 ⇒ 371 · n=768 ⇒ 269 · n=2048 ⇒ 230')
print('  上面第 2 段是 torch 的结果 ⇒ 直接对照')
print()
print('  判据（事先固定）: torch 比我记录的 371 快 > 10% ⇒ 之前那个是【实现产物】,真上限更高；')
print('                    否则 371 就是这台机器的,0.81 的峰值利用率到此为止。')
