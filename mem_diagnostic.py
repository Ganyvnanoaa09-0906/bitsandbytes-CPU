"""Memory subsystem diagnostic: why does dual-channel DDR4-2667 deliver 23.7 of 42.7 GB/s?

WMI confirms both DIMMs report 2667 MT/s with ConfiguredClockSpeed also 2667, so nothing is
downclocked and the 42.7 GB/s figure is arithmetically right (2667 MT/s x 8 B x 2 channels).
The gap is real and unexplained.

The one hard clue so far is the shape of the scaling: 1 thread 13 GB/s, 4 threads 23.7, and
6 threads no better. A plateau at 4 threads rules out "not enough cores" and points at
something shared saturating.

What this measures, chosen so the results discriminate between explanations rather than just
producing more numbers:

  * scaling across 1/2/3/4/6 threads -- a plateau identifies where saturation begins
  * read vs write vs copy -- if copy costs far more than read, the shared resource is the
    bus turnaround rather than raw line rate
  * a mixed read/write kernel -- if it is slower than pure read, the controller is
    switching, which is a real efficiency loss on DDR4
  * access stride -- large strides should show whether row-buffer locality matters here
  * pointer-chase latency -- a high latency with flat bandwidth is the signature of a
    throughput-limited controller rather than one that cannot keep enough requests in flight

Not measured: single-channel operation. It cannot be selected from software, so the
comparison would need a BIOS change or pulling a DIMM, and neither is appropriate to do
unattended.

Written to run after the training finishes; measure_guard refuses while it is going.
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
stamp('内存子系统诊断')
print()

N_GB = 0.5                                  # per-array size; 3 arrays for copy
n = int(N_GB * 1024 ** 3 // 4)
print('每个缓冲区 %.1f GB（%.1f 亿个 float32）' % (N_GB, n / 1e8))
print()

try:
    import torch
except Exception:
    torch = None


def bench(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print('=== 1. 线程数扩展（纯读 23.7 GB/s 是哪个数？）===')
print('%-10s %12s %12s' % ('线程', 'GB/s', '每线程 GB/s'))
print('-' * 38)
if torch is not None:
    for nt in (1, 2, 3, 4, 6):
        torch.set_num_threads(nt)
        a = torch.ones(n)
        s = torch.zeros(())
        r = bench(lambda: s.copy_(a.sum()))
        gb = N_GB / r
        print('%-10d %12.2f %12.2f' % (nt, gb, gb / nt))
        del a, s
    torch.set_num_threads(6)
print()
print('判据: 若 4 与 6 线程几乎相同 ⇒ 共享资源在 4 线程处饱和；')
print('      若每线程带宽持续下降 ⇒ 无饱和，是请求并发不足。')
print()

print('=== 2. 读 / 写 / 拷贝 / 混合 ===')
print('%-16s %12s %s' % ('模式', 'GB/s', '说明'))
print('-' * 52)
a = np.ones(n, dtype=np.float32)
b = np.zeros(n, dtype=np.float32)
c = np.zeros(n, dtype=np.float32)


def read_only():
    a.sum()


def write_only():
    b.fill(1.0)


def copyv():
    np.copyto(c, a)


def mixed():
    np.add(a, b, out=c)


for name, fn, note in (('纯读', read_only, 'sum 归约'),
                       ('纯写', write_only, 'fill'),
                       ('拷贝', copyv, '读+写各一份'),
                       ('逐元素混合', mixed, '读 2 写 1')):
    dt = bench(fn)
    moved = N_GB * (2 if name == '拷贝' else (3 if name == '逐元素混合' else 1))
    print('%-16s %12.2f %s' % (name, moved / dt, note))
print()
print('判据: 若拷贝的 GB/s 远低于纯读 ⇒ 总线【方向切换】是主要损失（DDR4 的典型行为）；')
print('      若三者接近 ⇒ 是线速限制。')
print()

print('=== 3. 访问步长（行缓冲局部性）===')
print('%-12s %12s' % ('步长(元素)', 'GB/s'))
print('-' * 26)
for stride in (1, 8, 64, 512):
    idx = np.arange(0, n, stride, dtype=np.int64)
    sub = a[idx] if stride > 1 else a
    dt = bench(lambda: sub.sum())
    gb = (sub.nbytes / dt) / 1e9 if stride == 1 else (sub.nbytes / dt) / 1e9
    print('%-12d %12.2f' % (stride, gb))
print()
print('判据: 步长变大若带宽大跌 ⇒ 行缓冲命中率是关键；若平稳 ⇒ 流式预取有效。')
print()

print('=== 4. 指针追逐延迟 ===')
try:
    m = 1 << 22
    perm = np.random.default_rng(0).permutation(m).astype(np.int64)
    chase = perm.copy()
    t0 = time.perf_counter()
    p = 0
    for _ in range(2_000_000):
        p = int(chase[p])
    dt = time.perf_counter() - t0
    print('  随机访问延迟 ≈ %.1f ns（%d 次追逐 / %.2f s）'
          % (dt / 2_000_000 * 1e9, 2_000_000, dt))
except Exception as e:
    print('  延迟测量失败: %s' % e)
print()
print('读法: 延迟高而带宽平 ⇒ 控制器吞吐受限（能发出的请求数被打满），')
print('      而不是"请求发不出去"。这两者的修法完全不同。')
print()
print('=== 未做的: 单通道对比 ===')
print('  无法从软件选择单通道；需要 BIOS 或拔一条内存，不适合无人值守时做。')
print('  若上面的扩展曲线在 4 线程处平掉，它间接指向控制器而非通道数。')
