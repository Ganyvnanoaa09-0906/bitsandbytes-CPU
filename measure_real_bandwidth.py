"""measure_real_bandwidth.py -- what is this machine's actual memory bandwidth?

Two claims are in play and only one of them can be right.

  A. 13.2 GB/s measured with a six-thread torch.mul is the bandwidth ceiling, so the
     memory system is saturated and that is why everything is stuck near 80% efficiency.
  B. That 13.2 is an elementwise-kernel issue rate, not a bandwidth limit, and the real
     ceiling is the DDR4-2667 dual-channel figure near 42.7 GB/s theoretical.

The distinction matters because it changes what can be done. If the ceiling is 13 GB/s
there is nothing to win and the only lever is doing less work. If it is near 40, a step
needing 0.8-1.5 GB at 12.84 s (62-117 GB/s) is impossible anyway -- which points the same
way but for a different reason, and matters for how close short-sequence training can get.

WMI reports two 8 GB Samsung M471A1G44AB0-CTD at 2667 MHz, 128 bits total, i.e. dual
channel DDR4-2667: 2667e6 * 8 * 2 = 42.7 GB/s theoretical.

Measured properly this time: a pure read (reduction), a pure write (fill), and a copy,
each over arrays far larger than L3 (8 MB), so the numbers describe the memory system
rather than any kernel's issue rate. torch.sum is a reduction with no elementwise output,
which is the closest thing to a pure read available without writing custom AVX2.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time

import torch

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)

N = 256 * 1024 * 1024          # 256M floats = 1 GB
x = torch.randn(N)
y = torch.empty_like(x)
GB = N * 4 / 1e9

print('数组 %.0f MB（L3 只有 8 MB ⇒ 数据必然来自内存 ✓）' % (N * 4 / 2**20))
print()


def timeit(fn, reps=5, bytes_moved=GB):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    dt = statistics.median(ts)
    return dt, bytes_moved / dt


print('%-28s %10s %14s' % ('操作', '秒', 'GB/s'))
print('-' * 56)

for th in (1, 2, 4, 6):
    torch.set_num_threads(th)
    dt, bw = timeit(lambda: torch.sum(x), bytes_moved=GB)
    print('%-28s %10.3f %14.1f' % ('纯读 torch.sum（%d 线程）' % th, dt, bw))

torch.set_num_threads(6)
dt, bw = timeit(lambda: y.fill_(1.0), bytes_moved=GB)
print('%-28s %10.3f %14.1f' % ('纯写 fill_（6 线程）', dt, bw))

dt, bw = timeit(lambda: y.copy_(x), bytes_moved=2 * GB)
print('%-28s %10.3f %14.1f' % ('拷贝 copy_（读+写）', dt, bw))

dt, bw = timeit(lambda: torch.mul(x, 1.0001, out=y), bytes_moved=2 * GB)
print('%-28s %10.3f %14.1f' % ('逐元素 mul（读+写）', dt, bw))

print()
print('=== 与物理上限对比 ===')
THEO = 2667e6 * 8 * 2 / 1e9
print('  DDR4-2667 双通道理论          %6.1f GB/s' % THEO)
print('  实际可达通常为理论的 65~80%%   %6.1f ~ %.1f GB/s' % (THEO * 0.65, THEO * 0.8))
print()
print('=== 训练一步的需求 ===')
for label, gb, sec in (('实测一步（batch 8, T=1024）', 1.2, 12.84),
                       ('乐观下限', 0.8, 12.84)):
    print('  %-28s %.1f GB / %.2f s = %6.1f GB/s' % (label, gb, sec, gb / sec))
print()
print('判据: 若纯读达到理论值的 65% 以上 ⇒ 13.2 是内核发射问题，不是带宽上限；')
print('      若训练一步的需求超过实测上限 ⇒ 无论哪种，减少流量都是唯一出路。')
