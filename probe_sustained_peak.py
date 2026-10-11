# -*- coding: utf-8 -*-
"""probe_sustained_peak.py —— 这台机器 fp32 GEMM 的【真实持续峰值】到底是多少。

为什么必须单独量：整步换算给出 GEMM 跑到 242 GFLOPS，而我手上记着的
「持续峰值 233 / 突发 351 GFLOPS」是早期测的。242 > 233，说明旧数字低估了。
这个数直接决定结论方向：
  · 若真实持续峰值 ≈ 250  → GEMM 已到顶，加速预训练这条路是死的
  · 若真实持续峰值 ≈ 350  → GEMM 还有 40% 余量，值得继续调内核

方法：大形状跑够 20 秒以上（避开突发窗口），取稳态吞吐；同时按秒采样
% Processor Performance 换算出实际运行频率，避免"其实是降频了"的误判。
"""
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import forge_gemm as FG

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
FG.set_threads(TH)
SECONDS = float(os.environ.get('SECONDS', '20'))

print('=' * 90)
print('持续峰值测量   线程=%d  目标时长 %.0f 秒' % (TH, SECONDS))
print('=' * 90)

I = J = R = 4096
A = torch.randn(I, R)
B = torch.randn(J, R)
fl = 2.0 * I * J * R / 1e9
print('  形状 %d x %d x %d  = %.1f GFLOP/次' % (I, J, R, fl))
print()

for tag, fn in (('forge(auto=0)', lambda: FG.mm(A, B, 'NT')),
                ('torch/oneDNN', lambda: torch.mm(A, B.t()))):
    for _ in range(2):
        fn()
    ts = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < SECONDS:
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_start
    ts.sort()
    n = len(ts)
    print('  %-14s  %d 次 / %.1f 秒' % (tag, n, total))
    print('    最好  %8.1f ms  %7.1f GFLOPS' % (ts[0] * 1e3, fl / ts[0]))
    print('    中位  %8.1f ms  %7.1f GFLOPS   <-- 稳态' % (ts[n // 2] * 1e3, fl / ts[n // 2]))
    print('    最差  %8.1f ms  %7.1f GFLOPS' % (ts[-1] * 1e3, fl / ts[-1]))
    print('    整段平均 %7.1f GFLOPS (= 总FLOPs/总时长，最保守)' % (fl * n / total))
    print()

print('DONE')
