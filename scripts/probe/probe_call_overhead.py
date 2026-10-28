# -*- coding: utf-8 -*-
"""probe_call_overhead.py —— 量 forge 每次调用的【固定开销】，并验证两种修法。

线索：端到端换 forge 反而慢 443 ms。而加权 GEMM 总量只快 80 ms。
差额 ~520 ms / 186 次调用 ≈ 2.8 ms/次 —— 这个固定开销必须找出来。

两个嫌疑：
  ① `#pragma omp parallel` 每次都重建线程队（MSVC vcomp 默认让空闲线程睡下去，
     唤醒代价高）→ 试 OMP_WAIT_POLICY=ACTIVE 让工作线程自旋
  ② 每次调用 6 个线程各做一次 _aligned_malloc(256KB) + _aligned_free

量法：
  · 极小的形状（64x64x64，算量 0.5 MFLOP）—— 计算本身 < 10µs，测到的几乎全是固定开销
  · 真实形状重启一遍作为对照
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib import forge_gemm as FG

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
FG.set_threads(TH)
REPS = int(os.environ.get('REPS', '60'))

print('=' * 88)
print('forge 调用固定开销   线程=%d' % TH)
print('  OMP_WAIT_POLICY = %s' % os.environ.get('OMP_WAIT_POLICY', '(未设，默认 PASSIVE)'))
print('  OMP_NUM_THREADS = %s' % os.environ.get('OMP_NUM_THREADS', '(未设)'))
print('=' * 88)
print()

# ---- ① 固定开销：极小形状 ----
print('---- ① 固定开销（形状越小，越接近纯开销） ----')
print('  %-18s %12s %12s %14s' % ('形状', 'forge µs/次', 'torch µs/次', '计算量 MFLOP'))
print('  ' + '-' * 60)
for I, J, R in ((64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)):
    A = torch.randn(I, R)
    Bm = torch.randn(J, R)
    for _ in range(5):
        FG.mm(A, Bm, 'NT')
        torch.mm(A, Bm.t())
    bf = 1e30
    for _ in range(REPS):
        t0 = time.perf_counter(); FG.mm(A, Bm, 'NT'); bf = min(bf, time.perf_counter() - t0)
    bt = 1e30
    for _ in range(REPS):
        t0 = time.perf_counter(); torch.mm(A, Bm.t()); bt = min(bt, time.perf_counter() - t0)
    print('  %-18s %12.1f %12.1f %14.3f'
          % ('%d x %d x %d' % (I, J, R), bf * 1e6, bt * 1e6, 2 * I * J * R / 1e6))

# ---- ② 真实形状 ----
print()
print('---- ② 真实形状（对照 ① 看固定开销占多少） ----')
SH = [('w12 前向 NT', 'NT', 2048, 2736, 512),
      ('w12 gx   NN', 'NN', 2048, 512, 2736),
      ('w12 gW   TN', 'TN', 2736, 512, 2048),
      ('w3  前向 NT', 'NT', 2048, 512, 1368),
      ('方阵 前向 NT', 'NT', 2048, 512, 512),
      ('256 前向 NT', 'NT', 2048, 256, 512)]
print('  %-14s %-4s %-20s %10s %10s %9s %10s' %
      ('形状', '布局', 'I x J x R', 'forge µs', 'torch µs', 'forge GF', 'torch GF'))
print('  ' + '-' * 92)
for nm, lay, I, J, R in SH:
    if lay == 'NT':
        A, Bm = torch.randn(I, R), torch.randn(J, R)
    elif lay == 'NN':
        A, Bm = torch.randn(I, R), torch.randn(R, J)
    else:
        A, Bm = torch.randn(R, I), torch.randn(R, J)
    tr = FG.torch_ref(A, Bm, lay)
    for _ in range(3):
        FG.mm(A, Bm, lay)
        tr = FG.torch_ref(A, Bm, lay)
    bf = 1e30
    for _ in range(20):
        t0 = time.perf_counter(); FG.mm(A, Bm, lay); bf = min(bf, time.perf_counter() - t0)
    bt = 1e30
    for _ in range(20):
        t0 = time.perf_counter(); FG.torch_ref(A, Bm, lay); bt = min(bt, time.perf_counter() - t0)
    fl = 2.0 * I * J * R
    print('  %-14s %-4s %-20s %10.1f %10.1f %9.1f %10.1f'
          % (nm, lay, '%d x %d x %d' % (I, J, R), bf * 1e6, bt * 1e6,
             fl / bf / 1e9, fl / bt / 1e9))
print()
print('DONE')
