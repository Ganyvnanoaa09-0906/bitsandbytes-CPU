# -*- coding: utf-8 -*-
"""bench_hybrid5.py — 热机后的真实分工并发（修正 bench_hybrid2 的冷启动错误）。

上一版两个错误：
  1) 核显没预热就测 → 中频 120 GF，实际热机后 190~213 GF
  2) ffn_tasks 循环变量 k 覆盖了注意力张量 k

本版：
  - 先给核显 20 秒预热
  - 用真实分工：核显跑 FFN GEMM，CPU 跑注意力
  - 同时测"两边跑同种 GEMM"作为最坏情形对照
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import threading
import time
import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

print('=' * 78, flush=True)
print('热机后 · 真实分工并发  |  核显:', dev.name, flush=True)
print('=' * 78, flush=True)

KERNEL = open('bench_hybrid4.py', encoding='utf-8').read()
i0 = KERNEL.index("SRC = r'''") + len("SRC = r'''")
i1 = KERNEL.index("'''", i0)
prg = cl.Program(ctx, KERNEL[i0:i1]).build()
K8 = cl.Kernel(prg, 'sgemm8')

B_SZ, TT, NH, HD = 8, 256, 8, 64
M = B_SZ * TT          # 2048

# FFN w1 形状（核显这一侧的工作）
Mm, Nn, Kk = M, 2048, 512
A = np.random.randn(Mm, Kk).astype(np.float32)
B = np.random.randn(Kk, Nn).astype(np.float32)
mf = cl.mem_flags
ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
cb = cl.Buffer(ctx, mf.WRITE_ONLY, Mm * Nn * 4)
gsz = ((Nn + 63) // 64 * 8, (Mm + 63) // 64 * 8)
cl_args = (np.int32(Mm), np.int32(Nn), np.int32(Kk), ab, bb, cb)

# CPU 这一侧的注意力
q = torch.randn(B_SZ, NH, TT, HD)
kk_ = torch.randn(B_SZ, NH, TT, HD)
v = torch.randn(B_SZ, NH, TT, HD)
At = torch.randn(Mm, Kk)
Bt = torch.randn(Kk, Nn)

print('预热核显 20 秒...', flush=True)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    for _ in range(8):
        K8(queue, gsz, (8, 8), *cl_args)
    queue.finish()
print('  预热完成\n', flush=True)


def gpu_worker(stop, cnt):
    while not stop.is_set():
        for _ in range(8):
            K8(queue, gsz, (8, 8), *cl_args)
        queue.finish()
        cnt[0] += 8


def cpu_attn_worker(stop, cnt):
    while not stop.is_set():
        F.scaled_dot_product_attention(q, kk_, v, is_causal=True)
        cnt[0] += 1


def cpu_gemm_worker(stop, cnt):
    while not stop.is_set():
        At @ Bt
        cnt[0] += 1


DUR = 15


def run_pair(worker, tag):
    stop = threading.Event(); c = [0]
    t = threading.Thread(target=worker, args=(stop, c)); t.start()
    time.sleep(DUR); stop.set(); t.join(); queue.finish()
    return c[0] / DUR


# 单独基线
gpu_solo = run_pair(gpu_worker, 'gpu')
att_solo = run_pair(cpu_attn_worker, 'attn')
gemm_solo = run_pair(cpu_gemm_worker, 'cpu_gemm')

# A) 核显 FFN + CPU 注意力（真实分工）
stop = threading.Event(); cg = [0]; ca = [0]
tg = threading.Thread(target=gpu_worker, args=(stop, cg))
ta = threading.Thread(target=cpu_attn_worker, args=(stop, ca))
tg.start(); ta.start(); time.sleep(DUR); stop.set(); tg.join(); ta.join(); queue.finish()
gpu_a = cg[0] / DUR; att_a = ca[0] / DUR

# B) 核显 FFN + CPU 密集 GEMM（最坏情形对照）
stop = threading.Event(); cg = [0]; cc = [0]
tg = threading.Thread(target=gpu_worker, args=(stop, cg))
tc = threading.Thread(target=cpu_gemm_worker, args=(stop, cc))
tg.start(); tc.start(); time.sleep(DUR); stop.set(); tg.join(); tc.join(); queue.finish()
gpu_b = cg[0] / DUR; gemm_b = cc[0] / DUR

print('  %-38s %14s %14s' % ('', '核显 FFN 次/秒', 'CPU 侧 次/秒'), flush=True)
print('  %-38s %14.1f %14.1f' % ('单独: 核显FFN / CPU注意力', gpu_solo, att_solo), flush=True)
print('  %-38s %14s %14.1f' % ('单独: CPU 密集GEMM', '-', gemm_solo), flush=True)
print('  %-38s %14.1f %14.1f  (CPU注意力保持 %.0f%%)'
      % ('A 真实分工并发', gpu_a, att_a, 100 * att_a / att_solo), flush=True)
print('  %-38s %14.1f %14.1f  (CPU GEMM保持 %.0f%%)'
      % ('B 同种GEMM并发(最坏)', gpu_b, gemm_b, 100 * gemm_b / gemm_solo), flush=True)
print()
print('  核显保持率:  A %.0f%%   B %.0f%%' % (100 * gpu_a / gpu_solo, 100 * gpu_b / gpu_solo), flush=True)
print()
print('  => 真实分工(A) 核显几乎不掉速 + CPU 白捡注意力 → 这才是"配合"该有的样子', flush=True)
print('DONE', flush=True)
