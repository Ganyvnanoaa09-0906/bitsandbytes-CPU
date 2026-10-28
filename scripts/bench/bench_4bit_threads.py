# -*- coding: utf-8 -*-
"""bench_4bit_threads.py — 查清"前向只用一个核"到底出在哪。

现象：SDXL 4-bit 前向时 CPU 累计 ≈ 墙钟 × 0.86 ⇒ 基本单线程。
嫌疑：
  A) 我们 CPU fork 的 4-bit GEMV 内核是单线程的（那大矩阵也不会随线程数加速）
  B) fp16 的 conv / matmul 在 CPU torch 上走慢路径（torch 的 fp16 CPU 支持历来很弱）
  C) torch.set_num_threads 没生效

做法：同一个大矩阵乘法，分别用 fp32 Linear / fp16 Linear / Linear4bit，
      各在 1 线程与 6 线程下各测一次，看加速比。
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

M, K, N = 256, 4096, 4096


def bench(fn, iters=3):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


print('  bnb', bnb.__version__, '| torch', torch.__version__)
print('  %-14s %10s %10s %8s' % ('层类型', '1线程(s)', '6线程(s)', '加速比'))

torch.manual_seed(0)
x32 = torch.randn(M, K)
x16 = x32.half()

lin32 = torch.nn.Linear(K, N)
lin16 = torch.nn.Linear(K, N).half()

q = Linear4bit(K, N, bias=True, compute_dtype=torch.float32,
               quant_type='nf4', compress_statistics=True)
with torch.no_grad():
    q.weight = bnb.nn.Params4bit(lin32.weight.data.clone(), requires_grad=False,
                                 quant_type='nf4', compress_statistics=True)
    q.weight = q.weight.to('cpu')
    q.bias.data.copy_(lin32.bias.data)

CASES = [
    ('fp32 Linear', lambda: lin32(x32)),
    ('fp16 Linear', lambda: lin16(x16)),
    ('4bit Linear', lambda: q(x32)),
]

for th in (1, 6):
    torch.set_num_threads(th)
    res = {}
    for name, fn in CASES:
        with torch.no_grad():
            try:
                res[name] = bench(fn)
            except Exception as e:
                res[name] = float('nan')
                if th == 1:
                    print('  %-14s 失败: %s' % (name, str(e)[:60]))
    if th == 1:
        r1 = res
    else:
        for name, _ in CASES:
            a, b = r1.get(name, float('nan')), res.get(name, float('nan'))
            sp = (a / b) if (b and b > 0) else float('nan')
            print('  %-14s %10.3f %10.3f %8.2fx' % (name, a, b, sp))

print()
print('  判读：')
print('   · 若 fp32/fp16 加速比 ≈5~6 而 4bit ≈1 ⇒ 【我们的 GEMV 内核是单线程的】')
print('   · 若 fp16 在 1 线程就远慢于 fp32 ⇒ torch CPU 的 fp16 走慢路径（SDXL 用 fp16 是错的选择）')
print('   · 若 4bit 反而是三者最快 ⇒ 换 fp32 conv + 4bit linear 会更快')
