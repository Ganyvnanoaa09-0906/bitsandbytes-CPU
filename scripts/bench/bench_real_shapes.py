# -*- coding: utf-8 -*-
"""bench_real_shapes.py — 用定稿 kernel 测【真实工作负载】的算子形状

1024³ 只是参照。我们的 AR 模型（b=8, T=256）真实形状是：
  FFN w1:  (2048×512) @ (512×2048)     M=2048 N=2048 K=512
  FFN w2:  (2048×2048) @ (2048×512)    M=2048 N=512  K=2048
  QKV:     (2048×512) @ (512×1024)     M=2048 N=1024 K=512
  o_proj:  (2048×512) @ (512×512)      M=2048 N=512  K=512
  大 batch 训练 (b=32): 各形状 M×4

还要算【对比 CPU】的加速比，以及加权后的整体收益。
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
import time
import numpy as np
import torch
import pyopencl as cl

SRC = open(r'D:\work\bitsandbytes-CPU\gemm_final.py', encoding='utf-8').read()
# 从 gemm_final.py 里抠出 kernel 源码
import re
m = re.search(r"SRC = r'''(.*?)'''", SRC, re.S)
KERNEL = m.group(1)

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
kern = cl.Kernel(cl.Program(ctx, KERNEL).build(), 'sgemm_t')
print('=' * 82)
print('真实工作负载形状：核显 vs CPU')
print('=' * 82)
print('  设备:', dev.name, '|', dev.max_compute_units, 'CU')
print()

SHAPES = [
    ('FFN w1  (gate/up)', 2048, 2048, 512),
    ('FFN w2  (down)',    2048,  512, 2048),
    ('QKV',               2048, 1024, 512),
    ('o_proj',            2048,  512, 512),
]

# 预热核显
print('  预热核显 20 秒...', flush=True)
WA = np.random.randn(512, 512).astype(np.float32)
WB = np.random.randn(512, 512).astype(np.float32)
mf = cl.mem_flags
wab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=WA)
wbb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=WB)
wcb = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 512 * 4)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    for _ in range(4):
        kern(queue, (512 // 64 * 8, 512 // 64 * 8), (8, 8),
             np.int32(512), np.int32(512), np.int32(512), wab, wbb, wcb)
    queue.finish()
del wab, wbb, wcb, WA, WB
print('  完成\n')

print('  %-22s %-22s %10s %10s %8s' % ('算子', '形状', 'iGPU', 'CPU', '加速比'))
print('  ' + '-' * 78)

torch.set_num_threads(6)
tot_gpu = tot_cpu = 0.0
weight = {'FFN w1  (gate/up)': 2, 'FFN w2  (down)': 1, 'QKV': 1, 'o_proj': 1}

for name, M, N, K in SHAPES:
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)

    def one():
        for _ in range(4):
            kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
        queue.finish()

    one()
    best = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        one()
        dt = (time.perf_counter() - t0) / 4.0
        best = max(best, 2.0 * M * N * K / dt / 1e9)

    tA, tB = torch.from_numpy(A), torch.from_numpy(B)
    with torch.no_grad():
        tA @ tB
        t0 = time.perf_counter()
        for _ in range(20):
            tA @ tB
        cdt = (time.perf_counter() - t0) / 20
    cgf = 2.0 * M * N * K / cdt / 1e9

    print('  %-22s %-22s %8.1f   %8.1f   %7.2fx'
          % (name, '%dx%d @ %dx%d' % (M, K, K, N), best, cgf, best / cgf))
    w = weight[name]
    tot_gpu += best * w
    tot_cpu += cgf * w
    del A, B, ab, bb, cb

print()
print('  --- 加权总和（按 AR 层的实际出现次数）---')
print('  iGPU 加权:  %.1f' % tot_gpu)
print('  CPU  加权:  %.1f' % tot_cpu)
print('  ⇒ 整体加速比: 【%.2fx】' % (tot_gpu / tot_cpu))
print()
print('  注：这是纯 GEMM 的加速比。实际收益还要考虑：')
print('      · CPU↔iGPU 数据传输（UMA 下只是缓存一致性，但仍要同步）')
print('      · 非 GEMM 算子（RMSNorm/SiLU/softmax）iGPU 未必更快')
print('      · 【并发会互拖到 0.64x，所以必须时分复用】')
print('DONE')
