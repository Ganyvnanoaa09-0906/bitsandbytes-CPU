"""sweep_ar_gemm_config.py -- occupancy sweep on the project's own GEMM kernel.

Target: make the iGPU carry part of the AR training so 22 hours becomes 9, at
unchanged quality. CLBlast is out -- it only works on cubic shapes here, and even
its own sample shape (m=128, n=64, k=512) fails with -1011 kInsufficientMemoryA, so
that is a defect in CLBlast or the driver, not a calling mistake. But CLBlast is
still the useful reference: it reaches 493.8 GFLOPS at 2048^3, 43% of the part's
1150 GFLOPS peak, while the project's own kernel reaches 266 (23%).

The kernels here take M, N and K as separate arguments with bounds checks, so they
already handle non-cubic shapes; the gap is efficiency, not capability. The most
likely cause is occupancy. The two shipped configurations hard-code local work
sizes of 8x8 (64 threads, 2 warps per workgroup) and 16x16 (256 threads, 8 warps),
and with only 6 CUs an LDS-heavy kernel needs many warps resident per CU to hide
latency. Nobody appears to have swept this.

So sweep the local workgroup size and the tile size, and -- unlike every earlier
benchmark in this project -- require the result to be correct, since a fast wrong
kernel is exactly what wasted the last several hours.
"""
from __future__ import annotations

import itertools
import re
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

# pull the two kernels straight out of bench_opencl3.py
src_text = open(r'D:\work\bitsandbytes-CPU\bench_opencl3.py', encoding='utf-8').read()
m = re.search(r'SRC\s*=\s*r?"""(.*?)"""', src_text, re.S)
if not m:
    m = re.search(r"SRC\s*=\s*r?'''(.*?)'''", src_text, re.S)
if not m:
    print('  ✗ 提取不到 kernel 源码')
    raise SystemExit(1)
KERNEL_SRC = m.group(1)
print('  kernel 源码 %d 字符，含 %d 个 __kernel'
      % (len(KERNEL_SRC), KERNEL_SRC.count('__kernel')))

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL_SRC).build()
print('  编译通过 | 设备 %s | %d CU | 本地内存 %d KB'
      % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print()

# 模型的真实形状（M=token 数）
SHAPES = [(1024, 512, 512), (1024, 2736, 512), (1024, 512, 2736), (1024, 1024, 512)]


def bench(kern_name, TS, gmul, M, N, K, iters=4):
    kern = cl.Kernel(prg, kern_name)
    A = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 61) - 30
    B = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 53) - 26
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * gmul, (M + TS - 1) // TS * gmul)
    args = (np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)

    def once():
        kern(queue, gsz, (gmul, gmul), *args)
        queue.finish()

    once()                                   # 预热
    out = np.zeros((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    d = float(np.abs(out - ref).max()) / (float(np.abs(ref).max()) or 1.0)

    t0 = time.perf_counter()
    for _ in range(iters):
        kern(queue, gsz, (gmul, gmul), *args)
    queue.finish()
    dt = (time.perf_counter() - t0) / iters
    del ba, bb, bc
    return 2.0 * M * N * K / dt / 1e9, d


print('%-8s %-6s %-6s %10s %10s %12s %s'
      % ('kernel', 'TS', '本地', 'GFLOPS', '相对CPU', '正确性', '形状'))
print('-' * 84)
CPU = 198.5
best = None
for kern_name, TS, gmul in itertools.product(('sgemm_v2a', 'sgemm_v2b'),
                                             (32, 64, 128), (4, 8, 16, 32)):
    if gmul * gmul > 1024:
        continue
    for (M, N, K) in SHAPES[:1]:          # 先用一个形状定配置
        try:
            gf, d = bench(kern_name, TS, gmul, M, N, K)
        except Exception as e:
            print('%-8s %-6d %-6d %10s %10s %12s %s'
                  % (kern_name, TS, gmul, 'FAIL', '-', str(e)[:26], '%dx%dx%d' % (M, N, K)))
            continue
        ok = '✓' if d < 1e-4 else '✗ %.2g' % d
        print('%-8s %-6d %-6d %10.1f %9.2fx %12s %s'
              % (kern_name, TS, gmul, gf, gf / CPU, ok, '%dx%dx%d' % (M, N, K)))
        if d < 1e-4 and (best is None or gf > best[0]):
            best = (gf, kern_name, TS, gmul)
print()
if best:
    print('  ⇒ 最佳可用配置: %s TS=%d 本地=%dx%d ⇒ %.1f GFLOPS（CPU 的 %.2fx）'
          % (best[1], best[2], best[3], best[3], best[0], best[0] / CPU))
    print('     目标 500 GFLOPS（CPU 的 2.5x，CLBlast 的水平）')
else:
    print('  ⇒ 没有正确配置')
