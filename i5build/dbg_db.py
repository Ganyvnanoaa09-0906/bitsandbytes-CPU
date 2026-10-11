"""定位双缓冲版的 NaN：小形状 + 打印第一处不符。"""
import os
import sys

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igpu_double_buffer import K32, K32DB  # noqa: E402

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([dev])
q = cl.CommandQueue(ctx)
mf = cl.mem_flags

for M, N, K in ((64, 64, 64), (64, 64, 128), (128, 64, 64)):
    a = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7 + 1) / 7.0
    b = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5 + 1) / 5.0
    ref = a @ b
    print(f'--- M={M} N={N} K={K}  (k-tiles={K//8}) ---')
    for name, src in (('g32', K32), ('g32db', K32DB)):
        prog, _ = __import__('igpu_double_buffer').build(ctx, name, src)
        k = cl.Kernel(prog, name)
        ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
              cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
        cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        k(q, gws, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cb)
        q.finish()
        o = np.empty((M, N), np.float32)
        cl.enqueue_copy(q, o, cb).wait()
        nan = int(np.isnan(o).sum())
        err = float(np.nanmax(np.abs(o - ref)))
        rel = err / (np.abs(ref).max() + 1e-9)
        idx = np.unravel_index(int(np.nanargmax(np.abs(o - ref))), o.shape)
        print(f'   {name:<7} NaN={nan:>5}  最大绝对误差={err:.3e}  相对={rel:.2e}  '
              f'最差位置={idx}  值={o[idx]:.4f} 应为={ref[idx]:.4f}  '
              f'{"OK" if rel < 1e-4 else "不匹配"}')
