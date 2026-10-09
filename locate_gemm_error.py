"""locate_gemm_error.py -- which elements does sgemm_v2a get wrong?

raw_gemm_check showed the first row of sgemm_v2a matching the expected product
exactly, on a cubic shape and on two non-cubic ones, while the overall maximum
difference is 306. So the kernel is fundamentally right -- the convention, the
strides and the tiles are all correct -- and something specific is wrong elsewhere
in the output.

Find the pattern: print the first differing (row, col), how many elements differ,
and whether the errors cluster by row, by column, by tile boundary, or by K-block.
For a single-tile 64^3 case there is no tile boundary to blame, so whatever shows
up there is the root cause and everything else may be a consequence.

This is the last diagnostic step before fixing the kernel, and it is deliberately
raw output rather than a summary statistic, because summaries have misled every
verification attempt tonight.
"""
from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

_lines = open(r'D:\work\bitsandbytes-CPU\bench_opencl3.py', encoding='utf-8').read().split(chr(10))
KERNEL_SRC = chr(10).join(_lines[36:192])

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL_SRC).build()

np.set_printoptions(precision=1, suppress=True, linewidth=250)

M = N = K = 64
A = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7)
B = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5)
ref = A @ B

kern = cl.Kernel(prg, 'sgemm_v2a')
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B))
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
gsz = (N // 64 * 8, M // 64 * 8)
kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
queue.finish()
out = np.zeros((M, N), dtype=np.float32)
cl.enqueue_copy(queue, out, bc).wait()

diff = np.abs(out - ref)
bad = diff > 1e-3
print('M=N=K=%d, sgemm_v2a' % M)
print('  错误元素 %d / %d （%.1f%%）' % (bad.sum(), bad.size, 100.0 * bad.sum() / bad.size))
if bad.any():
    rows = np.where(bad.any(axis=1))[0]
    cols = np.where(bad.any(axis=0))[0]
    print('  出错的行: %d 个，范围 [%d, %d]' % (len(rows), rows.min(), rows.max()))
    print('  出错的列: %d 个，范围 [%d, %d]' % (len(cols), cols.min(), cols.max()))
    print('  按行统计（前 20 行）: %s' % bad.sum(axis=1)[:20])
    print('  按列统计（前 20 列）: %s' % bad.sum(axis=0)[:20])
    r, c = np.argwhere(bad)[0]
    print()
    print('  第一个错误: C[%d][%d]  实际 %.4g  期望 %.4g' % (r, c, out[r, c], ref[r, c]))
    print('    该行实际 C[%d,:8] = %s' % (r, out[r, :8]))
    print('    该行期望        = %s' % ref[r, :8])
    print('    该列实际 C[:8,%d] = %s' % (c, out[:8, c]))
    print('    该列期望        = %s' % ref[:8, c])
    # 每行的最大差，看是否与行号有关
    rowmax = diff.max(axis=1)
    print()
    print('  每行最大差（前 24 行）: %s' % np.round(rowmax[:24], 1))
    print('  每行最大差（后 8 行） : %s' % np.round(rowmax[-8:], 1))
else:
    print('  ✓ 全部正确')
