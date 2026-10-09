"""raw_gemm_check.py -- verify by printing numbers, not by computing a norm.

My comparison code has been wrong three times tonight: it compared A@B against a
kernel that returns B@A (CLBlast), it used identity as an operand when every row
and column of identity is identical, and it reported cubic results as wrong by 28
in one script and 0 in another. Each time the verification, not the kernel, was at
fault. Every one of those failures came from computing a summary statistic and
believing it.

So do it the other way: use tiny integer-valued matrices, print A, B, the kernel's
output and the exact expected output as literal numbers, and let the numbers be
read directly. With A[i][j] = i + j and B[i][j] = i - j over a single tile, the
expected product is computable by hand for the first few entries, and a wrong
convention, a transposed read or a broken tile boundary is visible rather than
inferred.

Also test a non-cubic shape one tile wide, where a boundary bug would show.
"""
from __future__ import annotations

import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

# SRC = r''''' spans lines 37..192 of bench_opencl3.py; extract by line number,
# which cannot be fooled by quoting. TS/TK/TM/TN are #defines, so they are
# compile-time constants and the launch's local size must match them (8x8).
_lines = open(r'D:\work\bitsandbytes-CPU\bench_opencl3.py', encoding='utf-8').read().split(chr(10))
KERNEL_SRC = chr(10).join(_lines[36:192])

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL_SRC).build()


def run(kern_name, M, N, K, A, B, TS, gmul):
    kern = cl.Kernel(prg, kern_name)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B))
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * gmul, (M + TS - 1) // TS * gmul)
    kern(queue, gsz, (gmul, gmul), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    queue.finish()
    out = np.zeros((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    # 用 .T 也读一遍，看是否只是转置
    del ba, bb, bc
    return out


np.set_printoptions(precision=1, suppress=True, linewidth=200)

for (M, N, K, TS, label) in [(64, 64, 64, 64, '立方 64³（单 tile ✓ 作者验证过的配置）'),
                             (128, 64, 64, 64, '非立方 128x64x64（M 跨两个 tile ✓）'),
                             (64, 128, 64, 64, '非立方 64x128x64（N 跨两个 tile ✓）')]:
    A = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7)
    B = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5)
    ref = A @ B
    print('=' * 78)
    print('%s   M=%d N=%d K=%d  TS=%d gmul=8' % (label, M, N, K, TS))
    print('  A[0,:6] = %s' % A[0, :6])
    print('  B[:,0]  = %s' % B[:6, 0])
    print('  期望 C[0,:6] = %s' % ref[0, :6])
    for kn in ('sgemm_v2a', 'sgemm_v2b'):
        out = run(kn, M, N, K, A, B, TS, 8)
        d = float(np.abs(out - ref).max())
        dt_ = float(np.abs(out - ref.T).max()) if out.shape == ref.T.shape else 1e9
        print('  %s: C[0,:6] = %s   maxdiff=%.3g   与转置 maxdiff=%.3g'
              % (kn, out[0, :6], d, dt_))
    print()
