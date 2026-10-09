"""confirm_tk_bug.py -- test the diagnosis with one #define.

locate_gemm_error found rows 0..15 correct and 16..63 wrong, with 75% of the output
bad and the first column entirely right. The kernel declares

    __local float As[TK][TS + 4];     // TK = 16, TS = 64

so each K-iteration stages only 16 rows of A, while a TS=64 tile needs 64. The rows
that happen to be staged come out right; the rest are garbage. That is a real bug in
the project's kernel, not a calling convention problem -- and it means every
throughput number this project has for the iGPU (245.3, 266.58, the "+30-35%" claim)
was measured on a kernel that computes about a quarter of the matrix correctly,
because those benchmarks never read their output back.

If that is the cause, then setting TS equal to TK should make it compute correctly,
since the staged block would then cover the whole tile. That is a one-line change to
the kernel source, which is exactly the kind of test that settles a diagnosis
instead of arguing about it.
"""
from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

_lines = open(r'D:\work\bitsandbytes-CPU\bench_opencl3.py', encoding='utf-8').read().split(chr(10))
BASE_SRC = chr(10).join(_lines[36:192])

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


def check(src, M, N, K, TS, gmul, local_size):
    prg = cl.Program(ctx, src).build()
    kern = cl.Kernel(prg, 'sgemm_v2a')
    A = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 7)
    B = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 5)
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B))
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * gmul, (M + TS - 1) // TS * gmul)
    kern(queue, gsz, (gmul, gmul), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    queue.finish()
    out = np.zeros((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    bad = int((np.abs(out - ref) > 1e-3).sum())
    del ba, bb, bc
    return bad, M * N


print('基线（TS=64, TK=16 原样）:')
bad, tot = check(BASE_SRC, 64, 64, 64, 64, 8, (8, 8))
print('  错误 %d / %d' % (bad, tot))
print()

# TS=16 与 TK 匹配 ⇒ 每轮装 16 行正好覆盖 16 行的 tile
src16 = BASE_SRC.replace('#define TS 64', '#define TS 16').replace('#define TM 8', '#define TM 4')
src16 = src16.replace('#define TN 8', '#define TN 4')
print('TS=16, TM=TN=4（与 TK=16 匹配）:')
try:
    bad, tot = check(src16, 64, 64, 64, 16, 4, (4, 4))
    print('  错误 %d / %d  ⇒ %s' % (bad, tot, '✓ 全对' if bad == 0 else '仍有错'))
except Exception as e:
    print('  失败: %s' % str(e)[:120])
print()

# 另一个假设：把 As 的第二维反过来声明为 [TS][TK+4]
print('假设检验：把 LDS 声明改成 As[TS][TK+4] 是否可行（只看能否编译）')
src_alt = BASE_SRC.replace('__local float As[TK][TS + 4];', '__local float As[TS][TK + 4];')
src_alt = src_alt.replace('__local float Bs[TK][TS + 4];', '__local float Bs[TS][TK + 4];')
try:
    cl.Program(ctx, src_alt).build()
    print('  编译通过 ✓ ⇒ 布局可以这样改（但索引也要跟着改，不是一行的事）')
except Exception as e:
    print('  编译失败: %s' % str(e)[:100])
