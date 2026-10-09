"""throughput_of_correct_kernel.py -- what does a working version actually deliver?

Established: the project's sgemm_v2a stages only TK=16 rows of A into local memory
while a TS=64 tile needs 64, so 75% of the output is garbage. Setting TS=16 to match
TK drops the error count from 3072/4096 to 18/4096, which confirms the diagnosis.

Which means every iGPU figure this project has -- 245.3, 266.58 GFLOPS, and the
"+30-35% for GEMM-dense seq<=256" claim built on them -- was measured on a kernel
computing a quarter of the matrix correctly. Those benchmarks never read their
output back.

So measure the throughput of the version that is nearly correct, at the shapes the
model actually issues (M = 1024 tokens, N and K from 512 to 2736), and compare
against the CPU's 198.5 GFLOPS. Report correctness alongside, always.

If a correct configuration cannot beat the CPU, the iGPU cannot carry the training
and the 9-hour target has to come from elsewhere. That is a useful answer either
way, and it is the answer rather than another number that might be meaningless.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

_lines = open(r'D:\work\bitsandbytes-CPU\bench_opencl3.py', encoding='utf-8').read().split(chr(10))
BASE = chr(10).join(_lines[36:192])

# TS 与 TK 必须匹配：每轮装入的行数要覆盖整个 tile
SRC16 = (BASE.replace('#define TS 64', '#define TS 16')
             .replace('#define TM 8', '#define TM 4')
             .replace('#define TN 8', '#define TN 4'))

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC16).build()
CPU = 198.5

print('设备 %s | %d CU | 本地内存 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print('内核 TS=16 TM=TN=4（与 TK=16 匹配，错误率 18/4096）')
print()
print('%-22s %10s %12s %10s %s' % ('形状 MxNxK', 'GFLOPS', '相对 CPU', '错误', '结论'))
print('-' * 76)

SHAPES = [(1024, 512, 512), (1024, 1024, 512), (1024, 2736, 512),
          (1024, 512, 2736), (2048, 2048, 2048)]
for (M, N, K) in SHAPES:
    A = (np.arange(M * K, dtype=np.float32).reshape(M, K) % 61) - 30
    B = (np.arange(K * N, dtype=np.float32).reshape(K, N) % 53) - 26
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B))
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    TS, gmul = 16, 4
    gsz = ((N + TS - 1) // TS * gmul, (M + TS - 1) // TS * gmul)

    def once():
        cl.Kernel(prg, 'sgemm_v2a')(queue, gsz, (gmul, gmul),
                                    np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    try:
        once()
        out = np.zeros((M, N), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * (np.abs(ref).max() or 1)).sum())
        t0 = time.perf_counter()
        for _ in range(3):
            once()
        dt = (time.perf_counter() - t0) / 3
        gf = 2.0 * M * N * K / dt / 1e9
        verdict = '核显更快 ✓' if gf > CPU else 'CPU 更快 ✗'
        print('%-22s %10.1f %11.2fx %10d %s'
              % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, bad, verdict))
    except Exception as e:
        print('%-22s %10s %12s %10s %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-', str(e)[:30]))
    del ba, bb, bc
print()
print('判据: 要 9 小时内完成训练，核显需要达到 CPU 的 2.5 倍左右（约 500 GFLOPS），')
print('      因为只有这样才能让 CPU 与核显并行后把 22 小时压到 9 小时。')
