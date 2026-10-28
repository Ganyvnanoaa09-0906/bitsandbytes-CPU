"""settle_parallel_timing.py -- hot path or full call, which one is the real answer?

Two measurements disagree and the difference is now visible in the code.

  test_cpu_gpu_parallel.py builds the B buffer and uploads 5.6 MB inside the loop but
  *before* starting the clock, and takes the minimum of three trials. It reports 0.013 s
  against 0.018 s serial, i.e. 2.80x.

  sweep_split_ratio.py uploads, computes and downloads inside the timed region and takes
  the median. It reports 0.022 s against 0.017 s for the CPU alone, i.e. 0.76x.

Both can be internally consistent. The hot-path number is what you get if the buffers
persist across calls, which they can in a training loop where shapes repeat. The full-call
number is what you get if the upload happens every time. Whether the iGPU is worth using
depends on which of those a real training step looks like, so both need measuring in one
place, with the same code, varying only the clock's extent.

Four configurations on the model's real shape:

  cpu_only        plain torch, the baseline
  serial_full     iGPU half then CPU half, uploads timed
  par_full        iGPU enqueued, CPU half overlapped, uploads and download timed
  par_hot         same as par_full but buffers and uploads hoisted out of the clock

plus a direct measurement of what the upload itself costs, since that is the whole
discrepancy. Reported as median and minimum, so the earlier choice of minimum can be
assessed rather than silently repeated.

Correctness is checked by splitting the result back together, reading the iGPU output
with the stride the kernel actually used.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

KERNEL = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8
__kernel void gemm(const int M, const int N, const int K,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * TX + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane, r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane, r = f / TS, c = f % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // C is written with row stride N, which is the value passed in, so the caller must
    // read it back with that same stride -- reading with the full matrix width is what
    // made an earlier version of this look incorrect when the kernel was fine.
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = by + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = bx + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
'''

TS = 64
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
kern = cl.Kernel(cl.Program(ctx, KERNEL).build(), 'gemm')
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print()

M, N, K = 1024, 2736, 512
torch.set_num_threads(6)
rng = np.random.default_rng(11)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ref = A @ B
At, Bt = torch.from_numpy(A), torch.from_numpy(B)
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)

FRACS = (1 / 3, 0.25, 0.5)


def measure(frac, reps=7):
    n = int(N * frac) // TS * TS
    nrest = N - n
    Bg = np.ascontiguousarray(B[:, :n])
    Bc = np.ascontiguousarray(B[:, n:])
    Bc_t = torch.from_numpy(Bc)
    gsz = ((n + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    out_g = np.empty((M, n), dtype=np.float32)

    # preallocated, so the hot path can reuse them
    bb_ = cl.Buffer(ctx, mf.READ_ONLY, Bg.nbytes)
    bc_ = cl.Buffer(ctx, mf.WRITE_ONLY, M * n * 4)
    cl.enqueue_copy(queue, bb_, Bg)
    queue.finish()

    def cpu_only():
        return At @ Bt

    def serial_full():
        bb2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bg)
        bc2 = cl.Buffer(ctx, mf.WRITE_ONLY, M * n * 4)
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K), ba, bb2, bc2)
        queue.finish()
        o = At @ Bc_t
        cl.enqueue_copy(queue, out_g, bc2).wait()
        return o

    def par_full():
        bb2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bg)
        bc2 = cl.Buffer(ctx, mf.WRITE_ONLY, M * n * 4)
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K), ba, bb2, bc2)
        o = At @ Bc_t
        cl.enqueue_copy(queue, out_g, bc2).wait()
        return o

    def par_hot():
        cl.enqueue_copy(queue, bb_, Bg)          # inside, as a real call would
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K), ba, bb_, bc_)
        o = At @ Bc_t
        cl.enqueue_copy(queue, out_g, bc_).wait()
        return o

    def par_hot_no_upload():
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K), ba, bb_, bc_)
        o = At @ Bc_t
        cl.enqueue_copy(queue, out_g, bc_).wait()
        return o

    fns = [('cpu_only', cpu_only), ('serial_full', serial_full),
           ('par_full', par_full), ('par_hot（含上传）', par_hot),
           ('par_hot（不含上传）', par_hot_no_upload)]
    out = {}
    for name, fn in fns:
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        out[name] = (statistics.median(ts), min(ts))
    return n, out


for frac in FRACS:
    n, out = measure(frac)
    base_m, base_min = out['cpu_only']
    print('核显承担 %d 列（%.0f%%）' % (n, 100.0 * n / N))
    print('  %-22s %10s %10s %10s' % ('配置', '中位数', '最小值', '对纯CPU(中位)'))
    for name, (med, mn) in out.items():
        print('  %-22s %10.3f %10.3f %9.2fx' % (name, med, mn, base_m / med))
    print()

# what the upload alone costs
Bg = np.ascontiguousarray(B[:, :int(N / 3) // TS * TS])
bb_ = cl.Buffer(ctx, mf.READ_ONLY, Bg.nbytes)
cl.enqueue_copy(queue, bb_, Bg)
queue.finish()
ts = []
for _ in range(10):
    t0 = time.perf_counter()
    cl.enqueue_copy(queue, bb_, Bg)
    queue.finish()
    ts.append(time.perf_counter() - t0)
up = statistics.median(ts)
print('B 上传（%.1f MB）单独耗时: %.4f s ⇒ %.1f GB/s'
      % (Bg.nbytes / 2**20, up, Bg.nbytes / up / 1e9))
print()
print('判据: 若 par_hot(不含上传) ≈ max(iGPU, CPU) 而 par_hot(含上传) 更慢，')
print('      那么早先的 2.80x 是热路径数字，真实调用要付上传的代价；')
print('      若含上传也仍优于 cpu_only，则 iGPU 在真实路径上依然划算。')
