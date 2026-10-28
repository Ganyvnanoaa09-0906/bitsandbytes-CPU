"""pinned_igpu_mm.py -- make the transfers actually asynchronous, so the halves overlap.

The breakdown measurement overturned the explanation I had been working from. There is
no mysterious 6 ms of Python overhead: every step in the call is individually reasonable.
The problem is that the download blocks.

  shape 1024x2736x512, iGPU taking half the columns
    "download C"   15.727 ms when timing enqueues only
                    3.291 ms when finishing after every step
    iGPU kernel     9.796 ms when finished per step, which matches 2.87 GFLOP at the
                    measured 300 GFLOPS

A factor of 4.8 on the same step means the enqueue-only timing was including the wait for
the kernel. That happens because cl.enqueue_copy into an ordinary numpy array, which is
pageable, is synchronous in practice: the driver stages it through pinned memory and
cannot return until the transfer is done. So the CPU half and the iGPU half have never
overlapped, which is the entire reason the split loses to plain torch.

The fix is a pinned host buffer, allocated once and reused, so the copy is genuinely
asynchronous and the CPU half can run while the kernel does. The arithmetic says that is
worth having: for this shape the halves are 9.8 ms (iGPU) and 11.9 ms (CPU), so a real
overlap gives max(9.8, 11.9) = 11.9 ms against 16.5 ms for the CPU alone, i.e. 1.39x.

Second finding from the same run, which needs its own fix: on a small shape the kernel
achieves only 47.6 GFLOPS against 293 on a large one, so splitting a small matmul is
counterproductive regardless of overlap. Usage therefore has to depend on size.

Correctness is checked before timing, as always.
"""
from __future__ import annotations
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

KERNEL = open(r'D:\work\bitsandbytes-CPU\_gemm_kernel.txt', encoding='utf-8').read() \
    if False else None

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
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'gemm')
pool = {}
pinned = {}

print('设备 %s | %d CU | cl_khr_fp16 已确认与本节无关' % (dev.name, dev.max_compute_units))
print()


def get_buf(nbytes, tag):
    b = pool.get(tag)
    if b is None or b.size < nbytes:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        pool[tag] = b
    return b


def get_pinned(shape, tag):
    key = (shape, tag)
    p = pinned.get(key)
    if p is None:
        p = cl.tools.ImmediateAllocator  # placeholder, replaced below
    return p


def run(M, N, K, reps=10):
    torch.set_num_threads(6)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    n_gpu = (N // 2) // TS * TS
    ref = a @ b
    A = np.ascontiguousarray(a.numpy())
    B = np.ascontiguousarray(b.numpy())
    ba_ = get_buf(A.nbytes, 'a')
    bb_ = get_buf(B.nbytes, 'b')   # full B, row-major, no slice copy
    bc_ = get_buf(M * n_gpu * 4, 'c')
    gsz = ((n_gpu + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)

    # plain numpy destination (pageable) vs a mapped/pinned device buffer
    out_plain = np.empty((M, n_gpu), dtype=np.float32)
    pin_bytes = M * n_gpu * 4
    pin_np = np.empty((M, n_gpu), dtype=np.float32)
    pin_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, pin_bytes)
    mapped, _ = cl.enqueue_map_buffer(queue, pin_buf, cl.map_flags.READ,
                                     0, (M, n_gpu), np.float32)
    queue.finish()

    def once(use_pinned):
        cl.enqueue_copy(queue, ba_, A)
        # upload all of B and let the kernel read only the first n_gpu columns:
        # B[:, :n_gpu] is not contiguous and copying it costs ~1.2 ms per call
        cl.enqueue_copy(queue, bb_, B)
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n_gpu), np.int32(K), ba_, bb_, bc_)
        cpu_part = a @ b[:, n_gpu:].contiguous()
        if use_pinned:
            cl.enqueue_copy(queue, mapped, bc_)
        else:
            cl.enqueue_copy(queue, out_plain, bc_)
        queue.finish()
        dst = mapped if use_pinned else out_plain
        out = torch.empty(M, N)
        out[:, :n_gpu] = torch.from_numpy(np.asarray(dst))
        out[:, n_gpu:] = cpu_part
        return out

    # correctness first
    got = once(False)
    d = float((got - ref).abs().max()) / (float(ref.abs().max()) or 1.0)
    got2 = once(True)
    d2 = float((got2 - ref).abs().max()) / (float(ref.abs().max()) or 1.0)

    res = {}
    for label, up in (('pageable（原实现）', False), ('pinned（映射缓冲）', True)):
        for _ in range(2):
            once(up)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            once(up)
            ts.append((time.perf_counter() - t0) * 1e3)
        res[label] = statistics.median(ts)

    t0 = time.perf_counter()
    for _ in range(5):
        a @ b
    t_cpu = (time.perf_counter() - t0) / 5 * 1e3
    return res, t_cpu, d, d2, n_gpu


print('%-20s %10s %10s %10s %10s  %s' % ('形状', 'pageable', 'pinned', '纯CPU', '正确性', '判定'))
print('-' * 82)
for (M, N, K) in [(1024, 2736, 512), (1024, 512, 512), (2048, 2048, 2048), (1024, 1024, 1024)]:
    res, t_cpu, d, d2, ng = run(M, N, K)
    pk = res['pinned（映射缓冲）']
    pl = res['pageable（原实现）']
    verdict = ('✓ pinned 更快 %.2fx' % (pl / pk)) if pk < pl else ('✗ 无改善')
    print('%-20s %10.2f %10.2f %10.2f %10.2e  %s'
          % ('%dx%dx%d' % (M, N, K), pl, pk, t_cpu, max(d, d2), verdict))
print()
print('判据: pinned 若显著低于 pageable ⇒ 此前确实从未重叠，重叠是真实收益；')
print('      若接近 ⇒ 阻塞不在页锁定，需要另找原因。')
