"""diagnose_contention.py -- isolate what makes concurrent CPU+iGPU slower.

The previous test showed 0.81x against serial and a bandwidth collapse to 0.15x, but it
had a flaw: the CPU side was single-threaded numpy, so it was only pulling 4.1 GB/s to
begin with and the comparison was not the real one. The real case is six CPU threads
against the iGPU.

Three layers, each removing one candidate explanation.

  Layer 1 -- bandwidth, measured fairly. Both sides streaming, both with their natural
    parallelism. If the combined figure is near the larger of the two, the bus is a hard
    shared limit. If it exceeds either alone but falls short of their sum, capacity is
    partly shared. If it is far below even the larger, something other than raw bus
    capacity is at work -- arbitration, latency, or queueing.

  Layer 2 -- attribution. Time each side separately while they run together, using
    perf_counter around the CPU half and around the iGPU half. If both slow down the
    interference is mutual; if only one does, it is one-sided resource capture.

  Layer 3 -- compute only, no memory. The decisive one: a kernel that does arithmetic on
    registers and touches no memory, run against a CPU workload that also fits in cache.
    If they still interfere, the cause is not the memory bus at all but the scheduler,
    the driver's CPU threads, or clock/power contention -- and the fix is entirely
    different. If they stop interfering, the bus is confirmed.

Only after layer 3 is there any basis for saying which.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

STREAM = r'''
__kernel void stream(const int n, const __global float* restrict X,
                     __global float* restrict Y) {
    const int gid = get_global_id(0);
    if (gid < n) Y[gid] = X[gid] * 1.0001f;
}
'''

COMPUTE = r'''
#define CHAINS 8
#define ITERS  8192
__kernel void burn(__global float* out) {
    const int gid = get_global_id(0);
    float a[CHAINS], b[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) { a[i] = (float)(gid + i) * 1e-3f; b[i] = 0.999f; }
    for (int it = 0; it < ITERS; it++) {
        #pragma unroll
        for (int i = 0; i < CHAINS; i++) {
            a[i] = fma(a[i], b[i], 0.5f);
            b[i] = fma(b[i], a[i], 0.25f);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < CHAINS; i++) s += a[i] + b[i];
    out[gid] = s;
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
k_stream = cl.Kernel(cl.Program(ctx, STREAM).build(), 'stream')
k_burn = cl.Kernel(cl.Program(ctx, COMPUTE).build(), 'burn')
print('设备 %s | %d CU | 内存 %.1f GB' % (dev.name, dev.max_compute_units,
                                          dev.global_mem_size / 2**30))
print()


def timeit(fn, reps=5, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---------------------------------------------------------------- layer 1: bandwidth
NB = 16 * 1024 * 1024
X = np.ascontiguousarray(np.random.rand(NB).astype(np.float32))
Y = np.empty_like(X)
bx_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
by_ = cl.Buffer(ctx, mf.WRITE_ONLY, X.nbytes)
GB = 2 * X.nbytes / 1e9           # read + write


def gpu_stream():
    k_stream(queue, (NB,), None, np.int32(NB), bx_, by_)
    queue.finish()


t = timeit(gpu_stream, reps=3)
bw_gpu = GB / t
print('=== 第一层：带宽（公平对比）===')
print('  仅有核显流式拷贝            %6.1f GB/s' % bw_gpu)

# multithreaded CPU streaming: the natural parallelism, not single-threaded numpy
import torch  # noqa: E402
torch.set_num_threads(6)
Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)


def cpu_stream():
    torch.mul(Xt, 1.0001, out=Yt)


t = timeit(cpu_stream, reps=3)
bw_cpu = GB / t
print('  仅 CPU 流式拷贝（6 线程）   %6.1f GB/s' % bw_cpu)


def both_stream():
    k_stream(queue, (NB,), None, np.int32(NB), bx_, by_)
    torch.mul(Xt, 1.0001, out=Yt)
    queue.finish()


t = timeit(both_stream, reps=3)
bw_both = GB / t
print('  两者同时（合计量）          %6.1f GB/s' % bw_both)
print('  ⇒ 单侧最大 %.1f，两侧之和 %.1f' % (max(bw_gpu, bw_cpu), bw_gpu + bw_cpu))
if bw_both > max(bw_gpu, bw_cpu) * 1.15:
    print('  ⇒ 合计【超过】单侧最大 ⇒ 带宽【部分独立】')
elif bw_both < max(bw_gpu, bw_cpu) * 0.85:
    print('  ⇒ 合计【低于】单侧最大 ⇒ 不只是共享，是【互相损害】')
else:
    print('  ⇒ 合计≈单侧最大 ⇒ 带宽是【硬共享上限】')
print()

# ------------------------------------------------------------- layer 2: attribution
shape = (1024, 2736, 512)
M, N, K = shape
rng = np.random.default_rng(1)
A1 = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B1 = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
A2 = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B2 = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ba_, bb_ = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A1),
            cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B1))
bc_ = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
kern = cl.Kernel(cl.Program(ctx, KERNEL_SRC := open(
    r'D:\work\bitsandbytes-CPU\_k.cl', 'w', encoding='utf-8').name and r'''
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
''').build(), 'gemm')

At2, Bt2 = torch.from_numpy(A2), torch.from_numpy(B2)
C2 = torch.empty(M, N)


def cpu_gemm():
    torch.mm(At2, Bt2, out=C2)


def gpu_gemm():
    kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba_, bb_, bc_)
    queue.finish()


t_cpu = timeit(cpu_gemm)
t_gpu = timeit(gpu_gemm)
print('=== 第二层：并发时各自用了多久 ===')
print('  单独: CPU %.4f s   iGPU %.4f s' % (t_cpu, t_gpu))

cpu_times, gpu_times, total = [], [], []
for _ in range(5):
    t0 = time.perf_counter()
    kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba_, bb_, bc_)
    t1 = time.perf_counter()
    torch.mm(At2, Bt2, out=C2)
    t2 = time.perf_counter()
    queue.finish()
    t3 = time.perf_counter()
    gpu_times.append(t3 - t0)
    cpu_times.append(t2 - t1)
    total.append(t3 - t0)
print('  并发: CPU 侧 %.4f s（%.2fx 单独）  iGPU 侧 %.4f s（%.2fx 单独）  合计 %.4f s'
      % (statistics.median(cpu_times), statistics.median(cpu_times) / t_cpu,
         statistics.median(gpu_times), statistics.median(gpu_times) / t_gpu,
         statistics.median(total)))
print('  ⇒ 两侧都变慢 ⇒ 相互干扰；只有一侧 ⇒ 单向占用')
print()

# ---------------------------------------------------- layer 3: compute only, no memory
G = 6 * 256 * 8
out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, G * 4)
inner = np.random.rand(256, 256).astype(np.float32)
in_t = torch.from_numpy(inner)
out_t = torch.empty_like(in_t)


def gpu_burn():
    k_burn(queue, (G,), None, out_buf)
    queue.finish()


def cpu_burn():
    for _ in range(12):
        torch.mm(in_t, in_t, out=out_t)


t_gb = timeit(gpu_burn, reps=3)
t_cb = timeit(cpu_burn, reps=3)
print('=== 第三层（决定性）：只算不访存 ===')
print('  单独: 核显烧算 %.4f s   CPU 烧算（cache 内）%.4f s' % (t_gb, t_cb))


def both_burn():
    k_burn(queue, (G,), None, out_buf)
    for _ in range(12):
        torch.mm(in_t, in_t, out=out_t)
    queue.finish()


t_bb = timeit(both_burn, reps=3)
print('  并发: %.4f s   参照 max %.4f   sum %.4f' % (t_bb, max(t_gb, t_cb), t_gb + t_cb))
print('  ⇒ 若 ≈ max ⇒ 无干扰，第二层的慢【来自内存总线】')
print('  ⇒ 若 ≈ sum ⇒ 【即使不访存也串行】，问题在调度/驱动/功耗，与总线无关')
