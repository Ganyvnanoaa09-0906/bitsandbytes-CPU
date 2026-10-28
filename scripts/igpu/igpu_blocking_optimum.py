"""Does shrinking the CPU's working set buy back the iGPU? And is the combination ever worth it?

7.88 established that the iGPU's loss tracks the CPU's memory footprint and not its power: a pure ALU
spin costs it 3% while torch.mm at 48 MB costs it 6x. That makes the footprint a tunable, and raises a
question the earlier "power arbitration is a switch" reading could not even pose -- whether there is a
tiling at which the two engines together beat either one alone.

The shape is the model's real MLP (2048 x 2736 x 512), and the CPU thread count is held at 5 so the
working set is the only variable. A CPU GEMM whose operands all shrink requires tiling in two
dimensions: chunking M alone leaves B at 5.6 MB and therefore leaves the working set above L3 no matter
how small the chunks get.

    full                 A 4.0 + B 5.6 + C 22.4  = 32.0 MB
    M 512                1.0 + 5.6 +  5.6       = 12.2 MB     B is the floor
    MxN 512x512          1.0 + 1.0 +  0.56      =  2.6 MB
    MxN 256x256          0.52 + 0.52 + 0.26     =  1.3 MB
    MxN 128x128          0.26 + 0.26 + 0.065    =  0.59 MB

Measured simultaneously: the CPU's own throughput doing the tiled work, and the iGPU's throughput on a
fixed 5.7 GFLOP GEMM running at the same time. Combined is their sum, which is the quantity that has to
beat 392.7 -- the iGPU alone -- for concurrency to be worth anything at all.

Criterion, fixed in advance: if any tiling makes the combined figure exceed the iGPU alone (392.7
GFLOPS), concurrency becomes worth pursuing and this becomes a real optimisation. If the best tiling
still loses, the answer is unchanged from 7.62 but for a different reason -- not a hard power wall, but
a cache budget that neither engine can escape, in which case giving the whole job to the iGPU stays the
right call.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from scripts.lib.measure_guard import assert_idle, stamp

SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
q = cl.CommandQueue(ctx)
mf = cl.mem_flags
M, N, K = 2048, 2736, 512
GPU_FLOP = 2.0 * M * N * K
rng = np.random.default_rng(5)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')
gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
CPU_FLOP = GPU_FLOP          # same shape, so the comparison is like for like


def gpu_call():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    q.finish()


def gpu_steady(reps=6):
    for _ in range(2):
        gpu_call()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu_call()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def make_cpu_job(tile_m, tile_n, nthreads):
    """Returns (run_once, working_set_MB). Tiling in both dimensions so every operand shrinks."""
    if tile_m is None:
        am = torch.randn(M, K)
        bm = torch.randn(K, N)
        cm = torch.empty(M, N)

        def once():
            torch.mm(am, bm, out=cm)
        ws = (am.numel() + bm.numel() + cm.numel()) * 4 / 1024 ** 2
        return once, ws
    tm, tn = tile_m, tile_n
    A2 = torch.arange(M, dtype=torch.float32).unsqueeze(1).expand(M, K).contiguous()
    B2 = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(K, N).contiguous()
    C2 = torch.empty(M, N)

    def once():
        for i in range(0, M, tm):
            ie = min(i + tm, M)
            for j in range(0, N, tn):
                je = min(j + tn, N)
                torch.mm(A2[i:ie], B2[:, j:je], out=C2[i:ie, j:je])
    ws = ((tm * K) + (K * tn) + (tm * tn)) * 4 / 1024 ** 2
    return once, ws


assert_idle()
torch.set_num_threads(5)
stamp('压小 CPU 工作集能不能把核显买回来')
print()
print('  CPU 线程固定 5 · 核显 GEMM 固定 2048x2736x512（5.7 GFLOP）')
print()

idle_gpu = gpu_steady()
gpu_alone = GPU_FLOP / idle_gpu / 1e9
print('  核显单独: %.3f ms ⇒ 【%.1f GFLOPS】' % (idle_gpu * 1e3, gpu_alone))
print()

# CPU alone, for reference, on the full shape
once_full, ws_full = make_cpu_job(None, None, 5)
for _ in range(2):
    once_full()
ts = []
for _ in range(3):
    t0 = time.perf_counter()
    once_full()
    ts.append(time.perf_counter() - t0)
cpu_alone = CPU_FLOP / statistics.median(ts) / 1e9
print('  CPU 单独（全量）: %.1f GFLOPS' % cpu_alone)
print()

print('  %-16s %9s %10s %11s %11s %11s' %
      ('分块', '工作集', 'CPU GFLOPS', '核显 ms', '核显 GFLOPS', '合计'))
print('  ' + '-' * 76)

configs = [
    ('全量', None, None, 32.0),
    ('M 512', 512, N, 12.2),
    ('MxN 512x512', 512, 512, 2.6),
    ('MxN 256x256', 256, 256, 1.3),
    ('MxN 128x128', 128, 128, 0.59),
]
rows = []
for name, tm, tn, ws_nominal in configs:
    once, ws = make_cpu_job(tm, tn, 5)
    # CPU throughput on its own, tiled
    for _ in range(2):
        once()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        once()
        ts.append(time.perf_counter() - t0)
    cpu_g = CPU_FLOP / statistics.median(ts) / 1e9

    # now both at once
    stop = [False]
    count = [0]

    def worker():
        while not stop[0]:
            once()
            count[0] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(1.5)
    gt = gpu_steady()
    stop[0] = True
    th.join(timeout=5)
    time.sleep(1.5)

    gpu_g = GPU_FLOP / gt / 1e9
    rows.append((name, ws_nominal, cpu_g, gt, gpu_g, cpu_g + gpu_g))
    print('  %-16s %7.1f MB %10.1f %11.3f %11.1f %11.1f'
          % (name, ws_nominal, cpu_g, gt * 1e3, gpu_g, cpu_g + gpu_g))

print()
best = max(rows, key=lambda r: r[5])
print('  最好的合计: %s ⇒ 【%.1f GFLOPS】（CPU %.1f + 核显 %.1f）'
      % (best[0], best[5], best[2], best[4]))
print('  而【核显单独】= %.1f GFLOPS' % gpu_alone)
print()
print('  判据（事先固定）: 合计 > 核显单独（%.1f）⇒ 并发有价值' % gpu_alone)
if best[5] > gpu_alone:
    print('  ⇒ ★【成立】✓ 并发值得做，而这开了一条新路')
else:
    print('  ⇒ ★【不成立】✗ —— 但理由从"功耗墙"变成了【缓存预算】')
    print('  ⇒ 而这条教训是:两个引擎共享一个 8 MB L3，而合计吞吐受它管辖')
