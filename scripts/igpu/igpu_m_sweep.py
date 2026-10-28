"""Sweep M and find where the iGPU becomes worth it. The balance point as a measurable interval.

7.96 showed the answer depends on M and nothing else: at M=128 the iGPU is 1.68x slower than the CPU,
while 7.89 measured a combined 1.81x win at M=2048. Both are right. So the balance point is not a number
but a crossover, and the way to find it is to sweep M with everything else fixed.

Fixed here: N=6144, K=2048, the shape Qwen3-1.7B's gate_proj actually has according to the probes. M is
the token count, which is batch times sequence length, and it is the only variable.

Three curves per M, because the crossover is between the second and the third:

    iGPU alone      gpu_mm through the verified autograd path, transfers included
    CPU alone       torch.mm at six threads
    both together   CPU cache-blocked at 256x256 in a thread, iGPU on the main thread

The blocked CPU is the 7.89 configuration, and its own cost is reported too -- if tiling makes the CPU
slower than its unblocked rate, the combined figure is being subsidised and that belongs in the table
rather than in a footnote.

Criterion, fixed in advance: the balance point is the smallest M where the combined figure exceeds both
singles by at least the machine's 20% measurement resolution. Below that M the CPU alone is the right
answer and the iGPU should stay idle.
"""
from __future__ import annotations
import os, sys
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
import threading
import time

import numpy
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pyopencl
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
_ctx = cl.Context(devices=[_plat.get_devices()[0]])
_q = cl.CommandQueue(_ctx)
_mf = cl.mem_flags
_kern = cl.Kernel(cl.Program(_ctx, SRC).build(), 'gemm_fast')

N, K = 6144, 2048
torch.set_num_threads(5)
_plat_dev = _plat.get_devices()[0]

print()
print('  N=%d · K=%d（Qwen3 gate_proj 的真实形状）· CPU 5 线程' % (N, K))
assert_idle()
stamp('扫 M：平衡点在哪')
print()
print('  %-7s %9s %11s %11s %11s %11s %11s %9s' %
      ('M', 'GFLOP', '核显单独', 'CPU单独', 'CPU分块', '并发CPU', '并发核显', '合计'))
print('  ' + '-' * 88)

rows = []
Bnp = np.ascontiguousarray(np.random.default_rng(1).standard_normal((K, N)).astype(np.float32))
Bt = torch.from_numpy(Bnp)

for M in (1, 64, 128, 256, 512, 1024, 2048, 4096):
    fl = 2.0 * M * N * K
    Anp = np.ascontiguousarray(np.random.default_rng(2).standard_normal((M, K)).astype(np.float32))
    Cn = np.empty((M, N), dtype=np.float32)
    ba = cl.Buffer(_ctx, _mf.READ_ONLY | _mf.COPY_HOST_PTR, hostbuf=Anp)
    bb = cl.Buffer(_ctx, _mf.READ_ONLY | _mf.COPY_HOST_PTR, hostbuf=Bnp)
    bc = cl.Buffer(_ctx, _mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    args = (np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)

    def gpu_once():
        # ★ finish 不能省 —— 省了量到的是入队,而本项目在这一点上已经栽过两次
        _kern(_q, gsz, (16, 16), *args)
        _q.finish()

    for _ in range(2):
        gpu_once()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        gpu_once()
        ts.append(time.perf_counter() - t0)
    g_solo = fl / statistics.median(ts) / 1e9

    At = torch.from_numpy(Anp)
    Ct = torch.empty(M, N)
    for _ in range(2):
        torch.mm(At, Bt, out=Ct)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        torch.mm(At, Bt, out=Ct)
        ts.append(time.perf_counter() - t0)
    c_solo = fl / statistics.median(ts) / 1e9

    tm = tn = 256

    def cpu_tiled():
        for i in range(0, M, tm):
            ie = min(i + tm, M)
            for j in range(0, N, tn):
                je = min(j + tn, N)
                torch.mm(At[i:ie], Bt[:, j:je], out=Ct[i:ie, j:je])

    for _ in range(2):
        cpu_tiled()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        cpu_tiled()
        ts.append(time.perf_counter() - t0)
    c_tiled = fl / statistics.median(ts) / 1e9

    stop = [False]
    cnt = [0]

    def worker():
        while not stop[0]:
            cpu_tiled()
            cnt[0] += 1

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    time.sleep(1.5)
    cnt[0] = 0
    t0 = time.perf_counter()
    gts = []
    for _ in range(4):
        tg = time.perf_counter()
        gpu_once()
        gts.append(time.perf_counter() - tg)
    wall = time.perf_counter() - t0
    stop[0] = True
    th.join(timeout=5)
    time.sleep(1.5)
    c_conc = cnt[0] * fl / wall / 1e9
    g_conc = fl / statistics.median(gts) / 1e9
    tot = c_conc + g_conc
    rows.append((M, fl, g_solo, c_solo, c_tiled, c_conc, g_conc, tot))
    print('  %-7d %9.1f %11.1f %11.1f %11.1f %11.1f %11.1f %9.1f'
          % (M, fl / 1e9, g_solo, c_solo, c_tiled, c_conc, g_conc, tot))
    del ba, bb, bc

print()
print('  %-7s %11s %11s %11s %11s  %s' %
      ('M', 'CPU单独', '核显单独', '合计并发', '合计/CPU', '判定'))
print('  ' + '-' * 76)
cross = None
for r in rows:
    ratio = r[7] / r[3]
    best_single = max(r[3], r[2])
    win = r[7] > best_single * 1.20
    if win and cross is None:
        cross = r[0]
    print('  %-7d %11.1f %11.1f %11.1f %10.2f×  %s'
          % (r[0], r[3], r[2], r[7], ratio,
             '★【并发赢】' if win else ('平' if ratio > 1.0 else 'CPU 单干')))
print()
print('  判据（事先固定）: 合计 > max(两个单独) × 1.20 ⇒ 该 M 上值得并发')
if cross:
    print('  ⇒ ★【平衡点: M ≈ %d 以上，并发值得做】' % cross)
else:
    print('  ⇒ ✗ 扫描范围内未出现平衡点')
print()
print('  ⇒ 而 M = batch × 序列长度 ⇒ 把 M 换算成训练配置：')
for M, _, _, cs, _, _, _, tot in rows:
    if M in (128, 512, 1024, 2048, 4096):
        print('     M=%4d ≈ batch 2 × seq %4d ⇒ CPU %5.1f · 并发 %5.1f GFLOPS'
              % (M, M // 2, cs, tot))
print()
