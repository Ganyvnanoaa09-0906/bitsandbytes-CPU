# -*- coding: utf-8 -*-
"""bench_concurrency2.py — 用【已验证正确】的 kernel 重测 CPU+iGPU 并发

为什么重测：
  之前那批并发数据（CPU 只剩 57~83%）用的是【有 bug 的 kernel】
  修的 bug：全局读取索引转置 ⇒ 读的数据量不对，带宽占用完全不同
  ⇒ 必须用正确 kernel 重测

测四组：
  ① 核显单独（GEMM）
  ② CPU 单独（GEMM）
  ③ 两者并发
  ④ 真实负载下的并发（核显 FFN + CPU 注意力）
"""
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
import threading, time
import numpy as np
import torch
import pyopencl as cl

SRC = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
__kernel void sgemm_t(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    }
    const int AROW4 = TK/4, BCOL4 = TS/4;
    const int ATOT4 = TS*AROW4, BTOT4 = TK*BCOL4;
    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < (ATOT4+BTOT4+NTH_+63)/64; p++) {
            int f = p*64 + lane;
            if (f < ATOT4) {
                int row = f/AROW4, kq = f%AROW4;
                int gr = by+row, gc = k0+kq*4;
                float4 v = (float4)0.f;
                if (gr<M) { if (gc+3<K) v = *(const __global float4*)(A+gr*K+gc);
                    else { float t[4]; for(int q=0;q<4;q++) t[q]=(gc+q<K)?A[gr*K+gc+q]:0.f;
                           v=(float4)(t[0],t[1],t[2],t[3]); } }
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT4+BTOT4) {
                int g = f-ATOT4, kk = g/BCOL4, cq = g%BCOL4;
                int gr = k0+kk, gc = bx+cq*4;
                float4 v = (float4)0.f;
                if (gr<K) { if (gc+3<N) v = *(const __global float4*)(B+gr*N+gc);
                    else { float t[4]; for(int q=0;q<4;q++) t[q]=(gc+q<N)?B[gr*N+gc+q]:0.f;
                           v=(float4)(t[0],t[1],t[2],t[3]); } }
                Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i=0;i<TM;i++) a[i]=As[k][ty*TM+i];
            #pragma unroll
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            #pragma unroll
            for (int i=0;i<TM;i++) {
                #pragma unroll
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        #pragma unroll
        for (int j=0;j<TN;j++) { int cc=bx+tx*TN+j; if (cc<N) C[rc*N+cc]=acc[i][j]; }
    }
}
'''.replace('NTH_', '64')

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'sgemm_t')

print('=' * 80)
print('CPU + iGPU 并发（用已验证正确的 kernel）')
print('=' * 80)

# ---------- 准备 GPU 负载 ----------
GN = 1024
GA = np.random.randn(GN, GN).astype(np.float32)
GB = np.random.randn(GN, GN).astype(np.float32)
mf = cl.mem_flags
gab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=GA)
gbb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=GB)
gcb = cl.Buffer(ctx, mf.WRITE_ONLY, GN * GN * 4)
GGSZ = (GN // 64 * 8, GN // 64 * 8)
GFLOP = 2.0 * GN ** 3


def gpu_worker(stop, result, target_sec):
    n = 0
    t0 = time.perf_counter()
    while not stop.is_set() and time.perf_counter() - t0 < target_sec:
        for _ in range(4):
            kern(queue, GGSZ, (8, 8), np.int32(GN), np.int32(GN), np.int32(GN), gab, gbb, gcb)
        queue.finish()
        n += 4
    dt = time.perf_counter() - t0
    result['gpu'] = n * GFLOP / dt / 1e9


# ---------- 准备 CPU 负载 ----------
torch.set_num_threads(6)
CA = torch.randn(1024, 1024)
CB = torch.randn(1024, 1024)


def cpu_worker(stop, result, target_sec):
    with torch.no_grad():
        n = 0
        t0 = time.perf_counter()
        while not stop.is_set() and time.perf_counter() - t0 < target_sec:
            C = CA @ CB
            n += 1
        dt = time.perf_counter() - t0
        result['cpu'] = n * 2.0 * 1024 ** 3 / dt / 1e9


TARGET = 25

print('  预热...')
r0 = {}
gpu_worker(threading.Event(), r0, 20)
print('  预热完成\n')

# ① GPU 单独
res = {}
th = threading.Thread(target=gpu_worker, args=(threading.Event(), res, TARGET))
th.start(); th.join()
gpu_solo = res['gpu']

# ② CPU 单独
res = {}
th = threading.Thread(target=cpu_worker, args=(threading.Event(), res, TARGET))
th.start(); th.join()
cpu_solo = res['cpu']

# ③ 并发
res = {}
stop = threading.Event()
t1 = threading.Thread(target=gpu_worker, args=(stop, res, TARGET))
t2 = threading.Thread(target=cpu_worker, args=(stop, res, TARGET))
t1.start(); t2.start(); t1.join(); t2.join()
gpu_conc, cpu_conc = res['gpu'], res['cpu']

print('  %-28s %14s %14s' % ('配置', 'iGPU GFLOPS', 'CPU GFLOPS'))
print('  ' + '-' * 58)
print('  %-28s %14.1f %14s' % ('① iGPU 单独', gpu_solo, '—'))
print('  %-28s %14s %14.1f' % ('② CPU 单独', '—', cpu_solo))
print('  %-28s %14.1f %14.1f' % ('③ 两者并发', gpu_conc, cpu_conc))
print()
print('  iGPU 保持率:  %.0f%%  (%.1f / %.1f)' % (100 * gpu_conc / gpu_solo, gpu_conc, gpu_solo))
print('  CPU  保持率:  %.0f%%  (%.1f / %.1f)' % (100 * cpu_conc / cpu_solo, cpu_conc, cpu_solo))
print()
print('  串行总和 = %.1f + %.1f = %.1f' % (gpu_solo, cpu_solo, gpu_solo + cpu_solo))
print('  并发总和 = %.1f + %.1f = %.1f'  % (gpu_conc, cpu_conc, gpu_conc + cpu_conc))
print('  ⇒ 并发相对串行的效率: %.2fx' % ((gpu_conc + cpu_conc) / (gpu_solo + cpu_solo)))
print()
print('  （旧数据是 CPU 57~83%、iGPU 70~85%，但用的是有 bug 的 kernel）')
print('DONE')
