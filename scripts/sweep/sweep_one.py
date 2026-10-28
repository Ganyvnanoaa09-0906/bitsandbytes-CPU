# -*- coding: utf-8 -*-
"""sweep_one.py — 单配置扫描（独立进程跑，避免配置间互相干扰）

用法: python sweep_one.py <TS> <TM> <TN> <TK>
输出一行结果，供驱动脚本收集。

为什么要独立进程：
  之前的扫描在同进程内跑 7 个配置，结果严重不可复现
  （同一配置两次测到 231.3 和 98.9）
  而热衰减实测是 1.00x ⇒ 不是热，是同进程内的某种累积效应
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
import sys, time
import numpy as np
import pyopencl as cl

TS, TM, TN, TK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
NTH = (TS // TM) * (TS // TN)

SRC = r'''
#define TS   @TS@
#define TM   @TM@
#define TN   @TN@
#define TK   @TK@
#define NTH  ((TS/TM) * (TS/TN))
__kernel void sgemm_t(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    }
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int lane = ty * (TS/TN) + tx;
        #pragma unroll
        for (int p = 0; p < (TK*TS*2)/(NTH*4); p++) {
            int flat4 = p*NTH + lane, total4 = TK*TS/4;
            if (flat4 < total4) {
                int r=(flat4*4)/TS, c=(flat4*4)%TS, gr=by+r, gc=k0+c;
                float4 v = (gr<M && gc+3<K) ? *(const __global float4*)(A+gr*K+gc) : (float4)0.f;
                As[r][c]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
            } else {
                int f2=flat4-total4, r=(f2*4)/TS, c=(f2*4)%TS, gr=k0+r, gc=bx+c;
                float4 v = (gr<K && gc+3<N) ? *(const __global float4*)(B+gr*N+gc) : (float4)0.f;
                Bs[r][c]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty*TM+i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx*TN+j];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int rc = by + ty*TM + i;
        if (rc >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int cc = bx + tx*TN + j;
            if (cc < N) C[rc*N + cc] = acc[i][j];
        }
    }
}
'''
src = SRC.replace('@TS@', str(TS)).replace('@TM@', str(TM)).replace('@TN@', str(TN)).replace('@TK@', str(TK))

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)

try:
    kern = cl.Kernel(cl.Program(ctx, src).build(), 'sgemm_t')
except Exception as e:
    print('RESULT %d %d %d %d FAIL %s' % (TS, TM, TN, TK, str(e)[:40]))
    sys.exit(0)

N = 1024
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
mf = cl.mem_flags
ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
cb = cl.Buffer(ctx, mf.WRITE_ONLY, N * N * 4)
gsz = (N // TS * (TS // TN), N // TS * (TS // TM))
ls = (TS // TN, TS // TM)

# 队列限流（不限流会排爆队列，queue.finish() 永远等不完）
def one():
    for _ in range(4):
        kern(queue, gsz, ls, np.int32(N), np.int32(N), np.int32(N), ab, bb, cb)
    queue.finish()

# 预热 20 秒（核显必须！）
t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    one()
# 正式测：交替多轮取最大
best = 0.0
for _ in range(5):
    t0 = time.perf_counter()
    one()
    dt = (time.perf_counter() - t0) / 4.0
    gf = 2.0 * N ** 3 / dt / 1e9
    best = max(best, gf)
print('RESULT %d %d %d %d %.1f' % (TS, TM, TN, TK, best))
