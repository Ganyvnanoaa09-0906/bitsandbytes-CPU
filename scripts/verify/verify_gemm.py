# -*- coding: utf-8 -*-
"""verify_gemm.py — 验证最优 tile 配置的【正确性】

一个快的错 kernel 毫无价值。这里跟 numpy 的参考实现对拍。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
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

SRC = r'''
#define TS 64
#define TK 4
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
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    }
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int lane = ty * 8 + tx;
        #pragma unroll
        for (int p = 0; p < (TK*64*2)/(64*4); p++) {
            int flat4 = p*64 + lane, total4 = TK*64/4;
            if (flat4 < total4) {
                int r=(flat4*4)/64, c=(flat4*4)%64, gr=by+r, gc=k0+c;
                float4 v = (gr<M && gc+3<K) ? *(const __global float4*)(A+gr*K+gc) : (float4)0.f;
                As[r][c]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
            } else {
                int f2=flat4-total4, r=(f2*4)/64, c=(f2*4)%64, gr=k0+r, gc=bx+c;
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

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'sgemm_t')

print('=' * 74)
print('SGEMM 正确性验证（跟 numpy 对拍）')
print('=' * 74)
print('  配置: TS=64 TK=4 TM=8 TN=8')
print()

ok_all = True
for (M, N, K) in ((64, 64, 64), (128, 128, 128), (256, 192, 128),
                  (201, 137, 93), (512, 512, 512), (1024, 1024, 1024)):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = (N // 64 * 8, M // 64 * 8)
    for _ in range(4):
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
    C = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, C, cb)
    queue.finish()

    ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    # 相对误差
    denom = np.maximum(np.abs(ref), 1e-3)
    rel = np.abs(C - ref) / denom
    maxrel = float(rel.max())
    good = maxrel < 1e-3
    ok_all = ok_all and good
    print('  %-20s 形状 %-18s 最大相对误差 %.2e  %s'
          % ('', '(%d,%d)x(%d,%d)' % (M, K, K, N), maxrel, '✅' if good else '❌'))

print()
if ok_all:
    print('  ⇒ 【全部通过】—— 这个 343 GFLOPS 是真的')
else:
    print('  ⇒ 【有不通过的情况】—— 那个速度数字无效')
print('DONE')
