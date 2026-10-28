# -*- coding: utf-8 -*-
"""verify_gemm2.py — 修正载入逻辑后的 SGEMM，先验证正确性再测速

上一版的 bug：
  int r = (flat4*4)/TS;   // 这其实是 K 索引
  int c = (flat4*4)%TS;   // 这其实是行索引
  As[r][c] = ...;         // LDS 布局对
  A + (by+r)*K + (k0+c)   // 全局读取用反了 ⇒ 读的是 A 的转置

修正：把 flat 的枚举意义写清楚
  A 块 (TS行 × TK列)：
    每行有 TK/4 个 float4（沿 K 连续）
    flat4 -> row = flat4 / (TK/4),  kq = flat4 % (TK/4)
    全局读 A[(by+row)*K + k0 + kq*4]  ← 连续 4 个 ✅
    LDS 写 As[kq*4+..][row]           ← 布局 [k][row] 匹配计算 ✅
  B 块 (TK行 × TS列)：
    每行有 TS/4 个 float4（沿 N 连续）
    flat4 -> k = flat4 / (TS/4),  cq = flat4 % (TS/4)
    全局读 B[(k0+k)*N + bx + cq*4]
    LDS 写 Bs[k][cq*4+..]
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time
import numpy as np
import pyopencl as cl

SRC = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH 64
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

    const int AROW4 = TK / 4;      // A 每行的 float4 数
    const int BCOL4 = TS / 4;      // B 每行的 float4 数
    const int ATOT4 = TS * AROW4;  // A 块总 float4
    const int BTOT4 = TK * BCOL4;  // B 块总 float4

    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < (ATOT4 + BTOT4 + NTH - 1) / NTH; p++) {
            int f = p * NTH + lane;
            if (f < ATOT4) {
                int row = f / AROW4, kq = f % AROW4;
                int gr = by + row, gc = k0 + kq * 4;
                float4 v = (float4)0.f;
                if (gr < M) {
                    if (gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
                    else { float t[4];
                        for (int q=0;q<4;q++) t[q] = (gc+q < K) ? A[gr*K+gc+q] : 0.f;
                        v = (float4)(t[0], t[1], t[2], t[3]); }
                }
                As[kq*4+0][row] = v.x; As[kq*4+1][row] = v.y;
                As[kq*4+2][row] = v.z; As[kq*4+3][row] = v.w;
            } else if (f < ATOT4 + BTOT4) {
                int g = f - ATOT4;
                int kk = g / BCOL4, cq = g % BCOL4;
                int gr = k0 + kk, gc = bx + cq * 4;
                float4 v = (float4)0.f;
                if (gr < K) {
                    if (gc + 3 < N) v = *(const __global float4*)(B + gr * N + gc);
                    else { float t[4];
                        for (int q=0;q<4;q++) t[q] = (gc+q < N) ? B[gr*N+gc+q] : 0.f;
                        v = (float4)(t[0], t[1], t[2], t[3]); }
                }
                Bs[kk][cq*4+0] = v.x; Bs[kk][cq*4+1] = v.y;
                Bs[kk][cq*4+2] = v.z; Bs[kk][cq*4+3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
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
        int rc = by + ty * TM + i;
        if (rc >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int cc = bx + tx * TN + j;
            if (cc < N) C[rc * N + cc] = acc[i][j];
        }
    }
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'sgemm_t')

print('=' * 76)
print('SGEMM 修正版：先验正确性，再测速度')
print('=' * 76)
print('  配置: TS=64 TK=8 TM=8 TN=8 NTH=64')
print()

print('  --- 正确性（跟 numpy 对拍）---')
ok_all = True
for (M, N, K) in ((64, 64, 64), (128, 128, 128), (256, 192, 128),
                  (201, 137, 93), (512, 512, 512), (1024, 1024, 1024)):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    # 网格要按 ceil 算，因为 M/N 可能不是 64 的倍数
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
    for _ in range(4):
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
    C = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, C, cb)
    queue.finish()

    ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    denom = np.maximum(np.abs(ref), 1e-2)
    maxrel = float((np.abs(C - ref) / denom).max())
    tol = 5e-6 * K + 1e-4   # fp32 顺序累加的固有上界（参考用 float64 算，所以比的是 fp32 自己）
    good = maxrel < tol
    ok_all = ok_all and good
    print('    (%4d,%4d)x(%4d,%4d)  最大相对误差 %.2e (容差 %.1e) %s'
          % (M, K, K, N, maxrel, tol, '✅' if good else '❌'))

print()
if ok_all:
    print('  ⇒ 正确性【全部通过】，现在测的速度才有意义')
else:
    print('  ⇒ 仍有错误，速度数字无效')
    print('DONE'); sys.exit(0)

print()
print('  --- 速度（预热 20s，交替 5 轮取最大）---')
N = 1024
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
mf = cl.mem_flags
ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
cb = cl.Buffer(ctx, mf.WRITE_ONLY, N * N * 4)
gsz = (N // 64 * 8, N // 64 * 8)


def one():
    for _ in range(4):
        kern(queue, gsz, (8, 8), np.int32(N), np.int32(N), np.int32(N), ab, bb, cb)
    queue.finish()


t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    one()
best = 0.0
for _ in range(5):
    t0 = time.perf_counter()
    one()
    dt = (time.perf_counter() - t0) / 4.0
    best = max(best, 2.0 * N ** 3 / dt / 1e9)
print('  1024³ 最佳:  %.1f GFLOPS' % best)
print()
print('  对照（可信的成熟库）: CLBlast 278.6   DML 294.5')
print('  判断: %s' % ('✅ 超过成熟库' if best > 294.5 else
                    '⚠️ 低于 DML，但可能仍优于 CLBlast' if best > 278.6 else
                    '— 低于两个成熟库'))
print('DONE')
