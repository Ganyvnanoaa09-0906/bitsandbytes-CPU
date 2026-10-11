"""加载映射探针：只跑【一个 k-tile 的加载】，把 LDS 内容 dump 出来逐格比对。

为什么需要它：我手改加载映射已经翻车两次（双缓冲 NaN、v1 NaN），而手推索引
每次都"看起来覆盖完整"。所以在改任何东西之前，先把这一层验干净：
把 As/Bs 的每个格子跟 numpy 算出来的期望逐格比，不一致就立刻指出【哪一个格子】。

同时验两套映射（v0 的 p 循环版 / v1 的指针外提版），确认它们真的等价。
"""
import os
import sys

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

HEAD = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
'''

PROBE_V0 = HEAD + r'''
__kernel void probe_v0(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* dumpA, __global float* dumpB) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    const int AROW = TK/4, BCOL = TS/4;
    const int ATOT = TS*AROW, BTOT = TK*BCOL;
    const int k0 = 0;
    for (int p = 0; p < (ATOT+BTOT+NTH_-1)/NTH_; p++) {
        int f = p*NTH_ + lane;
        if (f < ATOT) {
            int row = f/AROW, kq = f%AROW;
            int gr = by+row, gc = k0+kq*4;
            float4 v = (float4)0.f;
            if (gr<M && gc+3<K) v = *(const __global float4*)(A+gr*K+gc);
            As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
        } else if (f < ATOT+BTOT) {
            int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
            int gr = k0+kk, gc = bx+cq*4;
            float4 v = (float4)0.f;
            if (gr<K && gc+3<N) v = *(const __global float4*)(B+gr*N+gc);
            Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lane; i < TS*TK; i += 64) {
        int k = i/TS, r = i%TS;
        dumpA[i] = As[k][r];      // dumpA[k*TS + row]
        dumpB[i] = Bs[k][r];      // dumpB[k*TS + col]
    }
}
'''

PROBE_V1 = HEAD + r'''
__kernel void probe_v1(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* dumpA, __global float* dumpB) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    const int AROW = TK/4, BCOL = TS/4;
    const int ar0 = lane/AROW,      ak0 = (lane%AROW)*4;
    const int ar1 = (lane+64)/AROW, ak1 = ((lane+64)%AROW)*4;
    const int bk0 = lane/BCOL,      bc0 = (lane%BCOL)*4;
    const int bk1 = (lane+64)/BCOL, bc1 = ((lane+64)%BCOL)*4;
    const __global float* Ap0 = A + (by+ar0)*KC + ak0;
    const __global float* Ap1 = A + (by+ar1)*KC + ak1;
    const __global float* Bp0 = B + bk0*NC + bx + bc0;
    const __global float* Bp1 = B + bk1*NC + bx + bc1;
    float4 va0 = *(const __global float4*)(Ap0);
    float4 va1 = *(const __global float4*)(Ap1);
    float4 vb0 = *(const __global float4*)(Bp0);
    float4 vb1 = *(const __global float4*)(Bp1);
    As[ak0/4+0][ar0]=va0.x; As[ak0/4+1][ar0]=va0.y;
    As[ak0/4+2][ar0]=va0.z; As[ak0/4+3][ar0]=va0.w;
    As[ak1/4+0][ar1]=va1.x; As[ak1/4+1][ar1]=va1.y;
    As[ak1/4+2][ar1]=va1.z; As[ak1/4+3][ar1]=va1.w;
    Bs[bk0][bc0+0]=vb0.x; Bs[bk0][bc0+1]=vb0.y;
    Bs[bk0][bc0+2]=vb0.z; Bs[bk0][bc0+3]=vb0.w;
    Bs[bk1][bc1+0]=vb1.x; Bs[bk1][bc1+1]=vb1.y;
    Bs[bk1][bc1+2]=vb1.z; Bs[bk1][bc1+3]=vb1.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lane; i < TS*TK; i += 64) {
        int k = i/TS, r = i%TS;
        dumpA[i] = As[k][r];
        dumpB[i] = Bs[k][r];
    }
}
'''


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    M = N = K = 512
    a = np.arange(M * K, dtype=np.float32).reshape(M, K) % 97 + 0.5
    b = np.arange(K * N, dtype=np.float32).reshape(K, N) % 89 + 0.25
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
    da = cl.Buffer(ctx, mf.WRITE_ONLY, 64 * 8 * 4)
    db = cl.Buffer(ctx, mf.WRITE_ONLY, 64 * 8 * 4)
    oa = np.full(64 * 8, np.nan, np.float32)
    ob = np.full(64 * 8, np.nan, np.float32)

    # 期望：As[k][row] = A[by+row][k0+k]，by=0,k0=0；Bs[k][col] = B[k][bx+col]，bx=0
    expA = np.zeros((8, 64), np.float32)
    for k in range(8):
        expA[k] = a[0:64, k]
    expB = np.zeros((8, 64), np.float32)
    for k in range(8):
        expB[k] = b[k, 0:64]

    for name, src in (('probe_v0', PROBE_V0), ('probe_v1', PROBE_V1)):
        prog = cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                     f'-DKC={K}', f'-DNC={N}'])
        k = cl.Kernel(prog, name)
        k(q, (8, 8), (8, 8), np.int32(M), np.int32(N), np.int32(K),
          ab[0], ab[1], da, db)
        q.finish()
        cl.enqueue_copy(q, oa, da).wait()
        cl.enqueue_copy(q, ob, db).wait()
        A2 = oa.reshape(8, 64)
        B2 = ob.reshape(8, 64)
        badA = np.argwhere(A2 != expA)
        badB = np.argwhere(B2 != expB)
        print(f'{name}: As 错格 {len(badA):>4}  Bs 错格 {len(badB):>4}   '
              f'NaN {int(np.isnan(A2).sum())}/{int(np.isnan(B2).sum())}')
        for tag, bad, exp, got in (('A', badA, expA, A2), ('B', badB, expB, B2)):
            for (kk, rr) in bad[:4]:
                print(f'   {tag}[k={kk}][pos={rr}]  期望 {exp[kk, rr]:.3f}  '
                      f'得到 {got[kk, rr]:.3f}')
        if not len(badA) and not len(badB):
            print('   ✅ 这一套映射正确')


if __name__ == '__main__':
    sys.exit(main())
