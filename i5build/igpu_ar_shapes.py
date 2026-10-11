"""AR 训练的真实形状上，核显 vs CPU 到底谁快？（之前只量过 1024³）

报告里那颗手写 OpenCL SGEMM 在 1024³ 上是 332.1 GFLOPS（超 DML 13%、CLBlast 19%），
CPU 同口径约 200 GFLOPS。但这决定不了任何事：**AR 训练一步的 GEMM 形状和 1024³
差得很远**。

AR（30.49M，d512/L10/FFN2048，T=256，batch 8）一步里每个 GEMM 的 M 都是
batch×T = 2048，而 N/K 只有 512~2048。这是"瘦长"形状：M 够大，但 K 小 ⇒
每个输出元素的计算量少、访存占比高。**核显的优势（332 vs 200）在这种形状上要打多少
折，没人量过。** 而这个折扣直接决定"值不值得为它写一整套反向内核"。

判据：若核显在这些形状上领先不到 ~1.5×，那么"写 dgrad/wgrad/softmax_bwd/norm_bwd/
Adam 全套内核"换 1.5× 就不划算 —— 这个问题当场结案，不写一行新内核。

kernel 用的是夺冠配置 TS=64/TK=8（来自 bench_concurrency2.py）。
"""
import sys
import time

import numpy as np
import torch

import pyopencl as cl

SRC = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
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
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW4 = TK/4, BCOL4 = TS/4;
    const int ATOT4 = TS*AROW4, BTOT4 = TK*BCOL4;
    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < (ATOT4+BTOT4+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
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
            for (int i=0;i<TM;i++)
                #pragma unroll
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        #pragma unroll
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}
'''


def main():
    plat = cl.get_platforms()[0]
    dev = plat.get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, SRC).build(options=['-cl-fast-relaxed-math'])
    kern = cl.Kernel(prog, 'sgemm_t')
    mf = cl.mem_flags

    # AR 一步里的真实形状（M = batch8 × T256 = 2048）
    shapes = [
        (1024, 1024, 1024, '锚点：报告里 332 GFLOPS 的那个形状'),
        (2048, 512, 512, 'AR：QKV / attn-out'),
        (2048, 2048, 512, 'AR：FFN up'),
        (2048, 512, 2048, 'AR：FFN down'),
        (2048, 1024, 512, 'AR：输出头'),
        (16 * 1024, 512, 512, '如果 batch 提到 64（M=16384）'),
    ]
    print(f'device: {dev.name}  {dev.max_compute_units} CU')
    print(f"{'形状 M×N×K':>22} {'CPU ms':>9} {'CPU GFLOPS':>11} "
          f"{'iGPU ms':>9} {'iGPU GFLOPS':>12} {'iGPU/CPU':>9}  说明")
    torch.set_num_threads(6)
    for M, N, K, note in shapes:
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
        fl = 2.0 * M * N * K

        ta = torch.from_numpy(a)
        tb = torch.from_numpy(b)
        for _ in range(2):
            torch.mm(ta, tb)
        t0 = time.perf_counter()
        for _ in range(5):
            torch.mm(ta, tb)
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3

        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        out = np.empty((M, N), dtype=np.float32)
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        lws = (8, 8)

        def call():
            kern(q, gws, lws, np.int32(M), np.int32(N), np.int32(K),
                 a_buf, b_buf, c_buf)

        call()
        q.finish()
        t0 = time.perf_counter()
        for _ in range(5):
            call()
        q.finish()
        gpu_ms = (time.perf_counter() - t0) / 5 * 1e3
        cl.enqueue_copy(q, out, c_buf).wait()

        ref = a @ b
        err = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'{M:>7}×{N:<5}×{K:<6} {cpu_ms:>9.1f} {fl/cpu_ms/1e6:>11.1f} '
              f'{gpu_ms:>9.1f} {fl/gpu_ms/1e6:>12.1f} {cpu_ms/gpu_ms:>8.2f}×  {note}'
              f'  [err {err:.1e}]', flush=True)


if __name__ == '__main__':
    sys.exit(main())
