"""两个判据，决定核显这条路还有没有肉。

背景：报告里那颗手写 SGEMM 在 1024³ 上 332 GFLOPS（fp32），而设备 fp32 峰值约 955~1150。
瓶颈已被 tile 扫描定位为【占用率：LDS 占用 + 寄存器压力】，不是带宽。于是两条出路：

  A. 这颗 APU 的 packed fp16（`v_pk_fma_f16`）到底是不是 2× fp32？
     桌面 Vega 是 2×，但 APU 上常被砍成 1×。bnb 的 SIMT 内核在 gfx9 上也【没有】用
     packed fp16（只在 RDNA3+ 上用 v_dot2）—— 这本身就是个疑点，必须实测。

  B. fp16 操作数（fp32 累加）能不能救回 TM=TN=16？
     fp32 下那个配置因寄存器溢出掉到 45 GFLOPS（灾难）。
     ⚠️ 但先想清楚：溢出的是【累加器】（TM×TN=256 个 float），而 fp16 操作数动的是
     LDS 里的 A/B，不动累加器。所以 B 有可能【根本救不了】—— 这个测试就是要证伪它。
     真正能减半累加器寄存器的是 fp16 累加，但那对训练数值不可接受（只适合推理）。

判据：A 若为 1×、B 若救不回来 ⇒ 核显在 fp32 训练负载上没有第二条路，结案。
"""
import sys
import time

import numpy as np
import pyopencl as cl

FMA_SRC = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fma_f32(__global float* out, const int iters) {
    float a = 1.0f + get_global_id(0) * 1e-7f;
    float b = 0.99999f, c = 1e-6f;
    for (int i = 0; i < iters; i++) { a = fma(a, b, c); b = fma(b, a, c); }
    out[get_global_id(0)] = a + b;
}
__kernel void fma_f16(__global float* out, const int iters) {
    half2 a = (half2)(1.0h, 1.0h);
    half2 b = (half2)(0.999h, 0.999h);
    half2 c = (half2)(0.001h, 0.001h);
    for (int i = 0; i < iters; i++) { a = fma(a, b, c); b = fma(b, a, c); }
    out[get_global_id(0)] = (float)(a.x + a.y + b.x + b.y);
}
'''

GEMM_FP32 = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void sgemm_f32(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW4 = TK/4, BCOL4 = TS/4;
    const int ATOT4 = TS*AROW4, BTOT4 = TK*BCOL4;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT4+BTOT4+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT4) {
                int row = f/AROW4, kq = f%AROW4;
                int gr = by+row, gc = k0+kq*4;
                float4 v = (float4)0.f;
                if (gr<M && gc+3<K) v = *(const __global float4*)(A+gr*K+gc);
                else if (gr<M) { float t[4]; for(int q=0;q<4;q++) t[q]=(gc+q<K)?A[gr*K+gc+q]:0.f;
                                 v=(float4)(t[0],t[1],t[2],t[3]); }
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT4+BTOT4) {
                int g = f-ATOT4, kk = g/BCOL4, cq = g%BCOL4;
                int gr = k0+kk, gc = bx+cq*4;
                float4 v = (float4)0.f;
                if (gr<K && gc+3<N) v = *(const __global float4*)(B+gr*N+gc);
                else if (gr<K) { float t[4]; for(int q=0;q<4;q++) t[q]=(gc+q<N)?B[gr*N+gc+q]:0.f;
                                 v=(float4)(t[0],t[1],t[2],t[3]); }
                Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}
'''

# fp16 操作数 + fp32 累加。A/B 用 half 存（LDS 占用减半、加载量减半），
# 读出来转 float 做 FMA —— 每次转换被 TM/TN=8 次复用摊薄。
GEMM_FP16 = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void sgemm_f16(const int M, const int N, const int K,
                        const __global half* restrict A,
                        const __global half* restrict B,
                        __global float* restrict C) {
    __local half As[TK][TS + 8];
    __local half Bs[TK][TS + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int AROW8 = TK/8, BCOL8 = TS/8;
    const int ATOT8 = TS*AROW8, BTOT8 = TK*BCOL8;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT8+BTOT8+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT8) {
                int row = f/AROW8, kq = f%AROW8;
                int gr = by+row, gc = k0+kq*8;
                half8 v = (half8)(0);
                if (gr<M && gc+7<K) v = *(const __global half8*)(A+gr*K+gc);
                else if (gr<M) { half t[8]; for(int q=0;q<8;q++) t[q]=(gc+q<K)?A[gr*K+gc+q]:(half)0;
                                 v=(half8)(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]); }
                As[kq*8+0][row]=v.s0; As[kq*8+1][row]=v.s1; As[kq*8+2][row]=v.s2; As[kq*8+3][row]=v.s3;
                As[kq*8+4][row]=v.s4; As[kq*8+5][row]=v.s5; As[kq*8+6][row]=v.s6; As[kq*8+7][row]=v.s7;
            } else if (f < ATOT8+BTOT8) {
                int g = f-ATOT8, kk = g/BCOL8, cq = g%BCOL8;
                int gr = k0+kk, gc = bx+cq*8;
                half8 v = (half8)(0);
                if (gr<K && gc+7<N) v = *(const __global half8*)(B+gr*N+gc);
                else if (gr<K) { half t[8]; for(int q=0;q<8;q++) t[q]=(gc+q<N)?B[gr*N+gc+q]:(half)0;
                                 v=(half8)(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]); }
                Bs[kk][cq*8+0]=v.s0; Bs[kk][cq*8+1]=v.s1; Bs[kk][cq*8+2]=v.s2; Bs[kk][cq*8+3]=v.s3;
                Bs[kk][cq*8+4]=v.s4; Bs[kk][cq*8+5]=v.s5; Bs[kk][cq*8+6]=v.s6; Bs[kk][cq*8+7]=v.s7;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=(float)As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=(float)Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
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
    ext = dev.extensions
    print(f'device {dev.name} | {dev.max_compute_units} CU | fp16 ext: '
          f'{"cl_khr_fp16" in ext}')
    opts = ['-cl-fast-relaxed-math', '-cl-mad-enable']

    # ---------------- A. FMA 峰值：fp32 vs packed fp16 ----------------
    print('\n=== A. FMA 微基准（每 lane 每迭代 2 次 FMA）===')
    try:
        prog = cl.Program(ctx, FMA_SRC).build(options=opts)
    except Exception as exc:                                  # noqa: BLE001
        print('  fp16 内核编译失败：', str(exc)[:300])
        return 1
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * 65536)
    iters = 20000
    n = 65536
    for name in ('fma_f32', 'fma_f16'):
        k = cl.Kernel(prog, name)
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        t0 = time.perf_counter()
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        dt = time.perf_counter() - t0
        fma = 2 * n * iters
        # fp16 若真是 packed，就是 4 次 fp16-FMA/lane/迭代
        print(f'  {name}: {dt*1e3:7.2f} ms   {fma/dt/1e9:7.1f} G-FMA/s '
              f'(若 packed 则实际 {2*fma/dt/1e9:.1f} G-fp16-FMA/s)')

    # ---------------- B. GEMM：fp32 vs fp16 操作数 ----------------
    print('\n=== B. GEMM：fp32 vs fp16 操作数（fp32 累加），我们的形状 ===')
    p32 = cl.Program(ctx, GEMM_FP32).build(options=opts)
    p16 = cl.Program(ctx, GEMM_FP16).build(options=opts)
    k32 = cl.Kernel(p32, 'sgemm_f32')
    k16 = cl.Kernel(p16, 'sgemm_f16')
    mf = cl.mem_flags
    shapes = [(1024, 1024, 1024), (2048, 512, 512), (2048, 2048, 512)]
    print(f"{'M×N×K':>20} {'fp32 ms':>9} {'fp32 GF':>9} {'fp16 ms':>9} {'fp16 GF':>9} "
          f"{'比值':>7} {'err':>9}")
    for M, N, K in shapes:
        a32 = np.random.rand(M, K).astype(np.float32)
        b32 = np.random.rand(K, N).astype(np.float32)
        a16 = a32.astype(np.float16)
        b16 = b32.astype(np.float16)
        fl = 2.0 * M * N * K
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        lws = (8, 8)
        bufs = {}
        for tag, a, b, dt_ in (('32', a32, b32, np.float32), ('16', a16, b16, np.float16)):
            bufs[tag] = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
                         cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
        c32 = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        c16 = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        o32 = np.empty((M, N), np.float32)
        o16 = np.empty((M, N), np.float32)
        k32(q, gws, lws, np.int32(M), np.int32(N), np.int32(K), bufs['32'][0],
            bufs['32'][1], c32)
        k16(q, gws, lws, np.int32(M), np.int32(N), np.int32(K), bufs['16'][0],
            bufs['16'][1], c16)
        q.finish()
        t0 = time.perf_counter()
        k32(q, gws, lws, np.int32(M), np.int32(N), np.int32(K), bufs['32'][0],
            bufs['32'][1], c32)
        q.finish()
        t32 = time.perf_counter() - t0
        t0 = time.perf_counter()
        k16(q, gws, lws, np.int32(M), np.int32(N), np.int32(K), bufs['16'][0],
            bufs['16'][1], c16)
        q.finish()
        t16 = time.perf_counter() - t0
        cl.enqueue_copy(q, o32, c32).wait()
        cl.enqueue_copy(q, o16, c16).wait()
        ref = a16.astype(np.float32) @ b16.astype(np.float32)
        err = float(np.abs(o16 - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'{M:>7}×{N:<5}×{K:<6} {t32*1e3:>9.1f} {fl/t32/1e9:>9.1f} '
              f'{t16*1e3:>9.1f} {fl/t16/1e9:>9.1f} {t32/t16:>6.2f}× {err:>9.1e}',
              flush=True)


if __name__ == '__main__':
    sys.exit(main())
