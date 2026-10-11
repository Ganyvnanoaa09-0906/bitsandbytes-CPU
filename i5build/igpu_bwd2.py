"""转置访问的真实代价 —— 给内核加 lda 步长，真按列读。

上一版的两个错：
  ① 我用 np.asfortranarray 预先把转置materialize了，bench 里又 ascontiguousarray
     转回行主序 ⇒ 内核读到的还是连续内存，**转置代价根本没被测到**（三个数相等就是证据）；
  ② 形状选了 M=2048,K=2048,N=512（A 复用率低、访存受限）⇒ 62 GFLOPS，
     和 igpu_waves_fixed 的 348（M=2048,N=2048,K=512）不是同一个形状，不可比。

这一版：
  · 内核吃 lda/ldb 步长，A 与 B 都以【行主序】传入，转置在**内核里按列读**实现；
  · 三个变体同形状对比，只差访问模式。
"""
import os
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))

HEAD = r'''
#define TS 64
#define TK 8
#define TM 4
#define TN 4
#define NTH_ 256
'''

LOAD_NN = r'''
        float2 va = *(const __global float2*)(A + (by+am)*LDAC + k0 + akp*2);
        As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
        float2 vb = *(const __global float2*)(B + (k0+bk)*LDBC + bx + bnp*2);
        Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;'''

# A 按列读：A 存为 [K, M] 行主序（lda=M），要取的是 Aᵀ[m][k] = A[k][m]
LOAD_TN = r'''
        As[akp*2+0][am] = A[(k0+akp*2)*LDAC + by + am];
        As[akp*2+1][am] = A[(k0+akp*2+1)*LDAC + by + am];
        float2 vb = *(const __global float2*)(B + (k0+bk)*LDBC + bx + bnp*2);
        Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;'''

# B 按列读：B 存为 [N, K] 行主序（ldb=K），要取的是 Bᵀ[k][n] = B[n][k]
LOAD_NT = r'''
        float2 va = *(const __global float2*)(A + (by+am)*LDAC + k0 + akp*2);
        As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
        Bs[bk][bnp*2+0] = B[(bx+bnp*2)*LDBC + k0 + bk];
        Bs[bk][bnp*2+1] = B[(bx+bnp*2+1)*LDBC + k0 + bk];'''

BODY = HEAD + r'''
__kernel void sgemm(const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    for (int k0 = 0; k0 < K; k0 += TK) {
__LOAD__
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
    for (int i=0;i<TM;i++) { int rc=by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) { int cc=bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc]=acc[i][j]; } }
}
'''

VARIANTS = {'nn': LOAD_NN, 'tn': LOAD_TN, 'nt': LOAD_NT}


def build(ctx, tag, ldac, ldbc, load):
    d = os.path.join(HERE, '_bwd2', tag)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        src = BODY.replace('__LOAD__', load)
        return cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                     f'-DLDAC={ldac}', f'-DLDBC={ldbc}'])
    finally:
        os.chdir(cwd)


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print('device:', dev.name, '|', dev.max_compute_units, 'CU')
    mf = cl.mem_flags
    rng = np.random.default_rng(0)

    # 用训练里的真实形状：M=2048(batch8×T256), K=512(d_model), N=2048(mlp down 的入维)
    M, K, N = 2048, 512, 2048
    fl = 2.0 * M * N * K
    print(f'\n形状 M={M} K={K} N={N}（前向/转置三个变体同形状）\n')
    print('%-24s %12s %10s %10s' % ('变体', 'GFLOPS', '秒', '相对前向'))

    a_nn = rng.random((M, K), dtype=np.float32)          # 前向的 A
    b_nn = rng.random((K, N), dtype=np.float32)
    a_tn = rng.random((K, M), dtype=np.float32)          # wgrad：A 存 [K,M]
    b_nt = rng.random((N, K), dtype=np.float32)          # dgrad：B 存 [N,K]
    cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gws = (int(np.ceil(N / 64)) * 16, int(np.ceil(M / 64)) * 16)

    base = None
    for tag, A, B, ldac, ldbc, ref in (
            ('① nn 前向', a_nn, b_nn, K, N, a_nn.astype(np.float64) @ b_nn),
            ('② tn wgrad(A转置)', a_tn, b_nn, M, N, a_tn.T.astype(np.float64) @ b_nn),
            ('③ nt dgrad(B转置)', a_nn, b_nt, K, K, a_nn.astype(np.float64) @ b_nt.T)):
        prog = build(ctx, tag[2:4], ldac, ldbc, VARIANTS[tag[2:4].strip()])
        ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A),
              cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B))
        kern = cl.Kernel(prog, 'sgemm')
        args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cbuf)
        kern(q, gws, (16, 16), *args)
        q.finish()
        t0 = time.perf_counter()
        for _ in range(3):
            kern(q, gws, (16, 16), *args)
        q.finish()
        dt = (time.perf_counter() - t0) / 3
        out = np.empty((M, N), np.float32)
        cl.enqueue_copy(q, out, cbuf).wait()
        err = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        g = fl / dt / 1e9
        if base is None:
            base = g
        ok = 'OK' if err < 1e-4 else f'⚠️ err={err:.1e}'
        print('%-24s %12.1f %10.3f %9.2fx  %s' % (tag, g, dt, g / base, ok))

    print('\n判读：②③ 若与 ① 接近 ⇒ 转置几乎不花钱，反向内核值得写；'
          '若掉一半以上 ⇒ 不值得。')


if __name__ == '__main__':
    main()
