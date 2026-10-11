"""写反向内核之前必须先量的一件事：**转置形状在核显上还快吗？**

前向 769 GFLOPS 是在 C = A·B（A:[M,K], B:[K,N]，都是行主序连续）上量的。
而反向是转置的：

    wgrad  dW = Xᵀ·dY      A 要按列读（转置）
    dgrad  dX = dY·Wᵀ      B 要按列读（转置）

转置改变访存模式。我们的核显带宽/LDS 本来就紧，所以**先量再写**：
若转置形状也能到 600~700，那套内核值得写；若只有 200，就不值得。

做法：同一个 256 线程内核（命中 769 的那个配置），分别喂
  ① 正常 A、B                 → 基线
  ② A 转置（用 Aᵀ 的视图）     → 模拟 wgrad
  ③ B 转置                     → 模拟 dgrad
"""
import os
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))

K256 = r'''
#define TS 64
#define TK 8
#define TM 4
#define TN 4
#define NTH_ 256
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
        float2 va = *(const __global float2*)(A + (by+am)*KC + k0 + akp*2);
        As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
        float2 vb = *(const __global float2*)(B + (k0+bk)*NC + bx + bnp*2);
        Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;
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


def build(ctx, K, N, tag):
    d = os.path.join(HERE, '_bwd', tag)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        return cl.Program(ctx, K256).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                     f'-DKC={K}', f'-DNC={N}'])
    finally:
        os.chdir(cwd)


def bench(q, prog, a, b, nt=3):
    """a: [M,K] 或已转置视图；b: [K,N]。返回 GFLOPS 与相对误差。"""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    gws = (int(np.ceil(n / 64)) * 16, int(np.ceil(m / 64)) * 16)
    mf = cl.mem_flags
    ab = (cl.Buffer(ctx_ref[0], mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(a)),
          cl.Buffer(ctx_ref[0], mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(b)))
    cb = cl.Buffer(ctx_ref[0], mf.WRITE_ONLY, m * n * 4)
    kern = cl.Kernel(prog, 'sgemm')
    args = (np.int32(m), np.int32(n), np.int32(k), ab[0], ab[1], cb)
    kern(q, gws, (16, 16), *args)
    q.finish()
    t0 = time.perf_counter()
    for _ in range(nt):
        kern(q, gws, (16, 16), *args)
    q.finish()
    dt = (time.perf_counter() - t0) / nt
    out = np.empty((m, n), np.float32)
    cl.enqueue_copy(q, out, cb).wait()
    ref = np.ascontiguousarray(a).astype(np.float64) @ np.ascontiguousarray(b).astype(np.float64)
    err = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
    return 2.0 * m * n * k / dt / 1e9, dt, err


ctx_ref = [None]
dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([dev])
ctx_ref[0] = ctx
q = cl.CommandQueue(ctx)
print('device:', dev.name, '|', dev.max_compute_units, 'CU')

rng = np.random.default_rng(0)
# 训练里的前向形状（已验证 769 的那个量级）
M, K, N = 2048, 2048, 512
print('\n=== ① 前向 C = A·B   A[%d,%d] × B[%d,%d] ===' % (M, K, K, N))
A = rng.random((M, K), dtype=np.float32)
B = rng.random((K, N), dtype=np.float32)
g, dt, e = bench(q, build(ctx, K, N, 'fwd'), A, B)
print('   %8.1f GFLOPS  %.3f s  误差 %.1e' % (g, dt, e))

print('\n=== ② wgrad: C = Aᵀ·B（A 按列读）A[%d,%d] → Aᵀ[%d,%d] ===' % (K, M, M, K))
At = np.asfortranarray(rng.random((K, M), dtype=np.float32))   # 转置视图
g, dt, e = bench(q, build(ctx, M, N, 'wgrad'), At, B)
print('   %8.1f GFLOPS  %.3f s  误差 %.1e' % (g, dt, e))

print('\n=== ③ dgrad: C = A·Bᵀ（B 按列读）B[%d,%d] → Bᵀ ===' % (N, K))
Bt = np.asfortranarray(rng.random((N, K), dtype=np.float32))
A2 = rng.random((M, N), dtype=np.float32)
g, dt, e = bench(q, build(ctx, N, K, 'dgrad'), A2, Bt)
print('   %8.1f GFLOPS  %.3f s  误差 %.1e' % (g, dt, e))

print('\n判读：转置形状若也在 600~700 ⇒ 反向内核值得写；若掉到 200 附近 ⇒ 不值得。')
