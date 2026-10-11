"""让核显测量可复现：不去读频率，而是同一次运行里测一个【频率参照物】。

问题：同一个形状、同一个内核，今天量到过 45 / 113 / 332 / 348 / 64 —— 差 5 倍。
15 W APU 上核显频率随 CPU 负载/温度/功耗预算一起动，而 WMI 读不到 GPU 频率。

解法（与今天反复用的同一条纪律一致：**比值要在同一次运行内取**）：
  每一轮里连续跑两件事：
     ① 零访存 FMA 循环  —— FMA 单元能被喂到 ~97%（已验证），所以它的速率
        直接反映【当前频率】，当参照物用；
     ② 待测的 GEMM。
  于是：
     · 绝对 GFLOPS 若随频率飘 —— 正常
     · **GEMM / FMA 的比值**若稳定 —— 那才是这颗 GPU 的真实效率，可以拿来决策
  同时记录 CPU 占用与电源状态，因为 15 W 预算是 CPU 和核显共享的。

判据：
  · 比值稳定（变异系数 < 15%）⇒ 有了可复现的数，可以据此决定要不要写反向内核
  · 比值也在飘 ⇒ 参照物选错了（可能不是频率，而是别的瓶颈在动），需要换方法
"""
import os
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

try:
    import psutil
except ImportError:
    psutil = None

HERE = os.path.dirname(os.path.abspath(__file__))

FMA_SRC = r'''
__kernel void fma32(__global float* out, const int iters) {
    float a0=1.0f,a1=1.01f,a2=1.02f,a3=1.03f,a4=1.04f,a5=1.05f,a6=1.06f,a7=1.07f;
    float a8=1.08f,a9=1.09f,a10=1.1f,a11=1.11f,a12=1.12f,a13=1.13f,a14=1.14f,a15=1.15f;
    const float b = 0.99999f, c = 1e-7f;
    for (int i = 0; i < iters; i++) {
        a0=fma(a0,b,c);  a1=fma(a1,b,c);  a2=fma(a2,b,c);  a3=fma(a3,b,c);
        a4=fma(a4,b,c);  a5=fma(a5,b,c);  a6=fma(a6,b,c);  a7=fma(a7,b,c);
        a8=fma(a8,b,c);  a9=fma(a9,b,c);  a10=fma(a10,b,c); a11=fma(a11,b,c);
        a12=fma(a12,b,c); a13=fma(a13,b,c); a14=fma(a14,b,c); a15=fma(a15,b,c);
    }
    out[get_global_id(0)] = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
}
'''

GEMM_SRC = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void sgemm(const int M, const int N, const int K,
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
    const int AROW = TK/4, BCOL = TS/4;
    const int ATOT = TS*AROW, BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = 0; p < (ATOT+BTOT+NTH_-1)/NTH_; p++) {
            int f = p*NTH_ + lane;
            if (f < ATOT) {
                int row = f/AROW, kq = f%AROW;
                float4 v = *(const __global float4*)(A+(by+row)*KC+k0+kq*4);
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                float4 v = *(const __global float4*)(B+(k0+kk)*NC+bx+cq*4);
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
    for (int i=0;i<TM;i++) { int rc=by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) { int cc=bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc]=acc[i][j]; } }
}
'''


def gpu_state():
    """电源 + CPU 占用。15 W 预算是 CPU 与核显共享的，所以这两项会直接影响核显频率。"""
    ac = '?'
    try:
        import subprocess
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            '(Get-CimInstance Win32_Battery).BatteryStatus'],
                           capture_output=True, text=True, timeout=15)
        v = r.stdout.strip()
        ac = {'1': '放电(电池)', '2': '交流电'}.get(v, v or '?')
    except Exception:
        pass
    cpu = psutil.cpu_percent(interval=0.3) if psutil else -1
    return ac, cpu


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print('device:', dev.name, '|', dev.max_compute_units, 'CU')

    fma = cl.Program(ctx, FMA_SRC).build(options=['-cl-fast-relaxed-math'])
    fma_k = cl.Kernel(fma, 'fma32')
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * 65536)
    NTH, ITERS = 65536, 20000

    M = N = K = 1024
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    fl = 2.0 * M * N * K
    mf = cl.mem_flags
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    gemm = cl.Program(ctx, GEMM_SRC).build(
        options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                 f'-DKC={K}', f'-DNC={N}'])
    gemm_k = cl.Kernel(gemm, 'sgemm')
    gargs = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cb)

    # 预热
    fma_k(q, (NTH,), None, out, np.int32(ITERS)); q.finish()
    gemm_k(q, gws, (8, 8), *gargs); q.finish()

    ac, cpu = gpu_state()
    print(f'电源: {ac} | 测前 CPU 占用 {cpu:.0f}%')
    print(f'\n{"轮":>3} {"FMA G-FMA/s":>13} {"GEMM GFLOPS":>13} {"比值 GFLOPS/(G-FMA/s)":>22} {"CPU%":>6}')
    rows = []
    for r in range(10):
        t0 = time.perf_counter()
        fma_k(q, (NTH,), None, out, np.int32(ITERS)); q.finish()
        fma_rate = 16.0 * NTH * ITERS / (time.perf_counter() - t0) / 1e9
        t0 = time.perf_counter()
        for _ in range(2):
            gemm_k(q, gws, (8, 8), *gargs)
        q.finish()
        dt = (time.perf_counter() - t0) / 2
        g = fl / dt / 1e9
        ratio = g / fma_rate          # 与频率无关的效率
        c = psutil.cpu_percent(interval=None) if psutil else -1
        rows.append((fma_rate, g, ratio, c))
        print(f'{r+1:>3} {fma_rate:>13.1f} {g:>13.1f} {ratio:>22.3f} {c:>6.0f}')

    print('\n=== 分布 ===')
    for idx, name, unit in ((0, 'FMA 速率（频率参照）', 'G-FMA/s'),
                            (1, 'GEMM 绝对吞吐', 'GFLOPS'),
                            (2, '**GEMM/FMA 比值**', '—')):
        v = [r[idx] for r in rows]
        cv = st.pstdev(v) / st.mean(v) * 100 if st.mean(v) else 0
        print('  %-22s min %8.2f  中位 %8.2f  max %8.2f  变异系数 %5.1f%%  (%s)'
              % (name, min(v), st.median(v), max(v), cv, unit))
    v = [r[2] for r in rows]
    cv = st.pstdev(v) / st.mean(v) * 100
    print('\n判读：比值变异系数 %.1f%% %s' % (
        cv, '⇒ 比值稳定，可以据此决策 ✓' if cv < 15 else
        '⇒ 比值也在飘，参照物选错（不是频率在动）✗ 需要换方法'))


if __name__ == '__main__':
    main()
