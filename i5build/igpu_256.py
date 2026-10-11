"""256 线程 / TM=TN=4：核显线上最后一个未测的设计。

动机（前几轮实测推出来的）：
  · GEMM 只到实测 FMA 上限的 33~39%
  · 而同形零访存内循环能到 97.2%  ⇒ 缺的全是"喂数据"
  · 已否掉：float4 读 LDS、A 出 LDS、packed fp16、K 特化去守卫、双缓冲预取
    （双缓冲失败的原因：为了藏延迟要多发 50% 的指令，而指令正是稀缺品）

这个变体不同：**它不增加任何指令，只改线程的组织方式。**
  · LDS 占用只由【tile】决定，与线程数无关
      —— tile 仍是 TS=64 × TK=8 ⇒ A/B 各 8×68×4 = 2176 B，合计 4352 B（和 64 线程版一样）
      ⇒ 每 CU 仍常驻 7 个工作组
  · 但工作组从 64 线程（1 个 wave）变成 256 线程（4 个 wave）
      ⇒ **每 CU 常驻 wave 从 7 变成 28**，隐藏 LDS 延迟的能力 4×
  · 代价：每线程 tile 从 8×8 缩到 4×4 ⇒ 每条 FMA 配的 LDS 读从 16:64(4:1) 变 8:16(2:1)
      ⇒ 指令密度从 80% 降到 67%，发射上限降到 322 G-FMA/s（= 644 GFLOPS）
      ⇒ 但我们实测只有 157~185，离两个上限都远，"能藏住延迟"优先于"更高密度"

流程照旧：探针逐格 → 全量正确性 → 才计时。
"""
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '_w256')

# ---------------- 64 线程版（基线，已验证） ----------------
V2 = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void v2(const int M, const int N, const int K,
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
                int gr = by+row, gc = k0+kq*4;
                float4 v = *(const __global float4*)(A+gr*KC+gc);
                As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
            } else if (f < ATOT+BTOT) {
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*4;
                float4 v = *(const __global float4*)(B+gr*NC+gc);
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

# ---------------- 256 线程 / 4x4（新） ----------------
# 加载映射（256 线程，每个线程 2 个 float2）：
#   A: 线程 t -> m = t%64, kpair = t/64 (0..3)  ⇒ A[(by+m)*K + k0 + kpair*2 .. +1]
#      存 As[kpair*2][m]、As[kpair*2+1][m]
#   B: 线程 t -> k = t/32 (0..7), npair = t%32 ⇒ B[(k0+k)*N + bx + npair*2 .. +1]
#      存 Bs[k][npair*2]、Bs[k][npair*2+1]
# 两者都是 t 上的双射：A 覆盖 4 kpair × 64 m = 256 ✓；B 覆盖 8 k × 32 npair = 256 ✓
V4 = r'''
#define TS 64
#define TK 8
#define TM 4
#define TN 4
#define NTH_ 256
__kernel void v4(const int M, const int N, const int K,
                 const __global float* restrict A,
                 const __global float* restrict B,
                 __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);   // 0..15
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;                             // 0..255
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    const int am = t % 64, akp = t / 64;        // A: m, kpair
    const int bk = t / 32, bnp = t % 32;        // B: k, npair

    for (int k0 = 0; k0 < K; k0 += TK) {
        // A 的 float2：同一 m 上两个相邻 k（A 是 [M,K] 行主序，k 连续）
        float2 va = *(const __global float2*)(A + (by+am)*KC + k0 + akp*2);
        As[akp*2+0][am] = va.x;
        As[akp*2+1][am] = va.y;
        // B 的 float2：同一个 k 上两个相邻 n（B 是 [K,N] 行主序，n 连续）
        float2 vb = *(const __global float2*)(B + (k0+bk)*NC + bx + bnp*2);
        Bs[bk][bnp*2+0] = vb.x;
        Bs[bk][bnp*2+1] = vb.y;
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

# ---------------- 探针：只跑一个 k-tile 的加载并 dump LDS ----------------
PROBE = r'''
#define TS 64
#define TK 8
__kernel void probe(const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* dA, __global float* dB) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    const int k0 = 0;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    float2 va = *(const __global float2*)(A + (by+am)*KC + k0 + akp*2);
    As[akp*2+0][am] = va.x;
    As[akp*2+1][am] = va.y;
    float2 vb = *(const __global float2*)(B + (k0+bk)*NC + bx + bnp*2);
    Bs[bk][bnp*2+0] = vb.x;
    Bs[bk][bnp*2+1] = vb.y;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t < 512) {
        dA[t] = As[t/TS][t%TS];
        dB[t] = Bs[t/TS][t%TS];
    }
}
'''
# ⚠️ 上面那个 dump 是错的：只有 256 个线程，`if (t < 512)` 只能覆盖 k=0..3，
# 于是 k=4..7 全是 0，探针误报"内核错"。改成跨步循环覆盖全部 512 格。
PROBE = PROBE.replace('''    if (t < 512) {
        dA[t] = As[t/TS][t%TS];
        dB[t] = Bs[t/TS][t%TS];
    }''', '''    for (int i = t; i < 512; i += NTH_) {
        dA[i] = As[i/TS][i%TS];
        dB[i] = Bs[i/TS][i%TS];
    }''').replace('#define TS 64\n#define TK 8\n__kernel void probe',
               '#define TS 64\n#define TK 8\n#define NTH_ 256\n__kernel void probe')


def build(ctx, tag, src, K, N, save=False):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        o = ['-cl-fast-relaxed-math', '-cl-mad-enable', f'-DKC={K}', f'-DNC={N}']
        if save:
            o.append('-save-temps')
        return cl.Program(ctx, src).build(options=o)
    finally:
        os.chdir(cwd)


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    print(f'device {dev.name} | {dev.max_compute_units} CU | '
          f'最大工作组 {dev.max_work_group_size} 线程')

    M = N = K = 512
    a = np.arange(M * K, dtype=np.float32).reshape(M, K) % 97 + 0.5
    b = np.arange(K * N, dtype=np.float32).reshape(K, N) % 89 + 0.25
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))

    # ---- 1) 探针 ----
    print('\n=== 1) 加载探针（逐格）===')
    dA = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    dB = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    oA = np.full(512, np.nan, np.float32)
    oB = np.full(512, np.nan, np.float32)
    expA = np.stack([a[0:64, k] for k in range(8)])
    expB = np.stack([b[k, 0:64] for k in range(8)])
    pp = build(ctx, 'probe', PROBE, K, N)
    cl.Kernel(pp, 'probe')(q, (16, 16), (16, 16), np.int32(M), np.int32(N),
                           np.int32(K), ab[0], ab[1], dA, dB)
    q.finish()
    cl.enqueue_copy(q, oA, dA).wait()
    cl.enqueue_copy(q, oB, dB).wait()
    A2 = oA.reshape(8, 64)
    B2 = oB.reshape(8, 64)
    badA = np.argwhere(A2 != expA)
    badB = np.argwhere(B2 != expB)
    print(f'   As 错格 {len(badA)}  Bs 错格 {len(badB)}')
    for tag, bad, exp, got in (('A', badA, expA, A2), ('B', badB, expB, B2)):
        for (kk, rr) in bad[:3]:
            print(f'     {tag}[k={kk}][pos={rr}] 期望 {exp[kk,rr]:.3f} 得到 {got[kk,rr]:.3f}')
    if len(badA) or len(badB):
        print('\n加载映射没过探针，按规矩不进入计时。')
        return 1
    print('   ✅ 两套映射都逐格正确')

    # ---- 2) 正确性 ----
    print('\n=== 2) 全量正确性 ===')
    progs = {'v2': build(ctx, 'v2', V2, K, N), 'v4': build(ctx, 'v4', V4, K, N)}
    cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    out = np.empty((M, N), np.float32)
    ref = a @ b
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    for tag in ('v2', 'v4'):
        ws = (8, 8) if tag == 'v2' else (16, 16)
        cl.Kernel(progs[tag], tag)(q, gws, ws, np.int32(M), np.int32(N),
                                   np.int32(K), ab[0], ab[1], cbuf)
        q.finish()
        cl.enqueue_copy(q, out, cbuf).wait()
        e = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'   {tag}: 相对误差 {e:.2e}  {"OK" if e < 1e-4 else "❌"}')
        if e >= 1e-4:
            return 1

    # ---- 3) 计时 ----
    print('\n=== 3) 计时 ===')
    print(f"{'形状':>20} {'v2 (64线程)':>12} {'v4 (256线程)':>13} {'v4/v2':>8}"
          f"  {'常驻wave':>9}")
    for M2, N2, K2 in ((512, 512, 512), (1024, 1024, 1024),
                       (2048, 512, 512), (2048, 2048, 512)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        p2 = progs['v2'] if (K2 == K and N2 == N) else build(ctx, 'v2', V2, K2, N2)
        p4 = progs['v4'] if (K2 == K and N2 == N) else build(ctx, 'v4', V4, K2, N2)
        ab2 = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2),
               cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b2))
        cb2 = cl.Buffer(ctx, mf.WRITE_ONLY, M2 * N2 * 4)
        g2 = (int(np.ceil(N2 / 64)) * 8, int(np.ceil(M2 / 64)) * 8)
        res = {}
        for tag, prog, ws in (('v2', p2, (8, 8)), ('v4', p4, (16, 16))):
            kk = cl.Kernel(prog, tag)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), ab2[0], ab2[1], cb2)
            kk(q, g2, ws, *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(5):
                kk(q, g2, ws, *args)
            q.finish()
            res[tag] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        print(f'{M2:>8}×{N2:<5}×{K2:<4} {res["v2"]:>12.1f} {res["v4"]:>13.1f} '
              f'{res["v4"]/res["v2"]:>7.2f}×  {"7→28":>9}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
