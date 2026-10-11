"""机制验证：性能是否随【常驻 wave 数】单调变化？

7.5 节的解释是"占用率，不是数据路径"。要证伪它，就得看中间点：
    64 线程（8×8 线程, 8×8 tile）  LDS 4352 B ⇒  7 工作组 × 1 wave =  7 wave
   128 线程（16×8 线程, 8×4 tile）  LDS 4352 B ⇒  7 工作组 × 2 wave = 14 wave
   256 线程（16×16 线程, 4×4 tile） LDS 4352 B ⇒  7 工作组 × 4 wave = 28 wave
三者的 LDS 完全相同（tile 不变），唯一变化的是每工作组的线程数 ⇒ wave 数。
若性能 ≈ 随 wave 单调上升 ⇒ 机制成立；若 128 线程反而更差 ⇒ 另有原因（如指令密度）。

加载映射（都经逐格探针验证）：
  128 线程：A: t→m=t%64, kq=t/64(0..1)，float4（连续 4 个 k）→ As[kq*4+i][m]
            B: t→k=t/16(0..7), nq=t%16，float4（连续 4 个 n）→ Bs[k][nq*4+j]
  256 线程：A: t→m=t%64, kp=t/64(0..3)，float2 → As[kp*2+i][m]
            B: t→k=t/32(0..7), np=t%32，float2 → Bs[k][np*2+j]
"""
import os
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '_waves')

COMMON = r'''
#define TS 64
#define TK 8
'''

# ---- 64 线程 / 8x8（已知 313~370 GFLOPS）----
K64 = COMMON + r'''
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void v64(const int M, const int N, const int K,
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

# ---- 128 线程 / 8x4（14 wave）----
K128 = COMMON + r'''
#define TM 8
#define TN 4
#define NTH_ 128
__kernel void v128(const int M, const int N, const int K,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);   // tx 0..15, ty 0..7
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;                             // 0..127
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int am = t % 64, akq = t / 64;                    // A: m, kquad(0..1)
    const int bk = t / 16, bnq = t % 16;                    // B: k(0..7), nquad(0..15)
    for (int k0 = 0; k0 < K; k0 += TK) {
        float4 va = *(const __global float4*)(A + (by+am)*KC + k0 + akq*4);
        As[akq*4+0][am]=va.x; As[akq*4+1][am]=va.y;
        As[akq*4+2][am]=va.z; As[akq*4+3][am]=va.w;
        float4 vb = *(const __global float4*)(B + (k0+bk)*NC + bx + bnq*4);
        Bs[bk][bnq*4+0]=vb.x; Bs[bk][bnq*4+1]=vb.y;
        Bs[bk][bnq*4+2]=vb.z; Bs[bk][bnq*4+3]=vb.w;
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

# ---- 256 线程 / 4x4（28 wave，已知 691~769）----
K256 = COMMON + r'''
#define TM 4
#define TN 4
#define NTH_ 256
__kernel void v256(const int M, const int N, const int K,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);   // 0..15
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int t = ty * 16 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int am = t % 64, akp = t / 64;                    // A: m, kpair(0..3)
    const int bk = t / 32, bnp = t % 32;                    // B: k(0..7), npair(0..31)
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


def probe_src(name, nthreads, tdim):
    """为每个变体生成【同一套映射】的探针，逐格验证。"""
    if nthreads == 64:
        body = r'''
    const int lane = ty*8 + tx;
    const int AROW = TK/4, BCOL = TS/4;
    for (int p = 0; p < 4; p++) {
        int f = p*64 + lane;
        if (f < TS*AROW) {
            int row = f/AROW, kq = f%AROW;
            float4 v = *(const __global float4*)(A+(by+row)*KC+kq*4);
            As[kq*4+0][row]=v.x; As[kq*4+1][row]=v.y; As[kq*4+2][row]=v.z; As[kq*4+3][row]=v.w;
        } else if (f < TS*AROW + TK*BCOL) {
            int g = f-TS*AROW, kk = g/BCOL, cq = g%BCOL;
            float4 v = *(const __global float4*)(B+kk*NC+bx+cq*4);
            Bs[kk][cq*4+0]=v.x; Bs[kk][cq*4+1]=v.y; Bs[kk][cq*4+2]=v.z; Bs[kk][cq*4+3]=v.w;
        }
    }'''
    elif nthreads == 128:
        body = r'''
    const int t = ty*16 + tx;
    const int am = t % 64, akq = t / 64;
    const int bk = t / 16, bnq = t % 16;
    float4 va = *(const __global float4*)(A+(by+am)*KC + akq*4);
    As[akq*4+0][am]=va.x; As[akq*4+1][am]=va.y; As[akq*4+2][am]=va.z; As[akq*4+3][am]=va.w;
    float4 vb = *(const __global float4*)(B+bk*NC+bx+bnq*4);
    Bs[bk][bnq*4+0]=vb.x; Bs[bk][bnq*4+1]=vb.y; Bs[bk][bnq*4+2]=vb.z; Bs[bk][bnq*4+3]=vb.w;'''
    else:
        body = r'''
    const int t = ty*16 + tx;
    const int am = t % 64, akp = t / 64;
    const int bk = t / 32, bnp = t % 32;
    float2 va = *(const __global float2*)(A+(by+am)*KC + akp*2);
    As[akp*2+0][am]=va.x; As[akp*2+1][am]=va.y;
    float2 vb = *(const __global float2*)(B+bk*NC+bx+bnp*2);
    Bs[bk][bnp*2+0]=vb.x; Bs[bk][bnp*2+1]=vb.y;'''
    return COMMON + f'''
#define NTH_ {nthreads}
__kernel void {name}(const int M, const int N, const int K,
                     const __global float* restrict A,
                     const __global float* restrict B,
                     __global float* dA, __global float* dB) {{
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
{body}
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = get_local_id(1)*{tdim[0]} + get_local_id(0); i < 512; i += NTH_) {{
        dA[i] = As[i/TS][i%TS];
        dB[i] = Bs[i/TS][i%TS];
    }}
}}
'''


VARIANTS = [
    ('v64', K64, (8, 8), 64, (8, 8)),
    ('v128', K128, (16, 8), 128, (16, 8)),
    ('v256', K256, (16, 16), 256, (16, 16)),
]


def build(ctx, tag, src, K, N):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        return cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable', f'-DKC={K}', f'-DNC={N}'])
    finally:
        os.chdir(cwd)


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
    expA = np.stack([a[0:64, k] for k in range(8)])
    expB = np.stack([b[k, 0:64] for k in range(8)])

    print('=== 1) 三个变体的加载探针（逐格）===')
    dA = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    dB = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    for name, _src, ws, nth, tdim in VARIANTS:
        prog = build(ctx, f'probe_{name}', probe_src(f'probe_{name}', nth, tdim), K, N)
        cl.Kernel(prog, f'probe_{name}')(q, ws, ws, np.int32(M), np.int32(N),
                                         np.int32(K), ab[0], ab[1], dA, dB)
        q.finish()
        oA = np.empty(512, np.float32)
        oB = np.empty(512, np.float32)
        cl.enqueue_copy(q, oA, dA).wait()
        cl.enqueue_copy(q, oB, dB).wait()
        bad = int((oA.reshape(8, 64) != expA).sum() + (oB.reshape(8, 64) != expB).sum())
        print(f'   {name:<6} ({nth:>3} 线程) 错格 {bad:>4}  '
              f'{"✅" if bad == 0 else "❌ 不进入计时"}')
        if bad:
            return 1

    print('\n=== 2) 全量正确性 ===')
    progs = {}
    for name, src, ws, nth, _t in VARIANTS:
        progs[name] = build(ctx, name, src, K, N)
    cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    out = np.empty((M, N), np.float32)
    ref = a @ b
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    for name, _s, ws, _n, _t in VARIANTS:
        cl.Kernel(progs[name], name)(q, gws, ws, np.int32(M), np.int32(N),
                                     np.int32(K), ab[0], ab[1], cbuf)
        q.finish()
        cl.enqueue_copy(q, out, cbuf).wait()
        e = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'   {name:<6} 相对误差 {e:.2e}  {"OK" if e < 1e-4 else "❌"}')
        if e >= 1e-4:
            return 1

    print('\n=== 3) 计时：性能是否随 wave 数单调？===')
    print(f"{'形状':>20} {'7 wave':>9} {'14 wave':>9} {'28 wave':>9}   {'7→14':>7} {'14→28':>7}")
    for M2, N2, K2 in ((512, 512, 512), (1024, 1024, 1024), (2048, 512, 512),
                       (2048, 2048, 512), (2048, 512, 2048)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        ps = {}
        for name, src, ws, nth, _t in VARIANTS:
            ps[name] = progs[name] if (K2 == K and N2 == N) else build(ctx, name, src, K2, N2)
        ab2 = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2),
               cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b2))
        cb2 = cl.Buffer(ctx, mf.WRITE_ONLY, M2 * N2 * 4)
        g2 = (int(np.ceil(N2 / 64)) * 8, int(np.ceil(M2 / 64)) * 8)
        r = {}
        for name, _s, ws, _n, _t in VARIANTS:
            kk = cl.Kernel(ps[name], name)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), ab2[0], ab2[1], cb2)
            kk(q, g2, ws, *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(5):
                kk(q, g2, ws, *args)
            q.finish()
            r[name] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        print(f'{M2:>8}×{N2:<5}×{K2:<4} {r["v64"]:>9.1f} {r["v128"]:>9.1f} '
              f'{r["v256"]:>9.1f}   {r["v128"]/r["v64"]:>6.2f}× '
              f'{r["v256"]/r["v128"]:>6.2f}×', flush=True)


if __name__ == '__main__':
    sys.exit(main())
