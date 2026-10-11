"""v3db：双缓冲预取，但这次【加载映射用宏只写一次】——不给手抄留机会。

教训：手写索引映射翻车三次（DB NaN、v1 NaN、DB 又 NaN），而 v0/v2 的映射逐格验过。
所以这次把"取数→寄存器"和"寄存器→LDS"写成**同一份宏**，两个缓冲只是传不同的数组名，
映射在物理上不可能不一致。K 循环展开 2 次 ⇒ 两个静态命名缓冲，没有动态索引。

流程（规矩已定）：探针逐格 → 全量误差 → 才计时。
"""
import collections
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
BASE = os.path.join(HERE, '_v3db')

HEAD = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
#define NLOAD 4
'''

MACROS = r'''
// —— 索引映射只写一次：取到寄存器 与 存进 LDS 共用同一份表达式 ——
#define IDX_INIT \
    const int AROW = TK/4, BCOL = TS/4; \
    const int ATOT = TS*AROW, BTOT = TK*BCOL;

#define PREFETCH(K0) { \
    _Pragma("unroll") \
    for (int p = 0; p < NLOAD; p++) { \
        int f = p*NTH_ + lane; \
        if (f < ATOT) { \
            int row = f/AROW, kq = f%AROW; \
            int gr = by+row, gc = (K0)+kq*4; \
            P[p] = *(const __global float4*)(A+gr*KC+gc); \
        } else { \
            int g = f-ATOT, kk = g/BCOL, cq = g%BCOL; \
            int gr = (K0)+kk, gc = bx+cq*4; \
            P[p] = *(const __global float4*)(B+gr*NC+gc); \
        } \
    } \
}

#define COMMIT(AX, BX) { \
    _Pragma("unroll") \
    for (int p = 0; p < NLOAD; p++) { \
        int f = p*NTH_ + lane; \
        if (f < ATOT) { \
            int row = f/AROW, kq = f%AROW; \
            AX[kq*4+0][row]=P[p].x; AX[kq*4+1][row]=P[p].y; \
            AX[kq*4+2][row]=P[p].z; AX[kq*4+3][row]=P[p].w; \
        } else { \
            int g = f-ATOT, kk = g/BCOL, cq = g%BCOL; \
            BX[kk][cq*4+0]=P[p].x; BX[kk][cq*4+1]=P[p].y; \
            BX[kk][cq*4+2]=P[p].z; BX[kk][cq*4+3]=P[p].w; \
        } \
    } \
}
'''

COMPUTE = r'''
#define COMPUTE_BUF(AX, BX) { \
    for (int k = 0; k < TK; k++) { \
        float a[TM], b[TN]; \
        for (int i=0;i<TM;i++) a[i]=AX[k][ty*TM+i]; \
        for (int j=0;j<TN;j++) b[j]=BX[k][tx*TN+j]; \
        for (int i=0;i<TM;i++) \
            for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]); \
    } \
}
'''

TAIL = r'''
    for (int i=0;i<TM;i++) {
        int rc = by+ty*TM+i; if (rc>=M) continue;
        for (int j=0;j<TN;j++) {
            int cc = bx+tx*TN+j; if (cc>=N) continue;
            C[rc*N+cc] = acc[i][j];
        }
    }
}
'''

DECL = r'''
    __local float As0[TK][TS + 4];
    __local float Bs0[TK][TS + 4];
    __local float As1[TK][TS + 4];
    __local float Bs1[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    float4 P[NLOAD];
    IDX_INIT
'''

SIG = ('__kernel void {name}(const int M,const int N,const int K,'
       'const __global float* restrict A,const __global float* restrict B,'
       '__global float* restrict C){')

# v2：单缓冲（已验证，作为基线）
V2 = HEAD + MACROS + COMPUTE + SIG.replace('{name}', 'v2') + DECL + r'''
    for (int k0 = 0; k0 < K; k0 += TK) {
        PREFETCH(k0)
        COMMIT(As0, Bs0)
        barrier(CLK_LOCAL_MEM_FENCE);
        COMPUTE_BUF(As0, Bs0)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
''' + TAIL + MACROS + COMPUTE

# v3db：双缓冲，k 循环展开 2 次 ⇒ 两个静态命名缓冲
V3DB = HEAD + MACROS + COMPUTE + SIG.replace('{name}', 'v3db') + DECL + r'''
    PREFETCH(0)
    COMMIT(As0, Bs0)
    barrier(CLK_LOCAL_MEM_FENCE);
    int k0 = 0;
    for (; k0 + 2*TK <= K; k0 += 2*TK) {
        PREFETCH(k0+TK)          // 恒有效：k0+TK <= K
        COMPUTE_BUF(As0, Bs0)    // 算第 k0 块（这 512 条 FMA 盖住上面的加载延迟）
        COMMIT(As1, Bs1)
        barrier(CLK_LOCAL_MEM_FENCE);
        // 最后一次迭代 k0+2*TK == K ⇒ 这次预取会越界读到 K 之外，必须挡掉。
        if (k0 + 2*TK < K) PREFETCH(k0+2*TK)
        COMPUTE_BUF(As1, Bs1)    // 算第 k0+TK 块
        COMMIT(As0, Bs0)
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // ⚠️ 这里【不能】再补算一块：K 是 2*TK 的整数倍时，[0,K) 的块在循环内已全部算完。
    // 第一版就是在这里多算了一个不存在的块（读越界 + 算垃圾），误差 2.67e+05 —— 被探针闸门拦下。
''' + TAIL + MACROS + COMPUTE

# 探针：只跑序言的加载，dump LDS（验证宏的映射）
PROBE = HEAD + MACROS + r'''
__kernel void probe(const int M,const int N,const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* dA, __global float* dB){
''' + DECL.replace('float4 P[NLOAD];', 'float4 P[NLOAD];') + r'''
    PREFETCH(0)
    COMMIT(As0, Bs0)
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lane; i < TS*TK; i += 64) {
        int k = i/TS, r = i%TS;
        dA[i] = As0[k][r]; dB[i] = Bs0[k][r];
    }
}
''' + MACROS


def build(ctx, tag, src, K, N, savetemps=False):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        o = ['-cl-fast-relaxed-math', '-cl-mad-enable', f'-DKC={K}', f'-DNC={N}']
        if savetemps:
            o.append('-save-temps')
        return cl.Program(ctx, src).build(options=o)
    finally:
        os.chdir(cwd)


def isa_stats(tag):
    for f in glob.glob(os.path.join(BASE, tag, '*.isa')):
        seg = open(f, encoding='utf-8', errors='replace').read().split('Disassembly:')
        if len(seg) < 2:
            continue
        ops = re.findall(r'^\s+([a-z][a-z0-9_]*)\s.*//\s*[0-9a-f]+:', seg[1], re.M)
        if not ops:
            continue
        h = collections.Counter(ops)
        n = len(ops)
        fma = sum(v for k, v in h.items() if k.startswith(('v_fma', 'v_mac')))
        br = sum(v for k, v in h.items() if k.startswith(('s_cbranch', 's_andn2')))
        wait = sum(v for k, v in h.items() if k.startswith('s_waitcnt'))
        return n, fma, br, wait
    return None


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

    print('=== 1) 加载探针（宏的映射，逐格）===')
    dA = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    dB = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    oA = np.full(512, np.nan, np.float32)
    oB = np.full(512, np.nan, np.float32)
    expA = np.stack([a[0:64, k] for k in range(8)])
    expB = np.stack([b[k, 0:64] for k in range(8)])
    pp = build(ctx, 'probe', PROBE, K, N)
    cl.Kernel(pp, 'probe')(q, (8, 8), (8, 8), np.int32(M), np.int32(N), np.int32(K),
                           ab[0], ab[1], dA, dB)
    q.finish()
    cl.enqueue_copy(q, oA, dA).wait()
    cl.enqueue_copy(q, oB, dB).wait()
    badA = int((oA.reshape(8, 64) != expA).sum())
    badB = int((oB.reshape(8, 64) != expB).sum())
    print(f'   As 错格 {badA}  Bs 错格 {badB}  '
          f'{"✅ 通过" if badA == 0 and badB == 0 else "❌ 毙掉，不进入计时"}')
    if badA or badB:
        return 1

    print('\n=== 2) 全量正确性 ===')
    progs = {}
    for tag, src in (('v2', V2), ('v3db', V3DB)):
        progs[tag] = build(ctx, tag, src, K, N, savetemps=True)
    cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    out = np.empty((M, N), np.float32)
    ref = a @ b
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    for tag in ('v2', 'v3db'):
        cl.Kernel(progs[tag], tag)(q, gws, (8, 8), np.int32(M), np.int32(N),
                                   np.int32(K), ab[0], ab[1], cbuf)
        q.finish()
        cl.enqueue_copy(q, out, cbuf).wait()
        e = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'   {tag}: 相对误差 {e:.2e}  {"OK" if e < 1e-4 else "❌"}')
        if e >= 1e-4:
            print(f'\n{tag} 数值不对，不进入计时。')
            return 1

    print('\n=== 3) ISA ===')
    print(f"{'变体':<6} {'总指令':>7} {'FMA':>6} {'FMA%':>7} {'分支':>6} {'s_waitcnt':>10}")
    for tag in ('v2', 'v3db'):
        r = isa_stats(tag)
        if r:
            n, fma, br, w = r
            print(f'{tag:<6} {n:>7} {fma:>6} {fma/n*100:>6.1f}% {br:>6} {w:>10}')

    print('\n=== 4) 计时 ===')
    for M2, N2, K2 in ((512, 512, 512), (2048, 512, 512), (2048, 2048, 512)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        p2 = progs['v2'] if (K2 == K and N2 == N) else build(ctx, 'v2', V2, K2, N2)
        p3 = progs['v3db'] if (K2 == K and N2 == N) else build(ctx, 'v3db', V3DB, K2, N2)
        ab2 = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2),
               cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b2))
        cb2 = cl.Buffer(ctx, mf.WRITE_ONLY, M2 * N2 * 4)
        g2 = (int(np.ceil(N2 / 64)) * 8, int(np.ceil(M2 / 64)) * 8)
        res = {}
        for tag, prog in (('v2', p2), ('v3db', p3)):
            kk = cl.Kernel(prog, tag)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), ab2[0], ab2[1], cb2)
            kk(q, g2, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(5):
                kk(q, g2, (8, 8), *args)
            q.finish()
            res[tag] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        print(f'   {M2}×{N2}×{K2}:  v2 {res["v2"]:.1f}  →  v3db {res["v3db"]:.1f}  '
              f'({res["v3db"]/res["v2"]:.2f}×)', flush=True)


if __name__ == '__main__':
    sys.exit(main())
