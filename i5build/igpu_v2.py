"""v2 = v0 的加载段【原样照抄】+ 只做两处最小改动：K 特化、去掉守卫。

为什么是"照抄"而不是"重写"：手写索引映射已经翻车三次（双缓冲 NaN、v1 NaN、
双缓冲再 NaN），而 v0 的映射刚才在 loader_probe 里【逐格 512/512 验证通过】。
所以 v2 的加载段一个字符都不改，只动两件事：

  · `A+gr*K` → `A+gr*KC`（KC 是编译期常量 ⇒ (by+row)*K 折叠成移位/立即数）
  · 删掉 `if (gr<M && gc+3<K)` / `if (gr<K && gc+3<N)` 两个守卫
    （我们的 M/N/K 全是 64/8 的整数倍；ISA 里这两条对应 32 条 s_cbranch_execz
      + 28 条 s_andn2_b64 的 exec 掩码操作）

流程（这次定了规矩）：
  1) 先过加载探针（逐格比对，错一格就毙）；
  2) 再比全量 GEMM 输出（err < 1e-4）；
  3) 最后才计时 + 数 ISA。
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
BASE = os.path.join(HERE, '_v2')

HEAD = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
'''

# 加载段与 v0 逐字相同，只有 KC/NC 和守卫的差别
LOAD_V0_GUARDED = r'''
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
'''

LOAD_V2 = r'''
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
'''

COMPUTE = r'''
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

PRE = r'''
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
'''

GEMM_V0 = HEAD + '__kernel void v0(const int M,const int N,const int K,const __global float* restrict A,const __global float* restrict B,__global float* restrict C){' + PRE + LOAD_V0_GUARDED + COMPUTE + TAIL
GEMM_V2 = HEAD + '__kernel void v2(const int M,const int N,const int K,const __global float* restrict A,const __global float* restrict B,__global float* restrict C){' + PRE + LOAD_V2 + COMPUTE + TAIL

# 加载探针：只跑一个 k-tile 的加载，dump LDS
PROBE_V0 = HEAD + '__kernel void pv0(const int M,const int N,const int K,const __global float* restrict A,const __global float* restrict B,__global float* dA,__global float* dB){' + PRE.replace('for (int k0 = 0; k0 < K; k0 += TK) {', 'const int k0 = 0;') + LOAD_V0_GUARDED + r'''
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lane; i < TS*TK; i += 64) { int k = i/TS, r = i%TS;
        dA[i] = As[k][r]; dB[i] = Bs[k][r]; }
}
'''
PROBE_V2 = HEAD + '__kernel void pv2(const int M,const int N,const int K,const __global float* restrict A,const __global float* restrict B,__global float* dA,__global float* dB){' + PRE.replace('for (int k0 = 0; k0 < K; k0 += TK) {', 'const int k0 = 0;') + LOAD_V2 + r'''
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = lane; i < TS*TK; i += 64) { int k = i/TS, r = i%TS;
        dA[i] = As[k][r]; dB[i] = Bs[k][r]; }
}
'''


def build(ctx, name, src, K, N, savetemps=False):
    d = os.path.join(BASE, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        opts = ['-cl-fast-relaxed-math', '-cl-mad-enable',
                f'-DKC={K}', f'-DNC={N}']
        if savetemps:
            opts.append('-save-temps')
        return cl.Program(ctx, src).build(options=opts)
    finally:
        os.chdir(cwd)


def isa_stats(name):
    for f in glob.glob(os.path.join(BASE, name, '*.isa')):
        seg = open(f, encoding='utf-8', errors='replace').read().split('Disassembly:')
        if len(seg) < 2:
            continue
        ops = re.findall(r'^\s+([a-z][a-z0-9_]*)\s.*//\s*[0-9a-f]+:', seg[1], re.M)
        if not ops:
            continue
        h = collections.Counter(ops)
        n = len(ops)
        fma = sum(v for k, v in h.items() if k.startswith(('v_fma', 'v_mac')))
        ints = sum(v for k, v in h.items()
                   if k.startswith(('v_add', 'v_mul', 'v_lshl', 'v_lshr', 'v_ashr',
                                    'v_and', 'v_or', 'v_xor', 'v_cndmask', 'v_mov')))
        br = sum(v for k, v in h.items() if k.startswith(('s_cbranch', 's_andn2')))
        return n, fma, ints, br
    return None


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    K = N = M = 512

    a = np.arange(M * K, dtype=np.float32).reshape(M, K) % 97 + 0.5
    b = np.arange(K * N, dtype=np.float32).reshape(K, N) % 89 + 0.25
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))

    # ---- 1) 加载探针 ----
    print('=== 1) 加载探针（逐格）===')
    dA = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    dB = cl.Buffer(ctx, mf.WRITE_ONLY, 512 * 4)
    oA = np.full(512, np.nan, np.float32)
    oB = np.full(512, np.nan, np.float32)
    expA = np.stack([a[0:64, k] for k in range(8)])
    expB = np.stack([b[k, 0:64] for k in range(8)])
    gate = {}
    for tag, src, kn in (('v0', PROBE_V0, 'pv0'), ('v2', PROBE_V2, 'pv2')):
        prog = build(ctx, f'probe_{tag}', src, K, N)
        cl.Kernel(prog, kn)(q, (8, 8), (8, 8), np.int32(M), np.int32(N), np.int32(K),
                            ab[0], ab[1], dA, dB)
        q.finish()
        cl.enqueue_copy(q, oA, dA).wait()
        cl.enqueue_copy(q, oB, dB).wait()
        badA = int((oA.reshape(8, 64) != expA).sum())
        badB = int((oB.reshape(8, 64) != expB).sum())
        gate[tag] = (badA == 0 and badB == 0)
        print(f'   {tag}: As 错格 {badA:>4}  Bs 错格 {badB:>4}  '
              f'{"✅ 通过" if gate[tag] else "❌ 毙掉"}')
    if not gate['v2']:
        print('\nv2 的加载段没过探针，按规矩不进入计时。')
        return 1

    # ---- 2) 全量正确性 ----
    print('\n=== 2) 全量 GEMM 正确性 ===')
    prog0 = build(ctx, 'v0', GEMM_V0, K, N, savetemps=True)
    prog2 = build(ctx, 'v2', GEMM_V2, K, N, savetemps=True)
    cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    out = np.empty((M, N), np.float32)
    ref = a @ b
    errs = {}
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    for tag, prog in (('v0', prog0), ('v2', prog2)):
        cl.Kernel(prog, tag)(q, gws, (8, 8), np.int32(M), np.int32(N), np.int32(K),
                             ab[0], ab[1], cbuf)
        q.finish()
        cl.enqueue_copy(q, out, cbuf).wait()
        errs[tag] = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        print(f'   {tag}: 相对误差 {errs[tag]:.2e}  {"OK" if errs[tag] < 1e-4 else "❌"}')
    if errs['v2'] >= 1e-4:
        print('\nv2 数值不对，不进入计时。')
        return 1

    # ---- 3) ISA + 计时 ----
    print('\n=== 3) ISA 构成 ===')
    print(f"{'变体':<5} {'总指令':>7} {'FMA':>6} {'FMA%':>7} {'整数/搬运':>10} "
          f"{'占比':>7} {'分支':>6}")
    for tag in ('v0', 'v2'):
        r = isa_stats(tag)
        if r:
            n, fma, ints, br = r
            print(f'{tag:<5} {n:>7} {fma:>6} {fma/n*100:>6.1f}% {ints:>10} '
                  f'{ints/n*100:>6.1f}% {br:>6}')

    print('\n=== 4) 计时（K=512 特化命中）===')
    for M2, N2, K2 in ((512, 512, 512), (2048, 512, 512)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        if K2 != K or N2 != N:
            p0 = build(ctx, 'v0', GEMM_V0, K2, N2)
            p2 = build(ctx, 'v2', GEMM_V2, K2, N2)
        else:
            p0, p2 = prog0, prog2
        ab2 = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2),
               cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b2))
        cb2 = cl.Buffer(ctx, mf.WRITE_ONLY, M2 * N2 * 4)
        gws2 = (int(np.ceil(N2 / 64)) * 8, int(np.ceil(M2 / 64)) * 8)
        res = {}
        for tag, prog in (('v0', p0), ('v2', p2)):
            k = cl.Kernel(prog, tag)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), ab2[0], ab2[1], cb2)
            k(q, gws2, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(5):
                k(q, gws2, (8, 8), *args)
            q.finish()
            res[tag] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        print(f'   {M2}×{N2}×{K2}:  v0 {res["v0"]:.1f}  →  v2 {res["v2"]:.1f}  '
              f'({res["v2"]/res["v0"]:.2f}×)', flush=True)


if __name__ == '__main__':
    sys.exit(main())
