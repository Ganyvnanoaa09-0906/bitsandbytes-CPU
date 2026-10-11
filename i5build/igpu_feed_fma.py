"""喂满 FMA：把地址算术从热路径赶出去。

ISA 直方图给的账（gemm16，651 条指令）：
    FMA 20.7% | LDS 7.2% |【整数/地址 44.4%】| 标量 9.1% | 分支/等待 9.4%
病因是我自己写的两处：
  ① `A + gr*K + gc` —— K 是内核参数（运行时值）⇒ 每次访问一次运行时乘法 + 64 位扩展
     （ISA 里能看到 v_mul_lo_u32 / v_add_co_u32+v_addc_co_u32 成对 / v_lshlrev_b64）
  ② `if (gr<M && gc+3<K)` —— 每个受保护加载变成分支 + exec 掩码
     （s_cbranch_execz 32 条 + s_andn2_b64 28 条）

本脚本对比：
  v0 = 现状（运行时 K + 守卫）
  v1 = 4 个加载指针提到循环外（每线程一次 64 位基址）+ 去守卫 + K 编译期常量
  v1db = v1 + 双缓冲预取（k 循环展开 2 次、两个【静态命名】缓冲 —— 顺便绕开上次 NaN
         的动态索引嫌疑）

判据两条：
  · GFLOPS：v1/v0 的比值
  · ISA：v1 的 FMA 占比是否从 21% 显著上升（"喂满 FMA"的直接读数）
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
BASE = os.path.join(HERE, '_feed')

HEAD = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
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

# ---------------- v0：现状 ----------------
V0 = HEAD + r'''
__kernel void v0(const int M, const int N, const int K,
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
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
''' + TAIL

# ---------------- v1：指针外提 + 去守卫 ----------------
# 每个线程固定负责 4 个 float4（A 两个、B 两个，都是 lane 与 lane+64），
# 所以这 4 个基址可以在 k 循环外【只算一次】；循环内只剩"基址 + 立即数偏移"。
# K 用编译期常量 => (by+row)*K 折叠成移位/立即数乘法。
V1 = HEAD + r'''
__kernel void v1(const int M, const int N, const int K,
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

    // ---- 4 个基址，循环外算一次 ----
    const int ar0 = lane/AROW,      ak0 = (lane%AROW)*4;
    const int ar1 = (lane+64)/AROW, ak1 = ((lane+64)%AROW)*4;
    const int bk0 = lane/BCOL,      bc0 = (lane%BCOL)*4;
    const int bk1 = (lane+64)/BCOL, bc1 = ((lane+64)%BCOL)*4;
    const __global float* Ap0 = A + (by+ar0)*KC + ak0;
    const __global float* Ap1 = A + (by+ar1)*KC + ak1;
    const __global float* Bp0 = B + bk0*NC + bx + bc0;
    const __global float* Bp1 = B + bk1*NC + bx + bc1;

    for (int k0 = 0; k0 < K; k0 += TK) {
        float4 va0 = *(const __global float4*)(Ap0 + k0);
        float4 va1 = *(const __global float4*)(Ap1 + k0);
        float4 vb0 = *(const __global float4*)(Bp0 + k0*NC);
        float4 vb1 = *(const __global float4*)(Bp1 + k0*NC);
        As[ak0/4+0][ar0]=va0.x; As[ak0/4+1][ar0]=va0.y;
        As[ak0/4+2][ar0]=va0.z; As[ak0/4+3][ar0]=va0.w;
        As[ak1/4+0][ar1]=va1.x; As[ak1/4+1][ar1]=va1.y;
        As[ak1/4+2][ar1]=va1.z; As[ak1/4+3][ar1]=va1.w;
        Bs[bk0][bc0+0]=vb0.x; Bs[bk0][bc0+1]=vb0.y;
        Bs[bk0][bc0+2]=vb0.z; Bs[bk0][bc0+3]=vb0.w;
        Bs[bk1][bc1+0]=vb1.x; Bs[bk1][bc1+1]=vb1.y;
        Bs[bk1][bc1+2]=vb1.z; Bs[bk1][bc1+3]=vb1.w;
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
''' + TAIL


def build(ctx, name, src, extra=()):
    d = os.path.join(BASE, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        prog = cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable', '-save-temps', *extra])
    finally:
        os.chdir(cwd)
    return prog


def isa_hist(name):
    d = os.path.join(BASE, name)
    for f in glob.glob(os.path.join(d, '*.isa')):
        txt = open(f, encoding='utf-8', errors='replace').read()
        seg = txt.split('Disassembly:')
        if len(seg) < 2:
            continue
        ops = re.findall(r'^\s+([a-z][a-z0-9_]*)\s.*//\s*[0-9a-f]+:', seg[1], re.M)
        if not ops:
            continue
        import collections
        h = collections.Counter(ops)
        n = len(ops)
        fma = sum(v for k, v in h.items() if k.startswith(('v_fma', 'v_mac')))
        ints = sum(v for k, v in h.items()
                   if k.startswith(('v_add', 'v_mul', 'v_lshl', 'v_lshr', 'v_ashr',
                                    'v_and', 'v_or', 'v_xor', 'v_cndmask')))
        br = sum(v for k, v in h.items() if k.startswith(('s_cbranch', 's_andn2')))
        return n, fma, ints, br
    return None


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print(f'device {dev.name} | {dev.max_compute_units} CU | 每 CU LDS '
          f'{dev.local_mem_size//1024} KB（机器空闲）')

    KC = 512   # K 特化；先只对 K=512 的形状做，验证机制
    progs = {'v0': build(ctx, 'v0', V0, extra=[f'-DKC={KC}', f'-DNC={KC}']),
             'v1': build(ctx, 'v1', V1, extra=[f'-DKC={KC}', f'-DNC={KC}'])}
    print(f"\n{'变体':<5} {'总指令':>7} {'FMA':>6} {'FMA%':>6} {'整数/地址':>9} "
          f"{'整数%':>6} {'分支':>6}")
    for name in ('v0', 'v1'):
        r = isa_hist(name)
        if r:
            n, fma, ints, br = r
            print(f'{name:<5} {n:>7} {fma:>6} {fma/n*100:>5.1f}% {ints:>9} '
                  f'{ints/n*100:>5.1f}% {br:>6}')

    mf = cl.mem_flags
    print(f"\n{'形状':>20} {'v0':>9} {'v1':>9} {'v1/v0':>8}  说明")
    for M, N, K, note in ((512, 512, 512, 'K=512 特化命中'),):
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
        fl = 2.0 * M * N * K
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
              cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
        cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        out = np.empty((M, N), np.float32)
        res, errs = {}, {}
        for name in ('v0', 'v1'):
            k = cl.Kernel(progs[name], name)
            args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cbuf)
            k(q, gws, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(5):
                k(q, gws, (8, 8), *args)
            q.finish()
            res[name] = fl / ((time.perf_counter() - t0) / 5) / 1e9
            cl.enqueue_copy(q, out, cbuf).wait()
            ref = a @ b
            errs[name] = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        ok = 'OK' if errs['v1'] < 1e-4 else f"⚠️ v1 err={errs['v1']:.1e}"
        print(f'{M:>8}×{N:<5}×{K:<4} {res["v0"]:>9.1f} {res["v1"]:>9.1f} '
              f'{res["v1"]/res["v0"]:>7.2f}×  {note}  {ok}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
