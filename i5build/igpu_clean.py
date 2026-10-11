"""空闲机器上的核显干净测量：先把"分母"量准，再谈优化。

要回答的三件事：
  1. 这颗 Vega 6 在【我们的条件下】实际能跑多少 FMA/s？⇒ 反推实际频率。
     报告里的"峰值 955~1150 GFLOPS"是按 1.5 GHz 算的；而 15 W APU 上 CPU 负载
     会吃掉功耗预算（同一个 die）。332 GFLOPS"只占 29%"这个说法，分母可能根本不对。
  2. 内循环的指令账：16 条 LDS + 64 条 FMA = 80 条发 64 次 FMA ⇒ FMA 密度 80%。
     把 LDS 读向量化（float4）后是 4 条 + 64 条 = 68 条 ⇒ 密度 94%，**上限只有 +17%**。
     （我之前说"接近翻倍"是把"LDS 指令少 4 倍"错当成"吞吐翻倍"，没做指令数算术。）
  3. 在这两个前提下，fp32 / float4-LDS / fp16 / fp16+float4-LDS 四版在【我们的形状】
     上各是多少。

微基准这次写对：**16 条独立累加链** + 充分展开（上一版两条串行依赖链，测的是延迟）。
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
BASE = os.path.join(HERE, '_clean')

FMA_SRC = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fma32(__global float* out, const int iters) {
    // 16 条独立链：FMA 延迟约 4~5 周期，4 条就能打满；16 条留足余量
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
__kernel void fma16(__global float* out, const int iters) {
    // 8 条 half2 链 = 每迭代 16 次 fp16 FMA（若编译器真的发出 packed 指令）
    half2 a0=(half2)(1.0h,1.01h), a1=(half2)(1.02h,1.03h), a2=(half2)(1.04h,1.05h), a3=(half2)(1.06h,1.07h);
    half2 a4=(half2)(1.08h,1.09h), a5=(half2)(1.10h,1.11h), a6=(half2)(1.12h,1.13h), a7=(half2)(1.14h,1.15h);
    const half2 b = (half2)(0.999h,0.999h), c = (half2)(0.001h,0.001h);
    for (int i = 0; i < iters; i++) {
        a0=fma(a0,b,c); a1=fma(a1,b,c); a2=fma(a2,b,c); a3=fma(a3,b,c);
        a4=fma(a4,b,c); a5=fma(a5,b,c); a6=fma(a6,b,c); a7=fma(a7,b,c);
    }
    out[get_global_id(0)] = (float)(a0.x+a0.y+a1.x+a1.y+a2.x+a2.y+a3.x+a3.y
                                   +a4.x+a4.y+a5.x+a5.y+a6.x+a6.y+a7.x+a7.y);
}
'''

GEMM_HEAD = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define NTH_ 64
__kernel void {name}(const int M, const int N, const int K,
                     const __global {ty}* restrict A,
                     const __global {ty}* restrict B,
                     __global float* restrict C) {{
    __local {ty} As[TK][TS + {pad}];
    __local {ty} Bs[TK][TS + {pad}];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    const int VEC = {vec};
    const int AROW = TK/VEC, BCOL = TS/VEC;
    const int ATOT = TS*AROW, BTOT = TK*BCOL;
    for (int k0 = 0; k0 < K; k0 += TK) {{
        for (int p = 0; p < (ATOT+BTOT+NTH_-1)/NTH_; p++) {{
            int f = p*NTH_ + lane;
            if (f < ATOT) {{
                int row = f/AROW, kq = f%AROW;
                int gr = by+row, gc = k0+kq*VEC;
                {tyv} v = ({tyv})(0);
                if (gr<M && gc+VEC-1<K) v = *(const __global {tyv}*)(A+gr*K+gc);
                As[kq*VEC+0][row]=v.s0; As[kq*VEC+1][row]=v.s1;
                As[kq*VEC+2][row]=v.s2; As[kq*VEC+3][row]=v.s3;
                {extra_a}
            }} else if (f < ATOT+BTOT) {{
                int g = f-ATOT, kk = g/BCOL, cq = g%BCOL;
                int gr = k0+kk, gc = bx+cq*VEC;
                {tyv} v = ({tyv})(0);
                if (gr<K && gc+VEC-1<N) v = *(const __global {tyv}*)(B+gr*N+gc);
                Bs[kk][cq*VEC+0]=v.s0; Bs[kk][cq*VEC+1]=v.s1;
                Bs[kk][cq*VEC+2]=v.s2; Bs[kk][cq*VEC+3]=v.s3;
                {extra_b}
            }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
{inner}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
{tail}
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

INNER_SCALAR = r'''        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i=0;i<TM;i++) a[i]=(float)As[k][ty*TM+i];
            for (int j=0;j<TN;j++) b[j]=(float)Bs[k][tx*TN+j];
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
        }'''

# 向量化 LDS 读：TM=TN=8 ⇒ 各 2 条 float4（对齐由 +4 padding 保证：行距 272 B = 17×16）
INNER_VEC4 = r'''        for (int k = 0; k < TK; k++) {
            float4 a0 = *(const __local float4*)(&As[k][ty*TM]);
            float4 a1 = *(const __local float4*)(&As[k][ty*TM + 4]);
            float4 b0 = *(const __local float4*)(&Bs[k][tx*TN]);
            float4 b1 = *(const __local float4*)(&Bs[k][tx*TN + 4]);
            float av[8] = {a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
            float bv[8] = {b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
            for (int i=0;i<TM;i++)
                for (int j=0;j<TN;j++) acc[i][j]=fma(av[i],bv[j],acc[i][j]);
        }'''

VARIANTS = {
    'g32':   dict(ty='float', tyv='float4', pad=4, vec=4, inner=INNER_SCALAR,
                  extra_a='', extra_b=''),
    'g32v':  dict(ty='float', tyv='float4', pad=4, vec=4, inner=INNER_VEC4,
                  extra_a='', extra_b=''),
    'g16':   dict(ty='half', tyv='half8', pad=8, vec=8, inner=INNER_SCALAR,
                  extra_a='As[kq*8+4][row]=v.s4; As[kq*8+5][row]=v.s5; '
                          'As[kq*8+6][row]=v.s6; As[kq*8+7][row]=v.s7;',
                  extra_b='Bs[kk*1][cq*8+4]=v.s4;'),
}


def make(name, cfg):
    d = dict(cfg)
    d['name'] = name
    d['tail'] = TAIL
    return GEMM_HEAD.format(**d)


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print(f'device {dev.name} | {dev.max_compute_units} CU | wave {dev.max_work_group_size} '
          f'| LDS {dev.local_mem_size//1024} KB')
    print('（机器现在空闲 —— 这是干净测量）\n')

    # ---------- 1. FMA 峰值 → 反推频率 ----------
    print('=== 1. FMA 峰值（16 条独立链，展开）===')
    prog = cl.Program(ctx, FMA_SRC).build(options=['-cl-fast-relaxed-math',
                                                   '-cl-mad-enable'])
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * 65536)
    print(f"{'内核':<8} {'ms':>9} {'G-FMA/s':>10} {'隐含频率(MHz)':>14}")
    for name in ('fma32', 'fma16'):
        k = cl.Kernel(prog, name)
        n, iters = 65536, 20000
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        t0 = time.perf_counter()
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        dt = time.perf_counter() - t0
        fma = 16 * n * iters
        per_s = fma / dt / 1e9
        # 6 CU × 64 lane × 1 FMA/周期
        mhz = per_s * 1e9 / (6 * 64) / 1e6
        print(f'{name:<8} {dt*1e3:>9.2f} {per_s:>10.1f} {mhz:>14.0f}')

    # ---------- 2. GEMM 四版 ----------
    (os.makedirs(BASE, exist_ok=True))
    progs = {}
    for name, cfg in VARIANTS.items():
        d = os.path.join(BASE, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(d)
        try:
            progs[name] = cl.Program(ctx, make(name, cfg)).build(
                options=['-cl-fast-relaxed-math', '-cl-mad-enable', '-save-temps'])
        finally:
            os.chdir(cwd)

    print('\n=== 2. GEMM：fp32 / fp32+float4LDS（我们的形状）===')
    mf = cl.mem_flags
    shapes = [(1024, 1024, 1024, '锚点（报告里 332 的那个）'),
              (2048, 512, 512, 'AR 注意力'),
              (2048, 2048, 512, 'AR FFN up'),
              (2048, 512, 2048, 'AR FFN down')]
    print(f"{'形状':>20} {'g32':>9} {'g32v':>9} {'v/标量':>8}  说明")
    for M, N, K, note in shapes:
        a = np.random.rand(M, K).astype(np.float32)
        b = np.random.rand(K, N).astype(np.float32)
        fl = 2.0 * M * N * K
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
              cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
        cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        res = {}
        for name in ('g32', 'g32v'):
            k = cl.Kernel(progs[name], name)
            args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cbuf)
            k(q, gws, (8, 8), *args)
            q.finish()
            t0 = time.perf_counter()
            for _ in range(3):
                k(q, gws, (8, 8), *args)
            q.finish()
            res[name] = fl / ((time.perf_counter() - t0) / 3) / 1e9
        print(f'{M:>8}×{N:<5}×{K:<4} {res["g32"]:>9.1f} {res["g32v"]:>9.1f} '
              f'{res["g32v"]/res["g32"]:>7.2f}×  {note}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
