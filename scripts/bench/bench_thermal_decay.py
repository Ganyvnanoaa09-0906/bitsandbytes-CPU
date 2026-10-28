# -*- coding: utf-8 -*-
"""bench_thermal_decay.py — 测核显的热衰减曲线

背景：
  同一个 SGEMM 配置，两次测量差 2.3 倍（231.3 vs 98.9）
  而且每个 run 的第一个配置最好，后面单调衰减
  ⇒ 强烈怀疑持续负载下核显降频

本脚本测两件事：
  ① 热衰减曲线：单一负载连续跑 90 秒，每秒采样一次 GFLOPS
     分两组：纯 FMA（纯算力） vs SGEMM（真实负载）
  ② 交替轮询：A→B→A→B... 各取最大值（消除热状态 carryover）

如果衰减是 3× 级别，那么"调 tile 拿 1.2×"就没意义 —— 优先要解决的是热。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import time
import numpy as np
import pyopencl as cl

SRC_FMA = r'''
__kernel void fma_peak(__global float* out, const int iters) {
    const int gid = get_global_id(0);
    float b = 1.0000001f, c = 1e-12f;
    float a0 = 1.0f + gid * 1e-9f;
    float a1=a0+.01f, a2=a0+.02f, a3=a0+.03f, a4=a0+.04f, a5=a0+.05f, a6=a0+.06f, a7=a0+.07f;
    float a8=a0+.08f, a9=a0+.09f, aa=a0+.10f, ab=a0+.11f, ac=a0+.12f, ad=a0+.13f, ae=a0+.14f, af=a0+.15f;
    for (int i = 0; i < iters; i++) {
        a0=fma(a0,b,c); a1=fma(a1,b,c); a2=fma(a2,b,c); a3=fma(a3,b,c);
        a4=fma(a4,b,c); a5=fma(a5,b,c); a6=fma(a6,b,c); a7=fma(a7,b,c);
        a8=fma(a8,b,c); a9=fma(a9,b,c); aa=fma(aa,b,c); ab=fma(ab,b,c);
        ac=fma(ac,b,c); ad=fma(ad,b,c); ae=fma(ae,b,c); af=fma(af,b,c);
    }
    out[gid] = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+aa+ab+ac+ad+ae+af;
}
'''

SRC_GEMM = r'''
#define TS 64
#define TK 16
#define TM 8
#define TN 8
__kernel void sgemm_t(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    float acc[TM][TN];
    #pragma unroll
    for (int i=0;i<TM;i++) {
            #pragma unroll
            for (int j=0;j<TN;j++) acc[i][j]=0.f;
        }
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int lane = ty * 8 + tx;
        #pragma unroll
        for (int p = 0; p < (TK*TS*2)/(64*4); p++) {
            int flat4 = p*64 + lane, total4 = TK*TS/4;
            if (flat4 < total4) {
                int r=(flat4*4)/TS, c=(flat4*4)%TS, gr=by+r, gc=k0+c;
                float4 v = (gr<M && gc+3<K) ? *(const __global float4*)(A+gr*K+gc) : (float4)0.f;
                As[r][c]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
            } else {
                int f2=flat4-total4, r=(f2*4)/TS, c=(f2*4)%TS, gr=k0+r, gc=bx+c;
                float4 v = (gr<K && gc+3<N) ? *(const __global float4*)(B+gr*N+gc) : (float4)0.f;
                Bs[r][c]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k=0;k<TK;k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i=0;i<TM;i++) a[i]=As[k][ty*TM+i];
            #pragma unroll
            for (int j=0;j<TN;j++) b[j]=Bs[k][tx*TN+j];
            #pragma unroll
            for (int i=0;i<TM;i++) {
                #pragma unroll
                for (int j=0;j<TN;j++) acc[i][j]=fma(a[i],b[j],acc[i][j]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i=0;i<TM;i++) {
        int rv=by+ty*TM+i; if (rv>=M) continue;
        #pragma unroll
        for (int j=0;j<TN;j++) { int cv=bx+tx*TN+j; if (cv<N) C[rv*N+cv]=acc[i][j]; }
    }
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
print('=' * 84)
print('核显热衰减曲线')
print('=' * 84)
print('  设备:', dev.name, '|', dev.max_compute_units, 'CU')
print()

fma = cl.Kernel(cl.Program(ctx, SRC_FMA).build(), 'fma_peak')
gem = cl.Kernel(cl.Program(ctx, SRC_GEMM).build(), 'sgemm_t')

# --- FMA 缓冲 ---
NF = 262144
bf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, NF * 4)
FMA_IT = 40000

# --- GEMM 缓冲 1024³ ---
GN = 1024
A = np.random.randn(GN, GN).astype(np.float32)
B = np.random.randn(GN, GN).astype(np.float32)
mf = cl.mem_flags
ag = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
cg = cl.Buffer(ctx, mf.WRITE_ONLY, GN * GN * 4)
GGSZ = (GN // 64 * 8, GN // 64 * 8)
FLOP_G = 2.0 * GN ** 3
FLOP_F = float(NF) * FMA_IT * 16 * 2


def run_fma(secs):
    # 队列必须限流！不限流会排几十万个 kernel，queue.finish() 永远等不完
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < secs:
        for _ in range(4):
            fma(queue, (NF,), None, bf, np.int32(FMA_IT))
        queue.finish()
        n += 4
    dt = time.perf_counter() - t0
    return n * FLOP_F / dt / 1e9


def run_gemm(secs):
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < secs:
        for _ in range(4):
            gem(queue, GGSZ, (8, 8), np.int32(GN), np.int32(GN), np.int32(GN), ag, bg, cg)
        queue.finish()
        n += 4
    dt = time.perf_counter() - t0
    return n * FLOP_G / dt / 1e9


def curve(name, fn, total=90, step=4):
    print('  预热 20 秒...', flush=True)
    fn(20)
    print('  --- %s：连续 %d 秒，每 %d 秒采样 ---' % (name, total, step))
    print('     时刻(s)   GFLOPS   相对峰值')
    vals = []
    for i in range(0, total, step):
        gf = fn(step)
        vals.append(gf)
        peak = max(vals)
        print('     %6d   %8.1f   %7.1f%%' % (i + step, gf, 100 * gf / peak))
    peak, last = max(vals), vals[-1]
    print('     ⇒ 峰值 %.1f  末值 %.1f  【衰减 %.2fx】\n' % (peak, last, peak / last))
    return peak, last


print('  提示：核显持续负载会降频。先量出幅度，再决定要不要优化 kernel。')
print()
p_f, l_f = curve('纯 FMA（算力）', run_fma)
p_g, l_g = curve('SGEMM 1024³（真实负载）', run_gemm)

print('=' * 84)
print('结论')
print('=' * 84)
print('  纯 FMA   : 峰值 %7.1f → 末值 %7.1f   衰减 %.2fx' % (p_f, l_f, p_f / l_f))
print('  SGEMM    : 峰值 %7.1f → 末值 %7.1f   衰减 %.2fx' % (p_g, l_g, p_g / l_g))
print()
print('  参照: CLBlast 278.6 / DML 294.5 / 我手写 245.3（都是不同时间的单点值）')
print()
if p_g / l_g > 2.0:
    print('  ⇒ 【热衰减超过 2 倍】—— 任何"SGEMM 上限"的数字都不可复现，')
    print('     而且把核显放进实时管线拿不到性能保证')
elif p_g / l_g > 1.3:
    print('  ⇒ 有可测的热衰减，测 SGEMM 上限必须用"交替轮询取最大"')
else:
    print('  ⇒ 热衰减不明显，"大 tile 更慢"是真实的 kernel 特性')
print('DONE')
