# -*- coding: utf-8 -*-
"""真 flash attention（前向）：把 S = Q·Kᵀ / online softmax / O = P·V 融进一个 kernel，
不落盘 S、P。

形状：T=1024 行（token 轴）、HD=64（头维）、BH=16 个 (batch,head)、causal。
映射：一个工作组 = 64 行 × 64 头维（O 的一块），16×16=256 线程，
      每线程负责 4 行 × 4 头维的 O 累加（16 个 fp32 累加器）。

local 预算（32KB）：Qt 8KB（Q tile，整个 WG 复用）+ KV 8KB（先放 K、算完 S 再换 V）
                  + St 8KB（先 S 后 P）+ rowmax/rowsum/中间量 ≈ 24.5KB ✓

每个列块 KS=64 的流程：
  ① 载入 K tile → 算 S（4×4/线程，1024 MAC）
  ② 行 max 归约（16 个 lx 线程的部分 max 汇总）→ 更新 mrow、rowscale
  ③ P = exp(S·scale − m)、行和归约 → 更新 lrow；O 累加器先乘 rowscale
  ④ 载入 V tile → O += P·V（4 行 × 4 维 × 64 = 1024 MAC）
最后 O /= lrow。
"""
from __future__ import annotations
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

SRC = r"""
__kernel void flash_fwd(__global const half* Q, __global const half* K, __global const half* V,
                        __global half* O, int T, float scale) {
    __local half Qt[64 * 64];
    __local half KV[64 * 64];
    __local half St[64 * 64];
    __local float mrow[64];
    __local float lrow[64];
    __local float rscale[64];
    __local float part[64 * 16];

    int bh = get_group_id(1), rb = get_group_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Qp = Q + (long)bh * T * 64;
    __global const half* Kp = K + (long)bh * T * 64;
    __global const half* Vp = V + (long)bh * T * 64;
    __global half* Op = O + (long)bh * T * 64;
    int r0 = rb * 64;

    // Q tile：64 行 × 64 维
    for (int t = tid; t < 64 * 8; t += 256) {
        int r = t / 8, c8 = (t % 8) * 8;
        vstore8(vload8(0, Qp + (long)(r0 + r) * 64 + c8), 0, (__local half*)Qt + r * 64 + c8);
    }
    if (tid < 64) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    int myr0 = ly * 4, myc0 = lx * 4;
    float oacc[4][4];          // O 的持久累加器（跨列块）
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) oacc[i][j] = 0.f;

    int nblk = (r0 + 64 + 63) / 64;          // 因果：只做到自己这一列块
    for (int cb = 0; cb < nblk; ++cb) {
        int c0 = cb * 64;
        // ① K tile
        for (int t = tid; t < 64 * 8; t += 256) {
            int r = t / 8, c8 = (t % 8) * 8;
            vstore8(vload8(0, Kp + (long)(c0 + r) * 64 + c8), 0, (__local half*)KV + r * 64 + c8);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // S = Qt · KT  （每线程 4 行 × 4 列，k=64 全在 local）
        float sacc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) sacc[i][j] = 0.f;
        for (int kk = 0; kk < 64; ++kk) {
            float av[4], bv[4];
            // Qt 是 [row][k] 行主序，kk 固定 → 按行标量取
            av[0] = (float)Qt[(myr0 + 0) * 64 + kk];
            av[1] = (float)Qt[(myr0 + 1) * 64 + kk];
            av[2] = (float)Qt[(myr0 + 2) * 64 + kk];
            av[3] = (float)Qt[(myr0 + 3) * 64 + kk];
            bv[0] = (float)KV[(myc0 + 0) * 64 + kk];
            bv[1] = (float)KV[(myc0 + 1) * 64 + kk];
            bv[2] = (float)KV[(myc0 + 2) * 64 + kk];
            bv[3] = (float)KV[(myc0 + 3) * 64 + kk];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) sacc[i][j] += av[i] * bv[j];
        }
        // 暂存 S（fp32→half）到 St，同时算本线程 4 列的部分 max
        float pm[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int grow = r0 + myr0 + i;
            pm[i] = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gcol = c0 + myc0 + j;
                float sv = sacc[i][j] * scale;
                int keep = (grow >= gcol) ? 1 : 0;          // causal 掩码
                St[(myr0 + i) * 64 + myc0 + j] = (half)(keep ? sv : -30000.f);
                pm[i] = fmax(pm[i], keep ? sv : -1e30f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // ② 行 max 归约：16 个 lx 的部分 max → mrow
#pragma unroll
        for (int i = 0; i < 4; ++i) part[(myr0 + i) * 16 + lx] = pm[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = myr0 + i;
                float mx = -1e30f;
                for (int q = 0; q < 16; ++q) mx = fmax(mx, part[r * 16 + q]);
                float mnew = fmax(mrow[r], mx);
                rscale[r] = exp(mrow[r] - mnew);
                mrow[r] = mnew;
                lrow[r] *= rscale[r];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // ③ P = exp(S − m)，行和归约
        float ps[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float m = mrow[myr0 + i];
            float rs = rscale[myr0 + i];
            float sum = 0.f;
            int grow = r0 + myr0 + i;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gcol = c0 + myc0 + j;
                float e = 0.f;
                if (grow >= gcol) e = exp((float)St[(myr0 + i) * 64 + myc0 + j] - m);
                St[(myr0 + i) * 64 + myc0 + j] = (half)e;   // 就地变成 P
                sum += e;
                oacc[i][j] *= rs;                           // O 累加器按行缩放
            }
            ps[i] = sum;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) part[(myr0 + i) * 16 + lx] = ps[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lx == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = myr0 + i;
                float s = 0.f;
                for (int q = 0; q < 16; ++q) s += part[r * 16 + q];
                lrow[r] += s;
            }
        }
        // ④ V tile（K 已用完）
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = tid; t < 64 * 8; t += 256) {
            int r = t / 8, c8 = (t % 8) * 8;
            vstore8(vload8(0, Vp + (long)(c0 + r) * 64 + c8), 0, (__local half*)KV + r * 64 + c8);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // O += P · V（4 行 × 4 维，k=64）
        for (int kk = 0; kk < 64; ++kk) {
            float pv[4];
            pv[0] = (float)St[(myr0 + 0) * 64 + kk];
            pv[1] = (float)St[(myr0 + 1) * 64 + kk];
            pv[2] = (float)St[(myr0 + 2) * 64 + kk];
            pv[3] = (float)St[(myr0 + 3) * 64 + kk];
            float vv0 = (float)KV[kk * 64 + myc0 + 0];
            float vv1 = (float)KV[kk * 64 + myc0 + 1];
            float vv2 = (float)KV[kk * 64 + myc0 + 2];
            float vv3 = (float)KV[kk * 64 + myc0 + 3];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float a0 = (j == 0) ? vv0 : (j == 1) ? vv1 : (j == 2) ? vv2 : vv3;
                oacc[0][j] += pv[0] * a0; oacc[1][j] += pv[1] * a0;
                oacc[2][j] += pv[2] * a0; oacc[3][j] += pv[3] * a0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // 输出
    for (int i = 0; i < 4; ++i) {
        int r = r0 + myr0 + i;
        float inv = 1.f / lrow[myr0 + i];
        for (int j = 0; j < 4; ++j) {
            int c = myc0 + j;
            Op[(long)r * 64 + c] = (half)(oacc[i][j] * inv);
        }
    }
}
"""


def main():
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A
    d0 = MM.Dev()
    d0.activate()

    ctx, q = d0.ctx, d0.q
    mf = cl.mem_flags
    T, HD, BH = MM.T, MM.HD, MM.BH
    scale = 0.125
    kern = cl.Kernel(cl.Program(ctx, SRC).build(options=['-cl-mad-enable']), 'flash_fwd')
    rng = np.random.default_rng(0)
    Qn = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    Kn = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    Vb = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    qb = cl.Buffer(ctx, mf.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Qn)
    kb = cl.Buffer(ctx, mf.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Kn)
    vb = cl.Buffer(ctx, mf.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=Vb)
    ob = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
    ref = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
    S = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)
    P = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)

    def old_path():
        # 当前模型里真正在跑的三个 kernel（都做过因果裁剪/向量化）
        d0.K['sgemm_nt64'](q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                           qb, kb, S, np.int32(T), np.float32(1.0))
        d0.K['softmax_causal_v'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
        d0.K['sgemm_nn_bt'](q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                            P, vb, ref, np.int32(T), np.float32(1.0))

    def flash():
        kern(q, ((T // 64) * 16, BH * 16), (16, 16), qb, kb, vb, ob, np.int32(T), np.float32(scale))

    import time

    def t(fn, n=10):
        fn(); q.finish()
        t0 = time.time()
        for _ in range(n):
            fn()
        q.finish()
        return (time.time() - t0) / n * 1000

    old_path(); flash(); q.finish()
    o1 = np.empty(BH * T * HD, np.float16); cl.enqueue_copy(q, o1, ref)
    o2 = np.empty(BH * T * HD, np.float16); cl.enqueue_copy(q, o2, ob)
    q.finish()
    rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                / (np.abs(o1.astype(np.float32)).max() + 1e-12))
    t_old, t_new = t(old_path), t(flash)
    print('O = softmax(Q·Kᵀ)·V   (16 × 1024×1024×64, causal)')
    print('  现在（S+softmax+O 三个 kernel）: %7.2f ms' % t_old)
    print('  flash（一个 kernel）           : %7.2f ms   %.2fx' % (t_new, t_old / t_new))
    print('  数值 rel=%.2e  %s' % (rel, 'OK' if rel < 5e-2 else 'FAIL'))


if __name__ == '__main__':
    main()
