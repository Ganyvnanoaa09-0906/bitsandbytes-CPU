# -*- coding: utf-8 -*-
"""F1-M4（多 context 版）：30M SmallARImageModelV2 整机训练在 iGPU 上。

为什么要「每层一个 OpenCL context」
--------------------------------
实测本机 AMD gfx902 驱动的一个硬缺陷：**同一个 context 里『大小混杂』的 buffer 超过约
130 个之后，后续分配的内存会互相踩**（读回来的内容变成别的 buffer 的内容 / 全 0），
而且整个过程不报任何错误。证据链：
  - 1 层整机（~67 个 buffer）：embed / 每层输入 / logits 全部与 torch 对齐（rel 2e-3）；
  - 2 层：仍然全对；3 层起：所有量都变成 0..1023（那是 tick buffer 的图案）；
  - 同尺寸 buffer 单独测：2MB×700、4KB×400 都正常；**大小混杂**时在 ~128 个处必坏。
所以把每层的 buffer 关进各自的 context（每层 ~41 个 buffer），层间用 host 传 x
（2MB/次，一层一个来回，代价 ~2ms，相对 3s/step 可忽略）。

其它实测教训（都留了探针）：
  - `mf.COPY_HOST_PTR` 在 buffer 多了以后会上传错数据 -> 一律「先分配 + 显式 copy + finish」
  - 8 字节回读在本机永远错，4KB 也会偶发错，**全量回读**才是对的
  - CLBlast 的 hgemm 在 B≠8 时 split_q 曾越界写坏 buffer（已在 attention_gpu 修掉）

结构
----
LayerCtx : 一个 context 里的一层（权重 + AdamW 状态 + 激活 + 前向/反向）
Model    : token_emb/row_emb/col_emb 在 host；逐层调 LayerCtx；head+CE 也在一个 LayerCtx 里
"""
from __future__ import annotations
import os
import sys
import ctypes

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import pyopencl as cl

import gt1030.attention_gpu as A

BATCH = int(os.environ.get('AR_B', '2'))
SEQ = int(os.environ.get('AR_T', '1024'))
A.B, A.T = BATCH, SEQ
A.M = A.B * A.T
A.BH = A.B * A.NH
import gt1030.full_gpu_layer as L          # noqa: E402
import gt1030.full_gpu_step_v2 as V2       # noqa: E402

B, T, MD, FF, WROW, BH, HD, NH, NKV = L.B, L.T, L.MD, L.FF, L.WROW, L.BH, L.HD, L.NH, L.NKV
M = L.M
scale = L.scale

# CE 梯度放大倍数：避免反向中间量掉进 fp16 次正规区（见文件头注释）
CE_SCALE = float(os.environ.get('AR_CE_SCALE', '1024'))
NO_TICK = os.environ.get('AR_NOTICK', '0') == '1'
NO_VERIFY = os.environ.get('AR_NOVERIFY', '0') == '1'

KSRC = r"""
__kernel void qk_norm_fwd(__global const half* x, __global const float* w,
                          __global half* y, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global half* out = y + (long)r * D;
    float ss = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) out[i] = (half)((float)row[i] * inv * w[i]);
}
__kernel void qk_norm_dx(__global const half* x, __global const float* w,
                         __global const half* dy, __global half* dx, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global const half* drow = dy + (long)r * D;
    __global half* ox = dx + (long)r * D;
    float ss = 0.f, dot = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) dot += (float)row[i] * (float)drow[i];
    float c = dot * inv * inv * inv / D;
    for (int i = 0; i < D; ++i)
        ox[i] = (half)((float)drow[i] * inv * w[i] - (float)row[i] * c * w[i]);
}
__kernel void split_silu(__global const half* a2, __global half* g, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    g[i] = (half)(a / (1.f + exp(-a)) * b);
}
__kernel void split_silu_bwd(__global const half* a2, __global const half* dg,
                             __global half* da2, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    float g = (float)dg[i];
    float sg = 1.f / (1.f + exp(-a));
    float silu_a = a * sg;
    float dsilu = sg * (1.f + a * (1.f - sg));
    da2[(long)r * 2 * FF + j] = (half)(g * b * dsilu);
    da2[(long)r * 2 * FF + FF + j] = (half)(g * silu_a);
}
__kernel void scale_vec(__global const half* x, __global const float* s, __global half* y,
                        int rows, int D) {
    int i = get_global_id(0); if (i >= rows * D) return;
    y[i] = (half)((float)x[i] * s[i % D]);
}
// 单行 softmax + CE：一个工作组处理一行，V<=1024。
// 用 get_group_id(0) 当行号 —— 所以 global size 必须是 rows*256（踩过一次坑）。
__kernel void ce_softmax(__global const half* lg, __global const int* tgt,
                         __global const float* mask, __global half* dlg,
                         __global float* loss_row, int V, float inv_cnt, float gscale) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float vals[1024];
    __local float red[256];
    __global const half* line = lg + (long)r * V;
    __global half* out = dlg + (long)r * V;
    float mx = -1e30f;
    for (int k = lid; k < V; k += lsz) mx = fmax(mx, (float)line[k]);
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.f;
    for (int k = lid; k < V; k += lsz) { float e = exp((float)line[k] - m); vals[k] = e; s += e; }
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    int t = tgt[r];
    float mk = mask[r];
    float ls = 0.f;
    for (int k = lid; k < V; k += lsz) {
        float p = vals[k] * inv;
        out[k] = (half)(mk * (p - ((k == t) ? 1.f : 0.f)) * inv_cnt * gscale);
        if (k == t) ls = -log(p + 1e-12f);
    }
    red[lid] = ls; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) loss_row[r] = red[0] * mk * inv_cnt;
}
// RMSNorm 的 w 梯度（并行版）：grid = (d, NCHUNK)，每个 work-item 扫一段行，部分和进 part
__kernel void rms_dw_part(__global const half* x, __global const half* dh,
                          __global const float* ss, __global float* part,
                          int rows, int d) {
    int i = get_global_id(0);
    int ch = get_global_id(1), nch = get_global_size(1);
    float acc = 0.f;
    for (int r = ch; r < rows; r += nch) {
        float inv = rsqrt(ss[r] + 1e-6f);
        acc += (float)dh[(long)r * d + i] * (float)x[(long)r * d + i] * inv;
    }
    part[(long)ch * d + i] = acc;
}
// RMSNorm 的 w 梯度：需要事先用 rms_bwd_stats 算好 ss（每行的均方）
__kernel void rms_dw_from_ss(__global const half* x, __global const half* dh,
                             __global const float* ss, __global float* dw,
                             int rows, int d) {
    int i = get_global_id(0);
    if (i >= d) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) {
        float inv = rsqrt(ss[r] + 1e-6f);
        acc += (float)dh[(long)r * d + i] * (float)x[(long)r * d + i] * inv;
    }
    dw[i] = acc;
}
// ls 向量梯度：out[i] = Σ_r a[r,i]·b[r,i]
__kernel void col_sum_prod(__global const half* a, __global const half* b,
                           __global float* out, int rows, int d) {
    int i = get_global_id(0);
    if (i >= d) return;
    float acc = 0.f;
    for (int r = 0; r < rows; ++r) acc += (float)a[(long)r * d + i] * (float)b[(long)r * d + i];
    out[i] = acc;
}
__kernel void scale_float(__global float* g, float f, int n) {
    int i = get_global_id(0); if (i >= n) return;
    g[i] = g[i] * f;
}
// k=64 的 batched GEMM：C[b] = A[b]·B[b]ᵀ（A/B 都是 (T,64) row-major）
// 用于 S = Qn·Knᵀ 与 dP = dO·Vbᵀ。128x64 tile / 每线程 8x4 / 16x16 线程 = 实测 1.78x CLBlast。
__kernel void sgemm_nt64(__global const half* A, __global const half* B,
                         __global half* C, int T, float alpha) {
    __local half At[64 * 128];
    __local half Bt[64 * 64];
    int bh = get_group_id(2);
    int ti = get_group_id(0), tj = get_group_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * 64;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * T;
    int mb = ti * 128, nb = tj * 64;
    // 因果：整块在严格上三角就不用算（softmax 只读下三角，上三角它自己写 0）
    if (nb > mb + 127) return;
    for (int t = tid; t < 128 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);
        At[(c4 + 0) * 128 + r] = va.x; At[(c4 + 1) * 128 + r] = va.y;
        At[(c4 + 2) * 128 + r] = va.z; At[(c4 + 3) * 128 + r] = va.w;
    }
    for (int t = tid; t < 64 * 16; t += 256) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 vb = vload4(0, Bp + (long)(nb + r) * 64 + c4);
        Bt[(c4 + 0) * 64 + r] = vb.x; Bt[(c4 + 1) * 64 + r] = vb.y;
        Bt[(c4 + 2) * 64 + r] = vb.z; Bt[(c4 + 3) * 64 + r] = vb.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int m0 = ly * 8, n0 = lx * 4;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        float av[8], bv[4];
        half4 a0 = vload4(0, (__local half*)At + kk * 128 + m0);
        half4 a1 = vload4(0, (__local half*)At + kk * 128 + m0 + 4);
        av[0] = (float)a0.x; av[1] = (float)a0.y; av[2] = (float)a0.z; av[3] = (float)a0.w;
        av[4] = (float)a1.x; av[5] = (float)a1.y; av[6] = (float)a1.z; av[7] = (float)a1.w;
        half4 b0 = vload4(0, (__local half*)Bt + kk * 64 + n0);
        bv[0] = (float)b0.x; bv[1] = (float)b0.y; bv[2] = (float)b0.z; bv[3] = (float)b0.w;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
    }
    for (int i = 0; i < 8; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < 4; ++j) {
            int gn = nb + n0 + j;
            if (gn >= T) break;
            Cp[(long)gm * T + gn] = (half)(acc[i][j] * alpha);
        }
    }
}

// C[b] = A[b]·B[b]，A (T,T) row-major、B (T,64) row-major、C (T,64)。
// 用于 O = P·Vb 与 dQ = dS·Kn。MS=128/NS=64/BK=64，1.50x CLBlast。
__kernel void sgemm_nn_bt(__global const half* A, __global const half* B,
                          __global half* C, int T, float alpha) {
    __local half At[64 * 128];
    __local half Bt[64 * 64];
    int bh = get_group_id(2);
    int mb = get_group_id(0) * 128;
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * T;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * 64;
    int m0 = ly * 8, n0 = lx * 4;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    int kend = (mb + 128 < T) ? (mb + 128) : T;   // 因果：P/dS 上三角为 0，k 不必跑满
    for (int k0 = 0; k0 < kend; k0 += 64) {
        for (int t = tid; t < 128 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Ap + (long)(mb + r) * T + k0 + c4);
            At[(c4 + 0) * 128 + r] = v.x; At[(c4 + 1) * 128 + r] = v.y;
            At[(c4 + 2) * 128 + r] = v.z; At[(c4 + 3) * 128 + r] = v.w;
        }
        for (int t = tid; t < 64 * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Bp + (long)(k0 + r) * 64 + c4);
            vstore4(v, 0, (__local half*)Bt + r * 64 + c4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 64; ++kk) {
            float av[8], bv[4];
            half4 a0 = vload4(0, (__local half*)At + kk * 128 + m0);
            half4 a1 = vload4(0, (__local half*)At + kk * 128 + m0 + 4);
            av[0] = (float)a0.x; av[1] = (float)a0.y; av[2] = (float)a0.z; av[3] = (float)a0.w;
            av[4] = (float)a1.x; av[5] = (float)a1.y; av[6] = (float)a1.z; av[7] = (float)a1.w;
            half4 b = vload4(0, (__local half*)Bt + kk * 64 + n0);
            bv[0] = (float)b.x; bv[1] = (float)b.y; bv[2] = (float)b.z; bv[3] = (float)b.w;
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 8; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < 4; ++j) {
            int gn = n0 + j;
            if (gn >= 64) break;
            Cp[(long)gm * 64 + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
// 梯度平方和：每个 work-item 归约一段，输出 256 个部分和（host 侧再求和）
__kernel void sumsq_half(__global const half* g, __global float* out, int n) {
    int i = get_global_id(0), lsz = get_global_size(0);
    float acc = 0.f;
    for (int k = i; k < n; k += lsz) { float v = (float)g[k]; acc += v * v; }
    out[i] = acc;
}

// ================= 向量化小 kernel（每线程 8 个元素）=================
// 融合版 RMSNorm 前向：一遍读 x 同时算均值平方并写 y = x*inv*w，ss 仍写出（供 w 梯度用）
__kernel void rms_fwd_fused(__global const half* x, __global const half* w, __global half* y,
                            __global float* ss, int rows, int d) {
    int r = get_group_id(0);          // 一个工作组 = 一行
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* xr = x + (long)r * d;
    __global half* yr = y + (long)r * d;
    float acc = 0.f;
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 v = vload8(0, xr + i);
        acc += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
             + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
             + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
             + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
    }
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = red[0] / (float)d;
    if (lid == 0) ss[r] = mean;
    float inv = rsqrt(mean + EPS);
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 v = vload8(0, xr + i);
        half8 wv = vload8(0, w + i);
        half8 o;
        o.s0 = (half)((float)v.s0 * inv * (float)wv.s0);
        o.s1 = (half)((float)v.s1 * inv * (float)wv.s1);
        o.s2 = (half)((float)v.s2 * inv * (float)wv.s2);
        o.s3 = (half)((float)v.s3 * inv * (float)wv.s3);
        o.s4 = (half)((float)v.s4 * inv * (float)wv.s4);
        o.s5 = (half)((float)v.s5 * inv * (float)wv.s5);
        o.s6 = (half)((float)v.s6 * inv * (float)wv.s6);
        o.s7 = (half)((float)v.s7 * inv * (float)wv.s7);
        vstore8(o, 0, yr + i);
    }
}
// 融合版 RMSNorm 反向：一遍算 ss 与 s=Σ dh*w*x，再写 dx
__kernel void rms_bwd_fused(__global const half* x, __global const half* dh,
                            __global const half* w, __global half* dx,
                            __global float* ss, __global float* sc, int rows, int d) {
    int r = get_group_id(0);          // 一个工作组 = 一行
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[512];
    __global const half* xr = x + (long)r * d;
    __global const half* hr = dh + (long)r * d;
    __global half* dr = dx + (long)r * d;
    float a = 0.f, b = 0.f;
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 xv = vload8(0, xr + i);
        half8 hv = vload8(0, hr + i);
        half8 wv = vload8(0, w + i);
        for (int t = 0; t < 8; ++t) {
            float xi = (t == 0) ? (float)xv.s0 : (t == 1) ? (float)xv.s1 : (t == 2) ? (float)xv.s2
                     : (t == 3) ? (float)xv.s3 : (t == 4) ? (float)xv.s4 : (t == 5) ? (float)xv.s5
                     : (t == 6) ? (float)xv.s6 : (float)xv.s7;
            float hi = (t == 0) ? (float)hv.s0 : (t == 1) ? (float)hv.s1 : (t == 2) ? (float)hv.s2
                     : (t == 3) ? (float)hv.s3 : (t == 4) ? (float)hv.s4 : (t == 5) ? (float)hv.s5
                     : (t == 6) ? (float)hv.s6 : (float)hv.s7;
            float wi = (t == 0) ? (float)wv.s0 : (t == 1) ? (float)wv.s1 : (t == 2) ? (float)wv.s2
                     : (t == 3) ? (float)wv.s3 : (t == 4) ? (float)wv.s4 : (t == 5) ? (float)wv.s5
                     : (t == 6) ? (float)wv.s6 : (float)wv.s7;
            a += xi * xi;
            b += hi * wi * xi;
        }
    }
    red[lid] = a;
    red[256 + lid] = b;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { red[lid] += red[lid + off]; red[256 + lid] += red[256 + lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float ssv = red[0] / (float)d;
    float scv = red[256];
    if (lid == 0) { ss[r] = ssv; sc[r] = scv; }
    float sig = sqrt(ssv + EPS);
    float inv = 1.f / sig;
    float coef = scv / ((float)d * sig * sig * sig);
    for (int i = lid * 8; i < d; i += lsz * 8) {
        half8 xv = vload8(0, xr + i);
        half8 hv = vload8(0, hr + i);
        half8 wv = vload8(0, w + i);
        half8 o;
        o.s0 = (half)((float)hv.s0 * (float)wv.s0 * inv - (float)xv.s0 * coef);
        o.s1 = (half)((float)hv.s1 * (float)wv.s1 * inv - (float)xv.s1 * coef);
        o.s2 = (half)((float)hv.s2 * (float)wv.s2 * inv - (float)xv.s2 * coef);
        o.s3 = (half)((float)hv.s3 * (float)wv.s3 * inv - (float)xv.s3 * coef);
        o.s4 = (half)((float)hv.s4 * (float)wv.s4 * inv - (float)xv.s4 * coef);
        o.s5 = (half)((float)hv.s5 * (float)wv.s5 * inv - (float)xv.s5 * coef);
        o.s6 = (half)((float)hv.s6 * (float)wv.s6 * inv - (float)xv.s6 * coef);
        o.s7 = (half)((float)hv.s7 * (float)wv.s7 * inv - (float)xv.s7 * coef);
        vstore8(o, 0, dr + i);
    }
}
// out = x + s*y（ls 残差；把 scale_vec + add_vec 融成一步）
__kernel void scale_add_v(__global const half* x, __global const float* s,
                          __global const half* y, __global half* o, int rows, int d) {
    int i = get_global_id(0) * 8;
    if (i >= rows * d) return;
    int c = i % d;
    half8 xv = vload8(0, x + i);
    half8 yv = vload8(0, y + i);
    half8 r;
    r.s0 = (half)((float)xv.s0 + s[c + 0] * (float)yv.s0);
    r.s1 = (half)((float)xv.s1 + s[c + 1] * (float)yv.s1);
    r.s2 = (half)((float)xv.s2 + s[c + 2] * (float)yv.s2);
    r.s3 = (half)((float)xv.s3 + s[c + 3] * (float)yv.s3);
    r.s4 = (half)((float)xv.s4 + s[c + 4] * (float)yv.s4);
    r.s5 = (half)((float)xv.s5 + s[c + 5] * (float)yv.s5);
    r.s6 = (half)((float)xv.s6 + s[c + 6] * (float)yv.s6);
    r.s7 = (half)((float)xv.s7 + s[c + 7] * (float)yv.s7);
    vstore8(r, 0, o + i);
}
// 单标量缩放（ls 反向用）
__kernel void scale_v(__global const half* x, __global const float* s, __global half* y,
                      int rows, int d) {
    int i = get_global_id(0) * 8;
    if (i >= rows * d) return;
    int c = i % d;
    half8 xv = vload8(0, x + i);
    half8 r;
    r.s0 = (half)((float)xv.s0 * s[c + 0]);
    r.s1 = (half)((float)xv.s1 * s[c + 1]);
    r.s2 = (half)((float)xv.s2 * s[c + 2]);
    r.s3 = (half)((float)xv.s3 * s[c + 3]);
    r.s4 = (half)((float)xv.s4 * s[c + 4]);
    r.s5 = (half)((float)xv.s5 * s[c + 5]);
    r.s6 = (half)((float)xv.s6 * s[c + 6]);
    r.s7 = (half)((float)xv.s7 * s[c + 7]);
    vstore8(r, 0, y + i);
}
__kernel void add_v(__global const half* a, __global const half* b, __global half* o, int n) {
    int i = get_global_id(0) * 8;
    if (i >= n) return;
    half8 av = vload8(0, a + i);
    half8 bv = vload8(0, b + i);
    half8 r;
    r.s0 = (half)((float)av.s0 + (float)bv.s0);
    r.s1 = (half)((float)av.s1 + (float)bv.s1);
    r.s2 = (half)((float)av.s2 + (float)bv.s2);
    r.s3 = (half)((float)av.s3 + (float)bv.s3);
    r.s4 = (half)((float)av.s4 + (float)bv.s4);
    r.s5 = (half)((float)av.s5 + (float)bv.s5);
    r.s6 = (half)((float)av.s6 + (float)bv.s6);
    r.s7 = (half)((float)av.s7 + (float)bv.s7);
    vstore8(r, 0, o + i);
}
// SwiGLU 拆分（前向）：g = silu(a2[:FF]) * a2[FF:]
__kernel void split_silu_v(__global const half* a2, __global half* g, int rows, int FF) {
    int t = get_global_id(0);
    int per = FF / 8;
    int r = t / per, c = (t % per) * 8;
    long base = (long)r * 2 * FF + c;
    half8 av = vload8(0, a2 + base);
    half8 bv = vload8(0, a2 + base + FF);
    float a[8];
    a[0] = (float)av.s0; a[1] = (float)av.s1; a[2] = (float)av.s2; a[3] = (float)av.s3;
    a[4] = (float)av.s4; a[5] = (float)av.s5; a[6] = (float)av.s6; a[7] = (float)av.s7;
    float b[8];
    b[0] = (float)bv.s0; b[1] = (float)bv.s1; b[2] = (float)bv.s2; b[3] = (float)bv.s3;
    b[4] = (float)bv.s4; b[5] = (float)bv.s5; b[6] = (float)bv.s6; b[7] = (float)bv.s7;
    half8 o;
    o.s0 = (half)(a[0] / (1.f + exp(-a[0])) * b[0]);
    o.s1 = (half)(a[1] / (1.f + exp(-a[1])) * b[1]);
    o.s2 = (half)(a[2] / (1.f + exp(-a[2])) * b[2]);
    o.s3 = (half)(a[3] / (1.f + exp(-a[3])) * b[3]);
    o.s4 = (half)(a[4] / (1.f + exp(-a[4])) * b[4]);
    o.s5 = (half)(a[5] / (1.f + exp(-a[5])) * b[5]);
    o.s6 = (half)(a[6] / (1.f + exp(-a[6])) * b[6]);
    o.s7 = (half)(a[7] / (1.f + exp(-a[7])) * b[7]);
    vstore8(o, 0, g + (long)r * FF + c);
}
// SwiGLU 拆分（反向）
__kernel void split_silu_bwd_v(__global const half* a2, __global const half* dg,
                               __global half* da2, int rows, int FF) {
    int t = get_global_id(0);
    int per = FF / 8;
    int r = t / per, c = (t % per) * 8;
    long base = (long)r * 2 * FF + c;
    half8 av = vload8(0, a2 + base);
    half8 bv = vload8(0, a2 + base + FF);
    half8 gv = vload8(0, dg + (long)r * FF + c);
    float a[8], b[8], gg[8];
    a[0] = (float)av.s0; a[1] = (float)av.s1; a[2] = (float)av.s2; a[3] = (float)av.s3;
    a[4] = (float)av.s4; a[5] = (float)av.s5; a[6] = (float)av.s6; a[7] = (float)av.s7;
    b[0] = (float)bv.s0; b[1] = (float)bv.s1; b[2] = (float)bv.s2; b[3] = (float)bv.s3;
    b[4] = (float)bv.s4; b[5] = (float)bv.s5; b[6] = (float)bv.s6; b[7] = (float)bv.s7;
    gg[0] = (float)gv.s0; gg[1] = (float)gv.s1; gg[2] = (float)gv.s2; gg[3] = (float)gv.s3;
    gg[4] = (float)gv.s4; gg[5] = (float)gv.s5; gg[6] = (float)gv.s6; gg[7] = (float)gv.s7;
    half8 o1, o2;
    float sg[8], silu[8], ds[8];
    __global half* outp = da2 + base;
    __global half* outp2 = da2 + base + FF;
    half h1[8], h2[8];
    for (int t2 = 0; t2 < 8; ++t2) {
        sg[t2] = 1.f / (1.f + exp(-a[t2]));
        silu[t2] = a[t2] * sg[t2];
        ds[t2] = sg[t2] * (1.f + a[t2] * (1.f - sg[t2]));
        h1[t2] = (half)(gg[t2] * b[t2] * ds[t2]);
        h2[t2] = (half)(gg[t2] * silu[t2]);
    }
    o1.s0 = h1[0]; o1.s1 = h1[1]; o1.s2 = h1[2]; o1.s3 = h1[3];
    o1.s4 = h1[4]; o1.s5 = h1[5]; o1.s6 = h1[6]; o1.s7 = h1[7];
    o2.s0 = h2[0]; o2.s1 = h2[1]; o2.s2 = h2[2]; o2.s3 = h2[3];
    o2.s4 = h2[4]; o2.s5 = h2[5]; o2.s6 = h2[6]; o2.s7 = h2[7];
    vstore8(o1, 0, outp);
    vstore8(o2, 0, outp2);
}

// ================= 向量化：qkv 拆分 / 合并 / qk-norm =================
// 沿最后的 D(=64) 方向一次处理 8 个元素
__kernel void split_q_v(__global const half* qkv, __global half* q,
                        int T, int D, int nh, int nkv) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;                    // hp*T + t
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    long src = ((long)b * T + t) * W + h * D + d8;
    long dst = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, qkv + src), 0, q + dst);
}
__kernel void split_kv_v(__global const half* qkv, __global half* k, __global half* v,
                         int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    int kw = h / (nh / nkv);
    long base = ((long)b * T + t) * W + D * nh;
    long dst = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, qkv + base + kw * D + d8), 0, k + dst);
    vstore8(vload8(0, qkv + base + D * nkv + kw * D + d8), 0, v + dst);
}
__kernel void merge_o_v(__global const half* o, __global half* otok,
                        int T, int D, int nh, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    long dst = ((long)b * T + t) * (D * nh) + h * D + d8;
    long src = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, o + src), 0, otok + dst);
}
__kernel void split_dout_v(__global const half* dtok, __global half* dst,
                           int T, int D, int nh, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    long src = ((long)b * T + t) * (D * nh) + h * D + d8;
    long out = ((long)hp * T + t) * D + d8;
    vstore8(vload8(0, dtok + src), 0, dst + out);
}
// dqkv 合并：q 直接拷；k/v 要在 kv 组内把 nkv 个头求和（原来是逐元素串行）
__kernel void merge_dqkv_v(__global const half* dq, __global const half* dk,
                           __global const half* dv, __global half* dqkv,
                           int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hp = trow / T;
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    int kvw = nh / nkv;
    long row = ((long)b * T + t) * W;
    long src = ((long)hp * T + t) * D + d8;
    if (h < nh) {                                   // q 通道：hp 已限定 h<nh
        vstore8(vload8(0, dq + src), 0, dqkv + row + h * D + d8);
    }
}
// k/v 通道的求和单独做一个 kernel（因为每个 (b,t,kw) 只该写一次，避免多线程重复写）
__kernel void merge_dkv_v(__global const half* dk, __global const half* dv,
                          __global half* dqkv, int T, int D, int nh, int nkv, int B) {
    int g = get_global_id(0);
    int per = D / 8;
    int d8 = (g % per) * 8;
    int trow = g / per;
    int t = trow % T, hk = trow / T;               // hk in [0, B*2*nkv)
    int b = hk / (2 * nkv), ch2 = hk % (2 * nkv);
    int kvw = nh / nkv;
    int W = D * (nh + 2 * nkv);
    long row = ((long)b * T + t) * W;
    float acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;
    for (int r = 0; r < kvw; ++r) {
        int hp = b * nh + (ch2 % nkv) * kvw + r;
        long src = ((long)hp * T + t) * D + d8;
        const __global half* p = (ch2 < nkv) ? (dk + src) : (dv + src);
        half8 x = vload8(0, p);
        acc[0] += (float)x.s0; acc[1] += (float)x.s1; acc[2] += (float)x.s2; acc[3] += (float)x.s3;
        acc[4] += (float)x.s4; acc[5] += (float)x.s5; acc[6] += (float)x.s6; acc[7] += (float)x.s7;
    }
    int ch = (ch2 < nkv) ? (nh + ch2) : (nh + nkv + (ch2 - nkv));
    half8 o;
    o.s0 = (half)acc[0]; o.s1 = (half)acc[1]; o.s2 = (half)acc[2]; o.s3 = (half)acc[3];
    o.s4 = (half)acc[4]; o.s5 = (half)acc[5]; o.s6 = (half)acc[6]; o.s7 = (half)acc[7];
    vstore8(o, 0, dqkv + row + ch * D + d8);
}
// qk-norm 前向：一个工作组(8 线程)处理一行 D=64，每线程 half8
__kernel void qk_norm_fwd_v(__global const half* x, __global const float* w,
                            __global half* y, int rows, int D) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[64];
    __global const half* row = x + (long)r * D;
    __global half* out = y + (long)r * D;
    float ss = 0.f;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        ss += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
            + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
            + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
            + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
    }
    red[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)D + 1e-6f);
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 o;
        o.s0 = (half)((float)v.s0 * inv * w[i + 0]);
        o.s1 = (half)((float)v.s1 * inv * w[i + 1]);
        o.s2 = (half)((float)v.s2 * inv * w[i + 2]);
        o.s3 = (half)((float)v.s3 * inv * w[i + 3]);
        o.s4 = (half)((float)v.s4 * inv * w[i + 4]);
        o.s5 = (half)((float)v.s5 * inv * w[i + 5]);
        o.s6 = (half)((float)v.s6 * inv * w[i + 6]);
        o.s7 = (half)((float)v.s7 * inv * w[i + 7]);
        vstore8(o, 0, out + i);
    }
}
__kernel void qk_norm_dx_v(__global const half* x, __global const float* w,
                           __global const half* dy, __global half* dx, int rows, int D) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[128];
    __global const half* row = x + (long)r * D;
    __global const half* drow = dy + (long)r * D;
    __global half* ox = dx + (long)r * D;
    float ss = 0.f, dot = 0.f;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 u = vload8(0, drow + i);
        ss += (float)v.s0 * (float)v.s0 + (float)v.s1 * (float)v.s1
            + (float)v.s2 * (float)v.s2 + (float)v.s3 * (float)v.s3
            + (float)v.s4 * (float)v.s4 + (float)v.s5 * (float)v.s5
            + (float)v.s6 * (float)v.s6 + (float)v.s7 * (float)v.s7;
        dot += (float)v.s0 * (float)u.s0 + (float)v.s1 * (float)u.s1
             + (float)v.s2 * (float)u.s2 + (float)v.s3 * (float)u.s3
             + (float)v.s4 * (float)u.s4 + (float)v.s5 * (float)u.s5
             + (float)v.s6 * (float)u.s6 + (float)v.s7 * (float)u.s7;
    }
    red[lid] = ss; red[64 + lid] = dot;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { red[lid] += red[lid + off]; red[64 + lid] += red[64 + lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)D + 1e-6f);
    float c = red[64] * inv * inv * inv / (float)D;
    for (int i = lid * 8; i < D; i += lsz * 8) {
        half8 v = vload8(0, row + i);
        half8 u = vload8(0, drow + i);
        half8 o;
        o.s0 = (half)(((float)u.s0 * inv - (float)v.s0 * c) * w[i + 0]);
        o.s1 = (half)(((float)u.s1 * inv - (float)v.s1 * c) * w[i + 1]);
        o.s2 = (half)(((float)u.s2 * inv - (float)v.s2 * c) * w[i + 2]);
        o.s3 = (half)(((float)u.s3 * inv - (float)v.s3 * c) * w[i + 3]);
        o.s4 = (half)(((float)u.s4 * inv - (float)v.s4 * c) * w[i + 4]);
        o.s5 = (half)(((float)u.s5 * inv - (float)v.s5 * c) * w[i + 5]);
        o.s6 = (half)(((float)u.s6 * inv - (float)v.s6 * c) * w[i + 6]);
        o.s7 = (half)(((float)u.s7 * inv - (float)v.s7 * c) * w[i + 7]);
        vstore8(o, 0, ox + i);
    }
}

// 向量化 causal softmax：每线程一次处理 8 个元素（half8）
__kernel void softmax_causal_v(__global const half* s, __global half* p, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float vals[1024];
    __local float red[256];
    __global const half* line = s + ((long)bh * T + row) * T;
    __global half* out = p + ((long)bh * T + row) * T;
    float mx = -1e30f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float a0 = (float)v.s0 * scale, a1 = (float)v.s1 * scale;
            float a2 = (float)v.s2 * scale, a3 = (float)v.s3 * scale;
            float a4 = (float)v.s4 * scale, a5 = (float)v.s5 * scale;
            float a6 = (float)v.s6 * scale, a7 = (float)v.s7 * scale;
            mx = fmax(mx, fmax(fmax(fmax(a0, a1), fmax(a2, a3)),
                              fmax(fmax(a4, a5), fmax(a6, a7))));
        } else {
            for (int t = k; t < n; ++t) mx = fmax(mx, (float)line[t] * scale);
        }
    }
    red[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float e0 = exp((float)v.s0 * scale - m), e1 = exp((float)v.s1 * scale - m);
            float e2 = exp((float)v.s2 * scale - m), e3 = exp((float)v.s3 * scale - m);
            float e4 = exp((float)v.s4 * scale - m), e5 = exp((float)v.s5 * scale - m);
            float e6 = exp((float)v.s6 * scale - m), e7 = exp((float)v.s7 * scale - m);
            vals[k + 0] = e0; vals[k + 1] = e1; vals[k + 2] = e2; vals[k + 3] = e3;
            vals[k + 4] = e4; vals[k + 5] = e5; vals[k + 6] = e6; vals[k + 7] = e7;
            sum += (e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7);
        } else {
            for (int t = k; t < n; ++t) {
                float e = exp((float)line[t] * scale - m);
                vals[t] = e; sum += e;
            }
        }
    }
    red[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        o.s0 = (k + 0 < n) ? (half)(vals[k + 0] * inv) : (half)0.f;
        o.s1 = (k + 1 < n) ? (half)(vals[k + 1] * inv) : (half)0.f;
        o.s2 = (k + 2 < n) ? (half)(vals[k + 2] * inv) : (half)0.f;
        o.s3 = (k + 3 < n) ? (half)(vals[k + 3] * inv) : (half)0.f;
        o.s4 = (k + 4 < n) ? (half)(vals[k + 4] * inv) : (half)0.f;
        o.s5 = (k + 5 < n) ? (half)(vals[k + 5] * inv) : (half)0.f;
        o.s6 = (k + 6 < n) ? (half)(vals[k + 6] * inv) : (half)0.f;
        o.s7 = (k + 7 < n) ? (half)(vals[k + 7] * inv) : (half)0.f;
        vstore8(o, 0, out + k);
    }
}
__kernel void softmax_bwd_causal_v(__global const half* p, __global const half* dp,
                                   __global half* ds, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float red[256];
    __global const half* pl = p + ((long)bh * T + row) * T;
    __global const half* dl = dp + ((long)bh * T + row) * T;
    __global half* ol = ds + ((long)bh * T + row) * T;
    float sacc = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            sacc += (float)a.s0 * (float)b.s0 + (float)a.s1 * (float)b.s1
                  + (float)a.s2 * (float)b.s2 + (float)a.s3 * (float)b.s3
                  + (float)a.s4 * (float)b.s4 + (float)a.s5 * (float)b.s5
                  + (float)a.s6 * (float)b.s6 + (float)a.s7 * (float)b.s7;
        } else {
            for (int t = k; t < n; ++t) sacc += (float)pl[t] * (float)dl[t];
        }
    }
    red[lid] = sacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tot = red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            o.s0 = (half)((float)a.s0 * ((float)b.s0 - tot) * scale);
            o.s1 = (half)((float)a.s1 * ((float)b.s1 - tot) * scale);
            o.s2 = (half)((float)a.s2 * ((float)b.s2 - tot) * scale);
            o.s3 = (half)((float)a.s3 * ((float)b.s3 - tot) * scale);
            o.s4 = (half)((float)a.s4 * ((float)b.s4 - tot) * scale);
            o.s5 = (half)((float)a.s5 * ((float)b.s5 - tot) * scale);
            o.s6 = (half)((float)a.s6 * ((float)b.s6 - tot) * scale);
            o.s7 = (half)((float)a.s7 * ((float)b.s7 - tot) * scale);
        } else {
            o.s0 = (k + 0 < n) ? (half)((float)pl[k + 0] * ((float)dl[k + 0] - tot) * scale) : (half)0.f;
            o.s1 = (k + 1 < n) ? (half)((float)pl[k + 1] * ((float)dl[k + 1] - tot) * scale) : (half)0.f;
            o.s2 = (k + 2 < n) ? (half)((float)pl[k + 2] * ((float)dl[k + 2] - tot) * scale) : (half)0.f;
            o.s3 = (k + 3 < n) ? (half)((float)pl[k + 3] * ((float)dl[k + 3] - tot) * scale) : (half)0.f;
            o.s4 = (k + 4 < n) ? (half)((float)pl[k + 4] * ((float)dl[k + 4] - tot) * scale) : (half)0.f;
            o.s5 = (k + 5 < n) ? (half)((float)pl[k + 5] * ((float)dl[k + 5] - tot) * scale) : (half)0.f;
            o.s6 = (k + 6 < n) ? (half)((float)pl[k + 6] * ((float)dl[k + 6] - tot) * scale) : (half)0.f;
            o.s7 = (k + 7 < n) ? (half)((float)pl[k + 7] * ((float)dl[k + 7] - tot) * scale) : (half)0.f;
        }
        vstore8(o, 0, ol + k);
    }
}
__kernel void scale_half(__global half* g, float f, int n) {
    int i = get_global_id(0); if (i >= n) return;
    g[i] = (half)((float)g[i] * f);
}
"""

XGEMM_PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
             "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def xgemm_override(kreg=None, bl=None, ctx=None):
    """按我们真实形状重扫出来的 Xgemm 参数（KREG=32 最好，见 ar_v2_gemm_sweep*.py）。"""
    # ⚠️ 默认关闭：实测 KREG>1 的覆盖参数会让 Xgemm 算错（rel≈1.0），
    # 扫出来的 300~500 GFLOPS 是空转。证据见 _t_override_correct.py。
    kreg = int(os.environ.get('AR_XGEMM_KREG', '0')) if kreg is None else int(kreg)
    if kreg <= 0:
        return -1
    bl = bl or A.bl
    ctx = ctx or A.ctx
    bl.CLBlastOverrideParameters.restype = ctypes.c_int
    bl.CLBlastOverrideParameters.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
    vals_d = dict(GEMMK=0, KREG=kreg, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
                  NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    names = (ctypes.c_char_p * len(XGEMM_PN))(*[x.encode() for x in XGEMM_PN])
    vals = (ctypes.c_size_t * len(XGEMM_PN))(*[int(vals_d[k]) for k in XGEMM_PN])
    return bl.CLBlastOverrideParameters(ctypes.c_void_p(int(ctx.devices[0].int_ptr)),
                                        b'Xgemm', 16, len(XGEMM_PN), names, vals)


KNAMES = ('rms_stats', 'rms_apply', 'rms_bwd_stats', 'rms_bwd_apply', 'split_q', 'split_kv',
          'softmax_causal', 'softmax_bwd_causal', 'merge_o', 'split_dout', 'merge_dqkv',
          'add_vec', 'adamw_single',
          'qk_norm_fwd', 'qk_norm_dx', 'split_silu', 'split_silu_bwd', 'scale_vec',
          'ce_softmax', 'scale_half', 'scale_float', 'rms_dw_from_ss', 'col_sum_prod',
          'sgemm_nt64', 'sgemm_nn_bt', 'sumsq_half', 'rms_dw_part',
          'rms_fwd_fused', 'rms_bwd_fused', 'scale_add_v', 'scale_v', 'add_v',
          'split_silu_v', 'split_silu_bwd_v',
          'split_q_v', 'split_kv_v', 'merge_o_v', 'split_dout_v',
          'merge_dqkv_v', 'merge_dkv_v', 'qk_norm_fwd_v', 'qk_norm_dx_v',
          'softmax_causal_v', 'softmax_bwd_causal_v')


ARENA_MB = int(os.environ.get('AR_ARENA_MB', '1800'))
ALIGN = 4096          # 驱动 mem_base_addr_align = 2048，取 4096 更稳


class Dev(object):
    """单 context + 单一大 arena；所有张量都是 arena 的 sub-buffer。"""

    def __init__(self, arena_mb=None):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        self.prog = cl.Program(self.ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + KSRC).build(
            options=['-cl-mad-enable'])
        self.K = {n: cl.Kernel(self.prog, n) for n in KNAMES}
        self.arena = cl.Buffer(self.ctx, self.mf.READ_WRITE, (arena_mb or ARENA_MB) << 20)
        self._off = ALIGN
        if os.environ.get('AR_NOOVERRIDE', '0') != '1':
            rc = xgemm_override(bl=None, ctx=self.ctx)
            if rc == 0:
                print('[xgemm] override KREG=%s applied'
                      % os.environ.get('AR_XGEMM_KREG', '0'), flush=True)
        self._health()
        self._tick_buf = self._alloc(2048)
        self._part = self._alloc(256 * 4)      # sumsq 的部分和
        self._part_pool = []                   # sumsq_many 用的槽位池
        self._dwpart = self._alloc(32 * 1024 * 4)   # rms_dw_part 的部分和（最多 32*d, d<=1024）

    # ---- arena 分配器 ----
    def _alloc(self, nbytes):
        n = (max(int(nbytes), 8) + ALIGN - 1) // ALIGN * ALIGN
        o = (self._off + ALIGN - 1) // ALIGN * ALIGN
        if o + n > self.arena.size:
            raise RuntimeError('arena 不够：需要 %d，已用 %d / %d'
                               % (n, o, self.arena.size))
        self._off = o + n
        return self.arena.get_sub_region(o, n)

    def activate(self):
        """把 A.* 的全局句柄指到本 context（hgemm_* 走 A.QP / A.queue / A.ctx）。"""
        A.ctx, A.queue, A.mf = self.ctx, self.q, self.mf
        A.QP = ctypes.c_void_p(int(self.q.int_ptr))

    # ---------------- buffer helpers ----------------
    def buf(self, nbytes):
        b = self._alloc(nbytes)
        self._tick()
        return b

    def up(self, a):
        a = np.ascontiguousarray(a)
        b = self._alloc(a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        self.q.finish()
        if NO_VERIFY:
            return b
        back = np.empty(a.size, a.dtype)
        cl.enqueue_copy(self.q, back, b)
        self.q.finish()
        if not np.array_equal(back, a.ravel()):
            raise RuntimeError('upload verify failed (%s)' % (a.shape,))
        return b

    def down(self, b, shape, dtype=np.float16, sync=True):
        a = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.q, a, b)
        if sync:
            self.q.finish()
        return a

    def finish(self):
        self.q.finish()

    def sumsq_f32(self, buf, n):
        g = np.empty(n, np.float32)
        cl.enqueue_copy(self.q, g, buf)
        self.q.finish()
        return float((g * g).sum())

    def rms_dw(self, x, dh, ss, out, rows, d, nch=32):
        """并行版 RMSNorm w 梯度：只回传 nch*d 个部分和再求和"""
        self.K['rms_dw_part'](self.q, (d, nch), None, x, dh, ss, self._dwpart,
                              np.int32(rows), np.int32(d))
        part = np.empty(nch * d, np.float32)
        cl.enqueue_copy(self.q, part, self._dwpart)
        self.q.finish()
        dw = part.reshape(nch, d).sum(axis=0)
        cl.enqueue_copy(self.q, out, np.ascontiguousarray(dw.astype(np.float32)),
                        is_blocking=False)
        return dw

    def sumsq_many(self, items):
        """items: [(buf, n), ...] —— 一次性 enqueue 全部回合，只同步一次"""
        if not self._part_pool:
            self._part_pool = [self._alloc(256 * 4) for _ in range(64)]
        n_it = len(items)
        while len(self._part_pool) < n_it:
            self._part_pool.append(self._alloc(256 * 4))
        for i, (buf, n) in enumerate(items):
            self.K['sumsq_half'](self.q, (256,), None, buf, self._part_pool[i], np.int32(n))
            # 同一队列顺序执行，槽位互不覆盖
        out = [np.empty(256, np.float32) for _ in range(n_it)]
        for i in range(n_it):
            cl.enqueue_copy(self.q, out[i], self._part_pool[i], is_blocking=False)
        self.q.finish()
        return [float(o.sum()) for o in out]

    def sumsq(self, buf, n):
        """设备端求 half 数组的平方和（回传 256 个部分和，再在 host 求和）"""
        self.K['sumsq_half'](self.q, (256,), None, buf, self._part, np.int32(n))
        p = np.empty(256, np.float32)
        cl.enqueue_copy(self.q, p, self._part)
        self.q.finish()
        return float(p.sum())

    def _tick(self):
        if NO_TICK:
            self.q.finish()
            return
        a = np.arange(1024, dtype=np.float16)
        cl.enqueue_copy(self.q, self._tick_buf, a, is_blocking=True)
        out = np.empty(1024, np.float16)
        cl.enqueue_copy(self.q, out, self._tick_buf)
        self.q.finish()
        if not np.array_equal(out, a):
            raise RuntimeError('tick failed')

    def _health(self):
        a = np.array([0.5, -1.5, 2.25, 3.75], np.float32)
        b = cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        cl.enqueue_copy(self.q, b, a, is_blocking=True)
        out = np.empty(4, np.float32)
        cl.enqueue_copy(self.q, out, b)
        self.q.finish()
        if not np.array_equal(out, a):
            raise RuntimeError('dev tick failed')


class Param(object):
    """一层里的一个权重矩阵：fp16 前向副本 + fp16 梯度（device AdamW，状态也在 device）。"""

    def __init__(self, dev, w_fp16):
        self.dev = dev
        self.n = int(np.prod(w_fp16.shape))
        self.w = dev.up(w_fp16)
        self.grad = dev.buf(self.n * 2)
        self.master = dev.up(w_fp16.astype(np.float32).ravel())
        self.mom = dev.up(np.zeros(self.n, np.float32))
        self.vel = dev.up(np.zeros(self.n, np.float32))

    def step(self, lr, b1, b2, eps, wd, bc1, bc2):
        d = self.dev
        d.K['adamw_single'](d.q, (self.n // 4,), None, self.master, self.mom, self.vel,
                            self.grad, self.w, np.int32(self.n), np.float32(lr),
                            np.float32(b1), np.float32(b2), np.float32(eps), np.float32(wd),
                            np.float32(bc1), np.float32(bc2))


class Block(object):
    """一层 V2 block（buffer 都从共享 arena 里切）。"""

    def __init__(self, sd, idx, dev=None):
        self.dev = d = dev or Dev()
        p = 'blocks.%d.' % idx
        Wqkv = torch.cat([sd[p + 'attn.q.weight'], sd[p + 'attn.k.weight'],
                          sd[p + 'attn.v.weight']], 0).T.contiguous().numpy().astype(np.float16)
        Wo = sd[p + 'attn.o.weight'].T.contiguous().numpy().astype(np.float16)
        W12 = sd[p + 'mlp.w12.weight'].T.contiguous().numpy().astype(np.float16)
        W3 = sd[p + 'mlp.w3.weight'].T.contiguous().numpy().astype(np.float16)
        self.W = {k: Param(d, v) for k, v in
                  (('Wqkv', Wqkv), ('Wo', Wo), ('W12', W12), ('W3', W3))}
        # host 侧的 float32 主副本（小向量走 host AdamW）
        self.host = {
            'ln1': sd[p + 'ln1.w'].numpy().astype(np.float32).copy(),
            'ln2': sd[p + 'ln2.w'].numpy().astype(np.float32).copy(),
            'qw': sd[p + 'attn.q_norm.w'].numpy().astype(np.float32).copy(),
            'kw': sd[p + 'attn.k_norm.w'].numpy().astype(np.float32).copy(),
            'ls1': sd[p + 'ls1'].numpy().astype(np.float32).copy(),
            'ls2': sd[p + 'ls2'].numpy().astype(np.float32).copy(),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.host.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.host.items()}
        self.dw = {k: d.buf(len(v) * 4) for k, v in self.host.items()}
        self.ln1 = d.up(sd[p + 'ln1.w'].numpy().astype(np.float16))
        self.ln2 = d.up(sd[p + 'ln2.w'].numpy().astype(np.float16))
        self.qw = d.up(sd[p + 'attn.q_norm.w'].numpy().astype(np.float32))
        self.kw = d.up(sd[p + 'attn.k_norm.w'].numpy().astype(np.float32))
        self.ls1 = d.up(sd[p + 'ls1'].numpy().astype(np.float32))
        self.ls2 = d.up(sd[p + 'ls2'].numpy().astype(np.float32))
        b = {}
        for k, nb in (('xin', M * MD), ('h1', M * MD), ('h2', M * MD), ('x1', M * MD),
                      ('o_tok', M * MD), ('P', BH * T * T), ('Qn', BH * T * HD),
                      ('Kn', BH * T * HD), ('Qraw', BH * T * HD), ('Kraw', BH * T * HD),
                      ('Vb', BH * T * HD), ('a2', M * 2 * FF), ('g', M * FF),
                      ('qkv', M * WROW), ('S', BH * T * T), ('O', BH * T * HD),
                      ('attn', M * MD), ('y', M * MD), ('xout', M * MD),
                      ('dqkv', M * WROW), ('dy', M * MD), ('dg', M * FF), ('da2', M * 2 * FF),
                      ('dh2', M * MD), ('dx1mlp', M * MD), ('dattn', M * MD),
                      ('dO_tok', M * MD), ('dO', BH * T * HD), ('dP', BH * T * T),
                      ('dS', BH * T * T), ('dQ', BH * T * HD), ('dK', BH * T * HD),
                      ('dV', BH * T * HD), ('dQraw', BH * T * HD), ('dKraw', BH * T * HD),
                      ('dh1', M * MD), ('dxattn', M * MD), ('dx', M * MD), ('din_a', M * MD),
                      ('ys', M * MD)):
            b[k] = d.buf(nb * 2)
        b['ss'] = d.buf(M * 4)
        b['sc'] = d.buf(M * 4)
        b['ss2'] = d.buf(M * 4)
        b['sc2'] = d.buf(M * 4)
        b['ss_q'] = d.buf(BH * T * 4)
        b['sc_q'] = d.buf(BH * T * 4)
        self.b = b
        self.din = b['din_a']      # 反向时上一层的 dout 会搬进来

    # ---------------------------------------------------------------- 前向
    def forward(self, x_host):
        """x_host: (M,MD) float16 -> 返回 (M,MD) float16（本层输出，host 侧）"""
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['xin'], np.ascontiguousarray(x_host), is_blocking=True)
        d.finish()
        self.fwd_body()
        d.finish()
        return d.down(b['xout'], (M, MD), np.float16)

    def fwd_body(self):
        """只跑 kernel：输入已在 b['xin']，输出留在 b['xout']（不传输、不同步）"""
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        K['rms_fwd_fused'](d.q, (M * 64,), (64,), b['xin'], self.ln1, b['h1'], b['ss'],
                           np.int32(M), np.int32(MD))
        L.hgemm_nn(b['h1'], self.W['Wqkv'].w, b['qkv'], M, WROW, MD)
        K['split_q_v'](d.q, (BH * T * HD // 8,), (128,), b['qkv'], b['Qraw'],
                       np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
        K['split_kv_v'](d.q, (BH * T * HD // 8,), (128,), b['qkv'], b['Kraw'], b['Vb'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        K['qk_norm_fwd_v'](d.q, (BH * T * 8,), (8,), b['Qraw'], self.qw, b['Qn'],
                           np.int32(BH * T), np.int32(HD))
        K['qk_norm_fwd_v'](d.q, (BH * T * 8,), (8,), b['Kraw'], self.kw, b['Kn'],
                           np.int32(BH * T), np.int32(HD))
        # S = Qn·Knᵀ：自写 k=64 batched（比 CLBlast strided-batched 快 1.78×，独占口径）
        K['sgemm_nt64'](d.q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                        b['Qn'], b['Kn'], b['S'], np.int32(T), np.float32(1.0))
        K['softmax_causal_v'](d.q, (T * 256, BH), (256, 1), b['S'], b['P'],
                              np.int32(T), np.float32(scale))
        K['sgemm_nn_bt'](d.q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                         b['P'], b['Vb'], b['O'], np.int32(T), np.float32(1.0))
        K['merge_o_v'](d.q, (BH * T * HD // 8,), (128,), b['O'], b['o_tok'],
                       np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
        L.hgemm_nn(b['o_tok'], self.W['Wo'].w, b['attn'], M, MD, MD)
        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['xin'], self.ls1, b['attn'], b['x1'],
                         np.int32(M), np.int32(MD))
        K['rms_fwd_fused'](d.q, (M * 64,), (64,), b['x1'], self.ln2, b['h2'], b['ss'],
                           np.int32(M), np.int32(MD))
        L.hgemm_nn(b['h2'], self.W['W12'].w, b['a2'], M, 2 * FF, MD)
        K['split_silu_v'](d.q, (M * FF // 8,), (128,), b['a2'], b['g'],
                          np.int32(M), np.int32(FF))
        L.hgemm_nn(b['g'], self.W['W3'].w, b['y'], M, MD, FF)
        K['scale_add_v'](d.q, (M * MD // 8,), (128,), b['x1'], self.ls2, b['y'], b['xout'],
                         np.int32(M), np.int32(MD))

    # ---------------------------------------------------------------- 反向
    def backward(self, dout_host):
        """dout_host: (M,MD) float16 -> 返回 dx (M,MD) float16"""
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        cl.enqueue_copy(d.q, b['din_a'], np.ascontiguousarray(dout_host), is_blocking=True)
        d.finish()
        self.bwd_body()
        d.finish()
        return d.down(b['dx'], (M, MD), np.float16)

    def bwd_body(self):
        """只跑 kernel：上游梯度已在 b['din_a']，结果留在 b['dx']"""
        d, K, b = self.dev, self.dev.K, self.b
        d.activate()
        din = b['din_a']
        K['scale_v'](d.q, (M * MD // 8,), (128,), din, self.ls2, b['dy'],
                     np.int32(M), np.int32(MD))
        L.hgemm_tn(b['g'], b['dy'], self.W['W3'].grad, FF, MD, M)
        L.hgemm_nt(b['dy'], self.W['W3'].w, b['dg'], M, FF, MD)
        K['split_silu_bwd_v'](d.q, (M * FF // 8,), (128,), b['a2'], b['dg'], b['da2'],
                              np.int32(M), np.int32(FF))
        L.hgemm_tn(b['h2'], b['da2'], self.W['W12'].grad, MD, 2 * FF, M)
        L.hgemm_nt(b['da2'], self.W['W12'].w, b['dh2'], M, MD, 2 * FF)
        K['rms_bwd_fused'](d.q, (M * 64,), (64,), b['x1'], b['dh2'], self.ln2, b['dx1mlp'],
                           b['ss2'], b['sc2'], np.int32(M), np.int32(MD))
        K['scale_v'](d.q, (M * MD // 8,), (128,), din, self.ls1, b['dattn'],
                     np.int32(M), np.int32(MD))
        L.hgemm_tn(b['o_tok'], b['dattn'], self.W['Wo'].grad, MD, MD, M)
        L.hgemm_nt(b['dattn'], self.W['Wo'].w, b['dO_tok'], M, MD, MD)
        K['split_dout_v'](d.q, (BH * T * HD // 8,), (128,), b['dO_tok'], b['dO'],
                          np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
        K['sgemm_nt64'](d.q, ((T + 127) // 128 * 16, (T + 63) // 64 * 16, BH), (16, 16, 1),
                        b['dO'], b['Vb'], b['dP'], np.int32(T), np.float32(1.0))
        K['softmax_bwd_causal_v'](d.q, (T * 256, BH), (256, 1), b['P'], b['dP'], b['dS'],
                                  np.int32(T), np.float32(scale))
        A.hgemm_batched(b['P'], b['dO'], b['dV'], T, HD, T, A.YES, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        K['sgemm_nn_bt'](d.q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1),
                         b['dS'], b['Kn'], b['dQ'], np.int32(T), np.float32(1.0))
        A.hgemm_batched(b['dS'], b['Qn'], b['dK'], T, HD, T, A.YES, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Qraw'], self.qw, b['dQ'], b['dQraw'],
                          np.int32(BH * T), np.int32(HD))
        K['qk_norm_dx_v'](d.q, (BH * T * 8,), (8,), b['Kraw'], self.kw, b['dK'], b['dKraw'],
                          np.int32(BH * T), np.int32(HD))
        K['merge_dqkv_v'](d.q, (BH * T * HD // 8,), (128,), b['dQraw'], b['dKraw'], b['dV'],
                          b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        K['merge_dkv_v'](d.q, (B * 2 * NKV * T * HD // 8,), (128,), b['dKraw'], b['dV'],
                         b['dqkv'], np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        L.hgemm_tn(b['h1'], b['dqkv'], self.W['Wqkv'].grad, MD, WROW, M)
        L.hgemm_nt(b['dqkv'], self.W['Wqkv'].w, b['dh1'], M, MD, WROW)
        K['rms_bwd_stats'](d.q, (M,), (64,), b['xin'], b['dh1'], self.ln1,
                           b['ss'], b['sc'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](d.q, (M * MD,), (256,), b['xin'], b['dh1'], self.ln1,
                           b['ss'], b['sc'], b['dxattn'], np.int32(M), np.int32(MD))
        K['add_vec'](d.q, (M * MD,), (256,), din, b['dx1mlp'], b['dx'], np.int32(M * MD))
        K['add_vec'](d.q, (M * MD,), (256,), b['dx'], b['dxattn'], b['dx'], np.int32(M * MD))
        # ---- 小向量梯度（ln1/ln2 的 w、qk-norm 的 w、ls1/ls2）----
        # ss 里还留着 attention 分支（xin）的统计量，直接复用
        d.rms_dw(b['xin'], b['dh1'], b['ss'], self.dw['ln1'], M, MD)
        d.rms_dw(b['x1'], b['dh2'], b['ss2'], self.dw['ln2'], M, MD)
        K['col_sum_prod'](d.q, (MD,), None, b['attn'], din, self.dw['ls1'],
                          np.int32(M), np.int32(MD))
        # dls2 用 din（xout = x1 + ls2*y 里对 ls2 求导只跟 din 和未缩放的 y 有关），
        # 不能用 dy（dy = din*ls2，会把梯度再乘一次 ls2）
        K['col_sum_prod'](d.q, (MD,), None, b['y'], din, self.dw['ls2'],
                          np.int32(M), np.int32(MD))
        # qk-norm：per-head RMSNorm，行数 = BH*T、列数 = HD
        K['rms_bwd_stats'](d.q, (BH * T,), (64,), b['Qraw'], b['dQ'], self.qw,
                           b['ss_q'], b['sc_q'], np.int32(BH * T), np.int32(HD))
        d.rms_dw(b['Qraw'], b['dQ'], b['ss_q'], self.dw['qw'], BH * T, HD)
        K['rms_bwd_stats'](d.q, (BH * T,), (64,), b['Kraw'], b['dK'], self.kw,
                           b['ss_q'], b['sc_q'], np.int32(BH * T), np.int32(HD))
        d.rms_dw(b['Kraw'], b['dK'], b['ss_q'], self.dw['kw'], BH * T, HD)

    def _small_buf(self, k):
        return {'ln1': self.ln1, 'ln2': self.ln2, 'qw': self.qw, 'kw': self.kw,
                'ls1': self.ls1, 'ls2': self.ls2}[k]

    def _upload_small(self, k, sync=True):
        d = self.dev
        v = self.host[k]
        arr = v.astype(np.float16) if k in ('ln1', 'ln2') else v.astype(np.float32)
        cl.enqueue_copy(d.q, self._small_buf(k), np.ascontiguousarray(arr))
        if sync:
            d.q.finish()

    def grad_items(self):
        """返回 [(buf, n)] 供 Model 一次性归约"""
        d = self.dev
        d.activate()
        items = [(self.W[k].grad, self.W[k].n) for k in ('Wqkv', 'Wo', 'W12', 'W3')]
        return items

    def grad_norm(self):
        d = self.dev
        d.activate()
        items = self.grad_items()
        vals = d.sumsq_many(items)
        tot = float(sum(vals))
        for k, buf in self.dw.items():
            tot += d.sumsq_f32(buf, len(self.host[k]))
        return tot

    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, clip=1.0):
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)
        grads = {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                 for k, v in self.host.items()}
        d.finish()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)
        d.finish()

    # ---- 三段式（供 Model.opt_step 跨层合并同步）----
    def opt_dev(self, lr, b1, b2, eps, wd, bc1, bc2):
        self.dev.activate()
        for k in ('Wqkv', 'Wo', 'W12', 'W3'):
            self.W[k].step(lr, b1, b2, eps, wd, bc1, bc2)

    def small_download(self):
        d = self.dev
        d.activate()
        return {k: d.down(self.dw[k], (v.size,), np.float32, sync=False)
                for k, v in self.host.items()}

    def small_apply(self, grads, lr, b1, b2, eps, wd, bc1, bc2):
        d = self.dev
        d.activate()
        for k, v in self.host.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * g * g
            v -= lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps) - lr * wd * v
            self._upload_small(k, sync=False)


class Head(object):
    """flat head + CE，同样独占一个 context。"""

    def __init__(self, sd, dev=None):
        self.dev = d = dev or Dev()
        Wh = sd['gen_head.lin.weight'].numpy().astype(np.float16)     # (V, MD)
        self.V = Wh.shape[0]
        self.W = Param(d, Wh.T.copy())                                 # (MD, V)
        self.logits = d.buf(M * self.V * 2)
        self.dlogits = d.buf(M * self.V * 2)
        self.loss_row = d.buf(M * 4)
        self.tgt = d.buf(M * 4)
        self.msk = d.buf(M * 4)
        self.dx = d.buf(M * MD * 2)
        self._xb = d.buf(M * MD * 2)      # 预分配，别每步从 arena 里新拿（bump 分配器不回收）

    def forward(self, x_host):
        d = self.dev
        d.activate()
        xb = self._xb
        cl.enqueue_copy(d.q, xb, np.ascontiguousarray(x_host))
        L.hgemm_nn(xb, self.W.w, self.logits, M, self.V, MD)
        d.finish()
        return xb

    def loss(self, tokens):
        d = self.dev
        d.activate()
        tgt = np.zeros(M, np.int64)
        msk = np.zeros(M, np.float32)
        for b_ in range(B):
            r0 = b_ * T
            tgt[r0:r0 + T - 1] = tokens[b_, 1:]
            msk[r0:r0 + T - 1] = 1.0
        inv_cnt = 1.0 / float(msk.sum())
        cl.enqueue_copy(d.q, self.tgt, tgt.astype(np.int32), is_blocking=True)
        cl.enqueue_copy(d.q, self.msk, msk, is_blocking=True)
        d.K['ce_softmax'](d.q, (M * 256,), (256,), self.logits, self.tgt, self.msk,
                          self.dlogits, self.loss_row, np.int32(self.V),
                          np.float32(inv_cnt), np.float32(CE_SCALE))
        d.finish()
        return float(d.down(self.loss_row, (M,), np.float32).sum())

    def grad_norm(self):
        self.dev.activate()
        return self.dev.sumsq(self.W.grad, self.W.n)

    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.dev.activate()
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)

    def opt_dev(self, lr, b1, b2, eps, wd, bc1, bc2):
        self.dev.activate()
        self.W.step(lr, b1, b2, eps, wd, bc1, bc2)

    def backward(self):
        d = self.dev
        d.activate()
        L.hgemm_tn(self._xb, self.dlogits, self.W.grad, MD, self.V, M)
        L.hgemm_nt(self.dlogits, self.W.w, self.dx, M, MD, self.V)
        d.finish()
        return d.down(self.dx, (M, MD), np.float16)


class Model(object):
    """整机：host 侧 embedding（token/2D pos）+ 每层一个 context + head context。"""

    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ck['cfg']
        assert cfg['cond_mode'] == 'none' and cfg['pos_mode'] == '2d'
        assert cfg['head_mode'] == 'flat' and not cfg['use_bos'] and cfg['token_order'] == 'raster'
        assert cfg['grid'] * cfg['grid'] == T, 'grid %d != T %d' % (cfg['grid'], T)
        sd = ck['model']
        self.ck = ck
        self.cfg = cfg
        self.n_layer = int(cfg['n_layer'])
        self.dev = Dev()
        self.blocks = [Block(sd, i, self.dev) for i in range(self.n_layer)]
        self.head = Head(sd, self.dev)
        V = self.head.V
        g = cfg['grid']
        self.tok_tab = sd['token_emb.weight'].numpy().astype(np.float32)
        self.row_tab = sd['row_emb'].numpy().astype(np.float32)
        self.col_tab = sd['col_emb'].numpy().astype(np.float32)
        ordr = sd['order'].numpy() if 'order' in sd else np.arange(g * g)
        self.row_idx = (ordr // g).astype(np.int64)[:T]
        self.col_idx = (ordr % g).astype(np.int64)[:T]
        self.pos_tab = (self.row_tab[self.row_idx] + self.col_tab[self.col_idx])
        self.row_idx_full = np.tile(self.row_idx, B)
        self.col_idx_full = np.tile(self.col_idx, B)
        self.tok_m = np.zeros_like(self.tok_tab); self.tok_v = np.zeros_like(self.tok_tab)
        self.row_m = np.zeros_like(self.row_tab); self.row_v = np.zeros_like(self.row_tab)
        self.col_m = np.zeros_like(self.col_tab); self.col_v = np.zeros_like(self.col_tab)
        self.last_loss = None

    # ---------------- 前向 ----------------
    def embed(self, tokens):
        ids = tokens.reshape(-1)
        return (self.tok_tab[ids] + np.tile(self.pos_tab, (B, 1))).astype(np.float16)

    def forward(self, tokens):
        d = self.dev
        d.activate()
        x = self.embed(tokens)
        cl.enqueue_copy(d.q, self.blocks[0].b['xin'], x)      # 唯一一次 host->device
        for i, blk in enumerate(self.blocks):
            blk.fwd_body()
            if i + 1 < len(self.blocks):                      # 层间 device->device
                cl.enqueue_copy(d.q, self.blocks[i + 1].b['xin'], blk.b['xout'])
        x = d.down(self.blocks[-1].b['xout'], (M, MD), np.float16)   # 唯一一次同步
        self.head.forward(x)
        self.last_loss = self.head.loss(tokens)
        self.x_last = x
        return self.last_loss

    # ---------------- 反向 ----------------
    def backward(self, tokens):
        d = self.dev
        d.activate()
        dx = self.head.backward()
        blocks = self.blocks
        cl.enqueue_copy(d.q, blocks[-1].b['din_a'], np.ascontiguousarray(dx))  # 唯一一次上传
        for i in range(len(blocks) - 1, -1, -1):
            blocks[i].bwd_body()
            if i > 0:                                          # 层间 device->device
                cl.enqueue_copy(d.q, blocks[i - 1].b['din_a'], blocks[i].b['dx'])
        dx = d.down(blocks[0].b['dx'], (M, MD), np.float16)
        # embedding 梯度
        d = dx.astype(np.float32)
        ids = tokens.reshape(-1)
        self.dtok = np.zeros_like(self.tok_tab); np.add.at(self.dtok, ids, d)
        self.drow = np.zeros_like(self.row_tab); np.add.at(self.drow, self.row_idx_full, d)
        self.dcol = np.zeros_like(self.col_tab); np.add.at(self.dcol, self.col_idx_full, d)
        return dx

    # ---------------- 优化器 ----------------
    def grad_norm(self):
        d = self.dev
        d.activate()
        items = []
        for blk in self.blocks:
            items.extend(blk.grad_items())
        items.append((self.head.W.grad, self.head.W.n))
        tot = float(sum(d.sumsq_many(items)))
        for blk in self.blocks:                     # 小向量（float32，量很小）
            for k, buf in blk.dw.items():
                tot += d.sumsq_f32(buf, len(blk.host[k]))
        # embedding（host）
        for gg in (self.dtok, self.drow, self.dcol):
            tot += float((gg.astype(np.float32) ** 2).sum())
        return float(np.sqrt(tot)) / CE_SCALE

    def clip_grads(self, f):
        if not np.isfinite(f):
            return
        for blk in self.blocks:
            d = blk.dev
            d.activate()
            for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                p = blk.W[k]
                d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
            for k, buf in blk.dw.items():
                d.K['scale_float'](d.q, (blk.host[k].size,), None, buf, np.float32(f),
                                   np.int32(blk.host[k].size))
            d.finish()
        d = self.head.dev
        d.activate()
        p = self.head.W
        d.K['scale_half'](d.q, (p.n,), None, p.grad, np.float32(f), np.int32(p.n))
        d.finish()
        for gg in (self.dtok, self.drow, self.dcol):
            gg *= f

    def opt_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, max_norm=1.0):
        gn = self.grad_norm()
        if not np.isfinite(gn):
            # inf/nan 梯度：整步跳过（grad 每步重算覆盖，跳过不会被带进下一步）
            return float(gn)
        if max_norm and gn > max_norm:
            self.clip_grads(np.float32(max_norm / (gn + 1e-8)))
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        d = self.dev
        d.activate()
        # ① 全部 device AdamW（40 个 launch，不插同步）
        for blk in self.blocks:
            blk.opt_dev(lr, b1, b2, eps, wd, bc1, bc2)
        self.head.opt_dev(lr, b1, b2, eps, wd, bc1, bc2)
        # ② 所有小向量梯度一次下载
        grads = [blk.small_download() for blk in self.blocks]
        d.finish()
        # ③ host AdamW + 一次上传
        for blk, g in zip(self.blocks, grads):
            blk.small_apply(g, lr, b1, b2, eps, wd, bc1, bc2)
        d.finish()
        # embedding：host numpy AdamW
        for tab, g, m, v in ((self.tok_tab, self.dtok, self.tok_m, self.tok_v),
                             (self.row_tab, self.drow, self.row_m, self.row_v),
                             (self.col_tab, self.dcol, self.col_m, self.col_v)):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            tab -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps) - lr * wd * tab
        self.pos_tab = self.row_tab[self.row_idx] + self.col_tab[self.col_idx]
        return gn
