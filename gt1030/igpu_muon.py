# -*- coding: utf-8 -*-
"""igpu_muon.py — dev13：**Muon 优化器的核显实现**（Newton-Schulz 正交化，全设备常驻）。

语义与 CPU 侧 `train_ar_v2.Muon` 一致：
    buf  ← momentum·buf + g
    d     = g + momentum·buf            (Nesterov)
    X     = zeropower_via_newtonschulz5(d)      # 5 次 quintic 迭代
    d    *= sqrt(max(1, m/n))
    w    ← w·(1 - lr·wd) - lr·d

只在 2D 权重上用（bs>1 的情形）；其余参数照旧走 AdamW。
fp32 master 保存权重，正交化用 fp16 计算（GEMM 是 fp16），误差见 `--check`。

    python gt1030/igpu_muon.py --check     # 与 torch 的 NS 对拍（cosine / rel）
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402

SRC = r"""
// NS 输出的 RMS 上限：正交化后的单位方向在热重启时会被 lr 放大，这里限幅

#define BLK32 32
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 1) 归一化：X = G / (||G||_F + eps)  —— 每个 workgroup 处理一行，最后再整体除
__kernel void row_sumsq(__global const half* x, __global float* out, int rows, int cols) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* line = x + (long)r * cols;
    float s = 0.f;
    for (int c = lid; c < cols; c += lsz) { float v = (float)line[c]; s += v * v; }
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) out[r] = red[0];
}

__kernel void scale_all(__global half* x, long n, float s) {
    long i = get_global_id(0);
    if (i < n) x[i] = (half)((float)x[i] * s);
}

// 2) Bm = b·A + c·A²（逐元素）
__kernel void ns_bm(__global const half* A, __global const half* A2, __global half* Bm,
                    long n, float b, float c) {
    long i = get_global_id(0);
    if (i < n) Bm[i] = (half)(b * (float)A[i] + c * (float)A2[i]);
}

// 3) X = a·X + Y（Y 已是 Bm@X）
__kernel void ns_ax_plus(__global half* X, __global const half* Y, long n, float a) {
    long i = get_global_id(0);
    if (i < n) X[i] = (half)(a * (float)X[i] + (float)Y[i]);
}

// 4) 应用：w = w·(1-lr·wd) - lr·scale·X；同步 fp16 副本
__kernel void muon_apply(__global float* w, __global const half* X, __global half* wh,
                         long n, float lr, float wd, float scale) {
    long i = get_global_id(0);
    if (i >= n) return;
    float nw = w[i] * (1.f - lr * wd) - lr * scale * (float)X[i];
    w[i] = nw;
    wh[i] = (half)nw;
}

// momentum 更新并写出 d（可选转置）：mom = μ·mom + g ; d = g + μ·mom
// 两种布局都用同一 kernel，用 step/col 的显式 stride 控制是否转置
__kernel void muon_d(__global float* mom, __global const half* g, __global half* d,
                     int rows, int cols, int out_rows, int out_cols,
                     float mu, int transposed) {
    int r = get_global_id(0), c = get_global_id(1);
    if (r >= rows || c >= cols) return;
    long i = (long)r * cols + c;
    float gt = (float)g[i];
    float m = mu * mom[i] + gt;
    mom[i] = m;
    float dv = gt + mu * m;
    long o = transposed ? ((long)c * out_cols + r) : ((long)r * out_cols + c);
    d[o] = (half)dv;
}

// 反转置写回（X 是 (m,n)，需要落地到 (orig_m, orig_n) 的权重方向）
__kernel void untranspose_h(__global const half* x, __global half* y,
                            int rows, int cols) {
    int r = get_global_id(0), c = get_global_id(1);
    if (r >= rows || c >= cols) return;
    y[(long)c * rows + r] = x[(long)r * cols + c];
}


// ---- 8-bit 动量：mom 为 int8 码，absmax 每 256 元素一个 fp32 ----
__kernel void muon_d_q8(__global char* mom, __global float* amax, __global const half* g,
                        __global half* d, int rows, int cols, int out_cols,
                        float mu, int transposed) {
    int r = get_global_id(0), c = get_local_id(0);
    long i = (long)r * cols + c;
    int blk = (int)(i / BLK32);
    float m = ((float)mom[i] / 127.f) * amax[blk];
    float gt = (float)g[i];
    float nm = mu * m + gt;
    float dv = gt + mu * nm;
    long o = transposed ? ((long)c * out_cols + r) : ((long)r * out_cols + c);
    d[o] = (half)dv;
}

// 重定量化（单独一趟，按 256 元素块求 absmax 后写回 int8）
__kernel void mom_requant8(__global const float* momf, __global char* mom, __global float* amax,
                           int n) {
    int blk = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int base = blk * BLK32;
    __local float red[256];
    float a = 0.f;
    int i = base + lid;
    if (i < n) a = fabs(momf[i]);
    red[lid] = a; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] = fmax(red[lid], red[lid + off]); barrier(CLK_LOCAL_MEM_FENCE); }
    float am = fmax(red[0], 1e-12f);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) amax[blk] = am;
    barrier(CLK_LOCAL_MEM_FENCE);
    am = amax[blk];
    if (i < n) mom[i] = (char)clamp(round(momf[i] / am * 127.f), -127.f, 127.f);
}

// 动量更新到 fp32 临时区（供 requant 用）
__kernel void mom_update_f32(__global const half* g, __global const char* mom,
                             __global const float* amax, __global float* momf,
                             long n, float mu) {
    long i = get_global_id(0);
    if (i >= n) return;
    int blk = (int)(i / BLK32);
    float m = ((float)mom[i] / 127.f) * amax[blk];
    momf[i] = mu * m + (float)g[i];
}


// 4-bit 动量：两个码打一个字节（低半=偶数下标，高半=奇数下标），每 256 元素一个 absmax
inline float nib_get(__global const uchar* mom, long i, float am) {
    uchar b = mom[i >> 1];
    int code = (i & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    return ((float)code - 8.f) / 7.f * am;
}
inline uchar nib_pack(uchar lo, uchar hi) { return (uchar)((hi << 4) | (lo & 0xF)); }

__kernel void muon_d_q4(__global const uchar* mom, __global const float* amax,
                        __global const half* g, __global half* d,
                        int rows, int cols, int out_cols, float mu, int transposed) {
    int r = get_global_id(0), c = get_local_id(0);
    long i = (long)r * cols + c;
    int blk = (int)(i / 256);
    float m = nib_get(mom, i, amax[blk]);
    float gt = (float)g[i];
    float nm = mu * m + gt;
    float dv = gt + mu * nm;
    long o = transposed ? ((long)c * out_cols + r) : ((long)r * out_cols + c);
    d[o] = (half)dv;
}

__kernel void mom_update_q4(__global const half* g, __global const uchar* mom,
                            __global const float* amax, __global float* momf,
                            long n, float mu) {
    long i = get_global_id(0);
    if (i >= n) return;
    int blk = (int)(i / 256);
    momf[i] = mu * nib_get(mom, i, amax[blk]) + (float)g[i];
}

__kernel void mom_requant4(__global const float* momf, __global uchar* mom,
                           __global float* amax, int n) {
    int blk = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int base = blk * 256;
    __local float red[256];
    int i = base + lid;
    float a = 0.f;
    if (i < n) a = fabs(momf[i]);
    red[lid] = a; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] = fmax(red[lid], red[lid + off]); barrier(CLK_LOCAL_MEM_FENCE); }
    float am = fmax(red[0], 1e-12f);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) amax[blk] = am;
    barrier(CLK_LOCAL_MEM_FENCE);
    am = amax[blk];
    if (i >= n) return;
    int code = (int)clamp(round(momf[i] / am * 7.f) + 8.f, 0.f, 15.f);
    uchar b = mom[i >> 1];
    if (i & 1) b = nib_pack(b & 0xF, (uchar)code);
    else b = nib_pack((uchar)code, (b >> 4) & 0xF);
    mom[i >> 1] = b;
}


// NS 输出的逐元素缩放裁剪：先求全矩阵 RMS（单 workgroup 归约），再按需缩放
__kernel void ns_clip(__global half* x, long n, float cap) {
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    float acc = 0.f;
    for (long i = lid; i < n; i += lsz) { float v = (float)x[i]; acc += v * v; }
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    __local float s_scale;
    if (lid == 0) {
        float rms = sqrt(red[0] / (float)n);
        s_scale = (rms > cap && cap > 0.f) ? (cap / rms) : 1.f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float sc = s_scale;
    if (sc < 0.9999f) for (long i = lid; i < n; i += lsz) x[i] = (half)((float)x[i] * sc);
}

"""

NS_RMS_CAP = 1.0

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in
                   ("row_sumsq", "scale_all", "ns_bm", "ns_ax_plus", "muon_apply",
                    "muon_d", "untranspose_h", "muon_d_q8", "mom_requant8",
                    "mom_update_f32", "muon_d_q4", "mom_update_q4",
                    "mom_requant4", "ns_clip")}
    return _C[key]


class MuonParam(object):
    """一个 2D 权重的 Muon 状态（momentum + NS 工作区）。"""

    def __init__(self, dev, w_np, name=""):
        import gt1030.igpu_graph as G
        self.dev, self.name = dev, name
        self.p = G.P(dev, w_np, name=name)          # fp32 master + fp16 副本 + fp16 grad
        m, n = w_np.shape
        self.transposed = m > n                     # 与 CPU 实现一致：st 用 (小, 大)
        self.m, self.n = (n, m) if self.transposed else (m, n)
        self.mom = dev.buf(max(4096, self.p.n * 4))  # fp32 momentum
        cl.enqueue_copy(dev.q, self.mom, np.zeros(self.p.n, np.float32))
        # NS 工作区（fp16）：d/X (m,n)、A/A2/Bm (m,m)、Y (m,n)
        self.x = dev.buf(max(4096, self.m * self.n * 2))
        self.y = dev.buf(max(4096, self.m * self.n * 2))
        self.A = dev.buf(max(4096, self.m * self.m * 2))
        self.A2 = dev.buf(max(4096, self.m * self.m * 2))
        self.Bm = dev.buf(max(4096, self.m * self.m * 2))
        self.ssq = dev.buf(max(4096, self.m * 4))
        dev.q.finish()

def _torch_ns(G, steps=5, eps=1e-7):
    import torch
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    return X.T if tr else X


def ns_device(dev, X_buf, m, n, steps=5, eps=1e-7):
    """在设备上对 X (m,n) 做 NS 正交化（原地）。返回 X_buf。"""
    import gt1030.igpu_graph as G
    k = K(dev)
    # 归一化
    ssq_buf = dev.buf(max(4096, m * 4))
    k["row_sumsq"](dev.q, (m * 256,), (256,), X_buf, ssq_buf, np.int32(m), np.int32(n))
    dev.q.finish()
    ssq = dev.down(ssq_buf, (m,), np.float32)
    norm = float(np.sqrt(ssq.sum())) + eps
    k["scale_all"](dev.q, (m * n,), None, X_buf, np.int64(m * n), np.float32(1.0 / norm))
    a, b, c = 3.4445, -4.7750, 2.0315
    A, A2, Bm, Y = dev.buf(max(4096, m * m * 2)), dev.buf(max(4096, m * m * 2)), \
        dev.buf(max(4096, m * m * 2)), dev.buf(max(4096, m * n * 2))
    for _ in range(steps):
        GEMM.hgemm_nt(X_buf, X_buf, A, m, m, n)          # A = X Xᵀ
        GEMM.hgemm_nn(A, A, A2, m, m, m)                 # A²
        k["ns_bm"](dev.q, (m * m,), None, A, A2, Bm, np.int64(m * m),
                   np.float32(b), np.float32(c))
        GEMM.hgemm_nn(Bm, X_buf, Y, m, n, m)             # Bm X
        k["ns_ax_plus"](dev.q, (m * n,), None, X_buf, Y, np.int64(m * n), np.float32(a))
        dev.q.finish()
    return X_buf




def _mom_buf(dev, p):
    """Muon 专用 fp32 动量：量化状态（state_bits 8/4）时 p.mom 是 absmax，不能拿来当满尺寸动量。"""
    need_full = getattr(p, "state_bits", 32) != 32
    if need_full:
        if getattr(p, "muon_mom", None) is None:
            p.muon_mom = dev.buf(max(4096, p.n * 4))
            cl.enqueue_copy(dev.q, p.muon_mom, np.zeros(p.n, np.float32))
            dev.q.finish()
        return p.muon_mom
    return p.mom


def muon_step(dev, p, grad_buf, lr, wd=0.0, momentum=0.95, ns_steps=5):
    """一步 Muon：momentum(Nesterov) → NS 正交化 → 缩放 → 应用（fp32 master）。

    p 需具备 master/buf/grad/mom( fp32 )与形状；transposed 时内部转置处理。
    """
    k = K(dev)
    m0, n0 = p.shape
    tr = m0 > n0
    m, n = (n0, m0) if tr else (m0, n0)
    x_buf = dev.buf(max(4096, m * n * 2))
    k["muon_d"](dev.q, (m0, n0), None, _mom_buf(dev, p), grad_buf, x_buf,
                np.int32(m0), np.int32(n0), np.int32(m), np.int32(n),
                np.float32(momentum), np.int32(1 if tr else 0))
    dev.q.finish()
    ns_device(dev, x_buf, m, n, steps=ns_steps)
    K(dev)["ns_clip"](dev.q, (256,), (256,), x_buf, np.int64(m * n), np.float32(NS_RMS_CAP))
    dev.q.finish()
    scale = float(max(1.0, m / n)) ** 0.5
    if tr:
        y_buf = dev.buf(max(4096, m0 * n0 * 2))
        k["untranspose_h"](dev.q, (m, n), None, x_buf, y_buf, np.int32(m), np.int32(n))
        dev.q.finish()
        src = y_buf
    else:
        src = x_buf
    k["muon_apply"](dev.q, (m0 * n0,), None, p.master, src, p.buf,
                    np.int64(m0 * n0), np.float32(lr), np.float32(wd), np.float32(scale))
    dev.q.finish()


def check_step(m=32, n=16, seed=0, lr=0.02):
    """单步对拍：与 CPU Muon（torch 实现）比较权重更新后的差异。"""
    import torch
    import gt1030.igpu_fx as FX
    rng = np.random.default_rng(seed)
    w0 = (rng.standard_normal((m, n)) * 0.05).astype(np.float32)
    g0 = (rng.standard_normal((m, n)) * 0.01).astype(np.float16)

    dev = CTX.DevCtx(arena_mb=300, name="muon-step")
    dev.activate()
    p = FX.P(dev, w0, name="w")
    p.mom = dev.buf(max(4096, p.n * 4))
    cl.enqueue_copy(dev.q, p.mom, np.zeros(p.n, np.float32))
    dev.q.finish()
    cl.enqueue_copy(dev.q, p.grad, g0.reshape(-1))
    dev.q.finish()
    muon_step(dev, p, p.grad, lr=lr, wd=0.0, momentum=0.95)
    w_dev = dev.down(p.master, (m, n), np.float32)

    # torch 参考（与 train_ar_v2.Muon 同语义，单步、buf 初值 0）
    d = g0.astype(np.float32) + 0.95 * g0.astype(np.float32)      # buf = g; d = g + μ·buf
    X = _torch_ns(torch.from_numpy(d))
    X = X * max(1.0, X.size(0) / X.size(1)) ** 0.5
    w_ref = w0 - lr * X.numpy()

    rel = float(np.linalg.norm(w_dev - w_ref) / (np.linalg.norm(w_ref) + 1e-12))
    print("  muon step %dx%d | rel %.3e" % (m, n, rel))
    return rel < 5e-2




def alloc_mom_q8(dev, n):
    """给 P 挂上 8-bit 动量：mom8（int8 码）+ amax（每 32 一个 fp32）+ momf（fp32 临时）。"""
    nblocks = (n + 31) // 32
    p_mom = dev.buf(max(4096, n))
    amax = dev.buf(max(4096, nblocks * 4))
    momf = dev.buf(max(4096, n * 4))
    cl.enqueue_copy(dev.q, p_mom, np.zeros(n, np.int8))
    cl.enqueue_copy(dev.q, amax, np.zeros(nblocks, np.float32))
    dev.q.finish()
    return p_mom, amax, momf


def muon_step_q8(dev, p, grad_buf, lr, wd=0.0, momentum=0.95, ns_steps=5):
    """Muon 一步，但动量用 8-bit 存储（权重 master 仍 fp32）。"""
    k = K(dev)
    m0, n0 = p.shape
    tr = m0 > n0
    m, n = (n0, m0) if tr else (m0, n0)
    x_buf = dev.buf(max(4096, m * n * 2))
    k["muon_d_q8"](dev.q, (m0, n0), None, p.mom8, p.amax, grad_buf, x_buf,
                   np.int32(m0), np.int32(n0), np.int32(n),
                   np.float32(momentum), np.int32(1 if tr else 0))
    dev.q.finish()
    ns_device(dev, x_buf, m, n, steps=ns_steps)
    K(dev)["ns_clip"](dev.q, (256,), (256,), x_buf, np.int64(m * n), np.float32(NS_RMS_CAP))
    dev.q.finish()
    scale = float(max(1.0, m / n)) ** 0.5
    if tr:
        y_buf = dev.buf(max(4096, m0 * n0 * 2))
        k["untranspose_h"](dev.q, (m, n), None, x_buf, y_buf, np.int32(m), np.int32(n))
        dev.q.finish()
        src = y_buf
    else:
        src = x_buf
    k["muon_apply"](dev.q, (m0 * n0,), None, p.master, src, p.buf,
                    np.int64(m0 * n0), np.float32(lr), np.float32(wd), np.float32(scale))
    # 动量重定量化
    nblk = (p.n + 31) // 32
    k["mom_update_f32"](dev.q, (p.n,), None, grad_buf, p.mom8, p.amax, p.momf,
                        np.int64(p.n), np.float32(momentum))
    k["mom_requant8"](dev.q, (nblk * 32,), (32,), p.momf, p.mom8, p.amax, np.int32(p.n))
    dev.q.finish()


def check_step_q8(m=32, n=16, seed=0, lr=0.02, steps=5):
    """量化动量 vs fp32 动量：跑 steps 步，比较权重相对差。"""
    import gt1030.igpu_fx as FX
    rng = np.random.default_rng(seed)
    w0 = (rng.standard_normal((m, n)) * 0.05).astype(np.float32)
    grads = [(rng.standard_normal((m, n)) * 0.01) for _ in range(steps)]

    dev = CTX.DevCtx(arena_mb=400, name="muon-q8")
    dev.activate()

    def make():
        p = FX.P(dev, w0, name="w")
        p.mom = dev.buf(max(4096, p.n * 4))
        cl.enqueue_copy(dev.q, p.mom, np.zeros(p.n, np.float32))
        p.mom8, p.amax, p.momf = alloc_mom_q8(dev, p.n)
        dev.q.finish()
        return p

    p_fp, p_q8 = make(), make()
    for g in grads:
        gbuf = dev.up(g.astype(np.float16))
        muon_step(dev, p_fp, gbuf, lr=lr, momentum=0.95)
        muon_step_q8(dev, p_q8, gbuf, lr=lr, momentum=0.95)
    a = dev.down(p_fp.master, (m, n), np.float32)
    b = dev.down(p_q8.master, (m, n), np.float32)
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    print("  muon q8 momentum %dx%d steps=%d | rel %.3e" % (m, n, steps, rel))
    return rel < 5e-2




def alloc_mom_q4(dev, n):
    nblocks = (n + 255) // 256
    mom4 = dev.buf(max(4096, (n + 1) // 2))
    amax = dev.buf(max(4096, nblocks * 4))
    momf = dev.buf(max(4096, n * 4))
    cl.enqueue_copy(dev.q, mom4, np.full((n + 1) // 2, 0x88, np.uint8))   # 0x88 = 两个码都表示 0
    cl.enqueue_copy(dev.q, amax, np.zeros(nblocks, np.float32))
    dev.q.finish()
    return mom4, amax, momf


def muon_step_q4(dev, p, grad_buf, lr, wd=0.0, momentum=0.95, ns_steps=5):
    k = K(dev)
    m0, n0 = p.shape
    tr = m0 > n0
    m, n = (n0, m0) if tr else (m0, n0)
    x_buf = dev.buf(max(4096, m * n * 2))
    k["muon_d_q4"](dev.q, (m0, n0), None, p.mom4, p.amax, grad_buf, x_buf,
                   np.int32(m0), np.int32(n0), np.int32(n),
                   np.float32(momentum), np.int32(1 if tr else 0))
    dev.q.finish()
    ns_device(dev, x_buf, m, n, steps=ns_steps)
    K(dev)["ns_clip"](dev.q, (256,), (256,), x_buf, np.int64(m * n), np.float32(NS_RMS_CAP))
    dev.q.finish()
    scale = float(max(1.0, m / n)) ** 0.5
    if tr:
        y_buf = dev.buf(max(4096, m0 * n0 * 2))
        k["untranspose_h"](dev.q, (m, n), None, x_buf, y_buf, np.int32(m), np.int32(n))
        dev.q.finish()
        src = y_buf
    else:
        src = x_buf
    k["muon_apply"](dev.q, (m0 * n0,), None, p.master, src, p.buf,
                    np.int64(m0 * n0), np.float32(lr), np.float32(wd), np.float32(scale))
    nblk = (p.n + 255) // 256
    k["mom_update_q4"](dev.q, (p.n,), None, grad_buf, p.mom4, p.amax, p.momf,
                       np.int64(p.n), np.float32(momentum))
    k["mom_requant4"](dev.q, (nblk * 256,), (256,), p.momf, p.mom4, p.amax, np.int32(p.n))
    dev.q.finish()


def check_loss(seed=0, steps=10, m=16, n=16, lr=0.02, batch=16, mode="q4"):
    """用 **loss 轨迹** 验收（比权重差更贴近"能不能用"）。"""
    import torch
    import gt1030.igpu_fx as FX
    rng = np.random.default_rng(seed)
    w0 = (rng.standard_normal((m, n)) * 0.2).astype(np.float32)
    xb = (rng.standard_normal((batch, n)) * 0.3).astype(np.float16)
    tb = (rng.standard_normal((batch, m)) * 0.2).astype(np.float32)

    dev = CTX.DevCtx(arena_mb=400, name="muon-loss")
    dev.activate()

    def make():
        p = FX.P(dev, w0, name="w")
        p.mom = dev.buf(max(4096, p.n * 4))
        cl.enqueue_copy(dev.q, p.mom, np.zeros(p.n, np.float32))
        p.mom8, p.amax8, p.momf8 = alloc_mom_q8(dev, p.n)
        p.mom4, p.amax4, p.momf4 = alloc_mom_q4(dev, p.n)
        # 复用名字：P 里 amax/momf 走 q8 的；q4 单独字段
        p.amax = p.amax8
        p.momf = p.momf8
        dev.q.finish()
        return p

    def run(step_fn, label):
        p = make()
        xd = dev.up(xb)
        td = dev.up(tb.astype(np.float16))
        losses = []
        for t in range(1, steps + 1):
            # y = x @ wᵀ  (batch, m)
            y = T_new = _buf = None
            ybuf = dev.buf(max(4096, batch * m * 2))
            GEMM.hgemm_nt(xd, p.buf, ybuf, batch, m, n)
            dev.q.finish()
            yv = dev.down(ybuf, (batch, m)).astype(np.float32)
            losses.append(float(((yv - tb) ** 2).mean()))
            # dy = 2(y-t)/batch*m ; 缩放后写 fp16
            dy = (2.0 / (batch * m)) * (yv - tb) * 65536.0
            dyb = dev.up(dy.astype(np.float16))
            # dW = dyᵀ x  (m, n)
            dwb = dev.buf(max(4096, m * n * 2))
            GEMM.hgemm_tn(dyb, xd, dwb, m, n, batch)
            dev.q.finish()
            step_fn(p, dwb)
        return losses

    base = run(lambda p, g: muon_step(dev, p, g, lr=lr), "fp32")
    q8 = run(lambda p, g: muon_step_q8(dev, p, g, lr=lr), "q8")
    q4 = run(lambda p, g: muon_step_q4(dev, p, g, lr=lr), "q4")
    print("  mode=%s  loss(1) -> loss(%d)" % (mode, steps))
    for name, ls in (("fp32", base), ("q8", q8), ("q4", q4)):
        print("    %-4s %.6f -> %.6f  (Δfirst %.2e, Δlast %.2e)"
              % (name, ls[0], ls[-1], abs(ls[0] - base[0]), abs(ls[-1] - base[-1])))
    rel_last = max(abs(q8[-1] - base[-1]), abs(q4[-1] - base[-1])) / max(abs(base[-1]), 1e-12)
    print("  最大末步相对差 %.2e" % rel_last)
    return rel_last < 5e-2


def check(m=64, n=32, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    G0 = rng.standard_normal((m, n)).astype(np.float16)
    dev = CTX.DevCtx(arena_mb=300, name="muon-check")
    dev.activate()
    Xb = dev.up(G0)
    ns_device(dev, Xb, m, n)
    out = dev.down(Xb, (m, n)).astype(np.float32)
    ref = _torch_ns(torch.from_numpy(G0.astype(np.float32))).numpy()
    rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
    cos = float((out.ravel() @ ref.ravel())
                / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    # 正交性：X Xᵀ ≈ I
    orth = float(np.linalg.norm(out @ out.T - np.eye(m)) / np.sqrt(m))
    print("  NS %dx%d | rel %.3e  cos %.5f  orthErr %.3e" % (m, n, rel, cos, orth))
    return rel < 5e-2 and cos > 0.98


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = True
    for (m, n) in ((64, 32), (32, 64), (128, 128)):
        ok &= check(m, n)
    ok &= check_step(32, 16)
    ok &= check_step(16, 32)
    ok &= check_step_q8(32, 16)
    ok &= check_step_q8(16, 32)
    ok &= check_loss()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
