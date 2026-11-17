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

"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in
                   ("row_sumsq", "scale_all", "ns_bm", "ns_ax_plus", "muon_apply",
                    "muon_d", "untranspose_h")}
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




def muon_step(dev, p, grad_buf, lr, wd=0.0, momentum=0.95, ns_steps=5):
    """一步 Muon：momentum(Nesterov) → NS 正交化 → 缩放 → 应用（fp32 master）。

    p 需具备 master/buf/grad/mom( fp32 )与形状；transposed 时内部转置处理。
    """
    k = K(dev)
    m0, n0 = p.shape
    tr = m0 > n0
    m, n = (n0, m0) if tr else (m0, n0)
    x_buf = dev.buf(max(4096, m * n * 2))
    k["muon_d"](dev.q, (m0, n0), None, p.mom, grad_buf, x_buf,
                np.int32(m0), np.int32(n0), np.int32(m), np.int32(n),
                np.float32(momentum), np.int32(1 if tr else 0))
    dev.q.finish()
    ns_device(dev, x_buf, m, n, steps=ns_steps)
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
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
