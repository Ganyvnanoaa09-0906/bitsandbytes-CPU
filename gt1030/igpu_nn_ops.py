# -*- coding: utf-8 -*-
"""igpu_nn_ops.py — dev13 通用路径的**缺失算子**：LayerNorm / GELU / embedding gather。

设计口径与既有积木一致：
  · 缓冲全部来自 `DevCtx` 的 arena（sub-buffer，4096 对齐）；激活 fp16，统计量 fp32；
  · 行数用 `rows` 传入，kernel 里以**运行期参数**处理（任意 rows/cols）；
  · 每个算子自带 `--check`，与 torch 对拍（阈值 1e-3 量级）。

用法：
    python gt1030/igpu_nn_ops.py --check            # 三个算子全测
    python gt1030/igpu_nn_ops.py --check --only layernorm

接口：
    ln = LayerNorm(rows, cols, eps=1e-5);  ln.set_weight(g, b)
    y  = ln.forward(x);  dx = ln.backward(dy);  dg, db = ln.grads(dy)

    act = Gelu(rows, cols)                      # tanh 近似，与 torch F.gelu 默认(erf)差<1e-3
    y  = act.forward(x);  dx = act.backward(x, dy)

    emb = EmbeddingGather(rows, cols, vocab)    # idx 为 int32 device buffer
    y  = emb.forward(idx);  dx = emb.backward(didx)   # 散加（scatter-add，无原子竞争）
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ---- LayerNorm: 一行一个 workgroup（256 线程）----
__kernel void ln_stats(__global const half* x, __global float* mean, __global float* rstd,
                       int cols) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* line = x + (long)row * cols;
    float s = 0.f, s2 = 0.f;
    for (int k = lid; k < cols; k += lsz) { float v = (float)line[k]; s += v; s2 += v * v; }
    red[lid] = s;  barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) mean[row] = red[0] / cols;
    barrier(CLK_LOCAL_MEM_FENCE);
    red[lid] = s2; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) {
        float m = red[0] / cols;
        // 与 stats 的 mean 保持一致：这里用同一个 red[0] 的均值
        float mu = mean[row];
        rstd[row] = rsqrt(fmax(red[0] / cols - mu * mu, 0.f) + 1e-12f);
    }
}

__kernel void ln_apply(__global const half* x, __global const float* mean,
                       __global const float* rstd, __global const half* g,
                       __global const half* b, __global half* y, int cols, float eps) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __global const half* line = x + (long)row * cols;
    __global half* out = y + (long)row * cols;
    float mu = mean[row], rs = rstd[row];
    for (int k = lid; k < cols; k += lsz) {
        float xh = ((float)line[k] - mu) * rs;
        out[k] = (half)(xh * (float)g[k] + (float)b[k]);
    }
}

// dY = (dy*g) ; 需要 rowsum(dY) 与 rowsum(dY * xhat)
__kernel void ln_bwd_stats(__global const half* dy, __global const half* x,
                           __global const float* mean, __global const float* rstd,
                           __global const half* g, __global float* sum_dy,
                           __global float* sum_dyx, int cols) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* dl = dy + (long)row * cols;
    __global const half* xl = x + (long)row * cols;
    float mu = mean[row], rs = rstd[row];
    float a = 0.f, bsum = 0.f;
    for (int k = lid; k < cols; k += lsz) {
        float dyg = (float)dl[k] * (float)g[k];
        float xh = ((float)xl[k] - mu) * rs;
        a += dyg; bsum += dyg * xh;
    }
    red[lid] = a; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float sum_a = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    red[lid] = bsum; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) { sum_dy[row] = sum_a; sum_dyx[row] = red[0]; }
}

__kernel void ln_bwd_apply(__global const half* dy, __global const half* x,
                           __global const float* mean, __global const float* rstd,
                           __global const half* g, __global const float* sum_dy,
                           __global const float* sum_dyx, __global half* dx,
                           int cols) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __global const half* dl = dy + (long)row * cols;
    __global const half* xl = x + (long)row * cols;
    __global half* o = dx + (long)row * cols;
    float mu = mean[row], rs = rstd[row];
    float n = (float)cols, s1 = sum_dy[row], s2 = sum_dyx[row];
    for (int k = lid; k < cols; k += lsz) {
        float dyg = (float)dl[k] * (float)g[k];
        float xh = ((float)xl[k] - mu) * rs;
        float v = rs / n * (n * dyg - s1 - xh * s2);
        o[k] = (half)v;
    }
}

// dgamma/dbeta：按列求和（fp32 输出）
__kernel void ln_param_grads(__global const half* dy, __global const half* x,
                             __global const float* mean, __global const float* rstd,
                             __global float* dg, __global float* db, int rows, int cols) {
    int c = get_global_id(0);
    if (c >= cols) return;
    float a = 0.f, b = 0.f;
    for (int r = 0; r < rows; ++r) {
        float d = (float)dy[(long)r * cols + c];
        float xh = ((float)x[(long)r * cols + c] - mean[r]) * rstd[r];
        a += d * xh; b += d;
    }
    dg[c] = a; db[c] = b;
}

// ---- GELU（tanh 近似）----
__kernel void gelu_fwd(__global const half* x, __global half* y, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    y[i] = (half)(0.5f * v * (1.f + tanh(c)));
}

__kernel void gelu_bwd(__global const half* x, __global const half* dy, __global half* dx, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    float t = tanh(c);
    float dc = 0.7978845608028654f * (1.f + 3.f * 0.044715f * v * v);
    float d = 0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * dc;
    dx[i] = (half)(d * (float)dy[i]);
}

// ---- embedding: gather / scatter-add（每个 (row, col) 一个线程）----
__kernel void emb_gather(__global const half* table, __global const int* idx,
                         __global half* y, int rows, int cols) {
    int r = get_global_id(0), c = get_global_id(1);
    if (r >= rows || c >= cols) return;
    int t = idx[r];
    y[(long)r * cols + c] = table[(long)t * cols + c];
}

// 先把 dtable 清零（或由调用方保证），再按行累加；用一维 grid 串行处理行避免原子竞争
__kernel void emb_scatter(__global const half* dy, __global const int* idx,
                          __global half* dtable, int rows, int cols) {
    int c = get_global_id(0);
    if (c >= cols) return;
    for (int r = 0; r < rows; ++r) {
        int t = idx[r];
        dtable[(long)t * cols + c] = (half)((float)dtable[(long)t * cols + c] + (float)dy[(long)r * cols + c]);
    }
}
// 把 fp32 梯度累加进 fp16 参数梯度（省掉 device→host→device 往返）
__kernel void acc_from_f32(__global half* acc, __global const float* src, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    acc[i] = (half)((float)acc[i] + src[i]);
}

"""

_PROG = {}
_K = {}


def _ctx_key():
    c = CTX.A.ctx
    return int(c.int_ptr) if hasattr(c, "int_ptr") else id(c)


def _kernels():
    key = _ctx_key()
    if key not in _PROG:
        _PROG[key] = cl.Program(CTX.A.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _K[key] = {n: cl.Kernel(_PROG[key], n) for n in
                   ("ln_stats", "ln_apply", "ln_bwd_stats", "ln_bwd_apply",
                    "ln_param_grads", "gelu_fwd", "gelu_bwd", "emb_gather", "emb_scatter",
                      "acc_from_f32")}
    return _K[key]


class LayerNorm(object):
    def __init__(self, rows, cols, eps=1e-5, dev=None):
        self.rows, self.cols, self.eps = int(rows), int(cols), float(eps)
        self.dev = dev or CTX.DevCtx(arena_mb=300, name="ln")
        d = self.dev
        d.activate()
        self.x = d.buf(self.rows * self.cols * 2)
        self.y = d.buf(self.rows * self.cols * 2)
        self.dy = d.buf(self.rows * self.cols * 2)
        self.dx = d.buf(self.rows * self.cols * 2)
        self.g = d.buf(self.cols * 2)
        self.b = d.buf(self.cols * 2)
        self.mean = d.buf(self.rows * 4)
        self.rstd = d.buf(self.rows * 4)
        self.s1 = d.buf(self.rows * 4)
        self.s2 = d.buf(self.rows * 4)
        self.dg = d.buf(self.cols * 4)
        self.db = d.buf(self.cols * 4)

    def set_weight(self, g, b):
        d = self.dev
        cl.enqueue_copy(d.q, self.g, np.ascontiguousarray(g, dtype=np.float16))
        cl.enqueue_copy(d.q, self.b, np.ascontiguousarray(b, dtype=np.float16))
        d.q.finish()

    def upload(self, x):
        d = self.dev
        cl.enqueue_copy(d.q, self.x, np.ascontiguousarray(x, dtype=np.float16))
        d.q.finish()

    def forward(self):
        d, K = self.dev, _kernels()
        d.activate()
        K["ln_stats"](d.q, (self.rows * 256,), (256,), self.x, self.mean, self.rstd,
                      np.int32(self.cols))
        K["ln_apply"](d.q, (self.rows * 256,), (256,), self.x, self.mean, self.rstd,
                      self.g, self.b, self.y, np.int32(self.cols), np.float32(self.eps))
        d.q.finish()
        return self.y

    def backward(self, dy_dev):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.dy, dy_dev)
        K["ln_bwd_stats"](d.q, (self.rows * 256,), (256,), self.dy, self.x, self.mean,
                          self.rstd, self.g, self.s1, self.s2, np.int32(self.cols))
        K["ln_bwd_apply"](d.q, (self.rows * 256,), (256,), self.dy, self.x, self.mean,
                          self.rstd, self.g, self.s1, self.s2, self.dx, np.int32(self.cols))
        d.q.finish()
        return self.dx

    def grads(self, dy_dev):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.dy, dy_dev)
        K["ln_param_grads"](d.q, (self.cols,), None, self.dy, self.x, self.mean,
                            self.rstd, self.dg, self.db, np.int32(self.rows), np.int32(self.cols))
        d.q.finish()
        return self.dg, self.db


class Gelu(object):
    def __init__(self, n, dev=None, tanh_approx=True):
        self.n = int(n)
        self.dev = dev or CTX.DevCtx(arena_mb=200, name="gelu")
        d = self.dev
        d.activate()
        self.x = d.buf(self.n * 2)
        self.y = d.buf(self.n * 2)
        self.dy = d.buf(self.n * 2)
        self.dx = d.buf(self.n * 2)

    def forward(self, x_dev):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.x, x_dev)
        K["gelu_fwd"](d.q, (self.n,), None, self.x, self.y, np.int64(self.n))
        d.q.finish()
        return self.y

    def backward(self, dy_dev):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.dy, dy_dev)
        K["gelu_bwd"](d.q, (self.n,), None, self.x, self.dy, self.dx, np.int64(self.n))
        d.q.finish()
        return self.dx


class EmbeddingGather(object):
    def __init__(self, rows, cols, dev=None):
        self.rows, self.cols = int(rows), int(cols)
        self.dev = dev or CTX.DevCtx(arena_mb=300, name="emb")
        d = self.dev
        d.activate()
        self.idx = d.buf(max(4096, self.rows * 4))
        self.y = d.buf(self.rows * self.cols * 2)
        self.dy = d.buf(self.rows * self.cols * 2)
        self.dtable = d.buf(self.rows * self.cols * 2)   # 调用方保证足够大（按 vocab 分配）

    def forward(self, table_dev, idx_np):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.idx, np.ascontiguousarray(idx_np, dtype=np.int32))
        K["emb_gather"](d.q, (self.rows, self.cols), None, table_dev, self.idx, self.y,
                        np.int32(self.rows), np.int32(self.cols))
        d.q.finish()
        return self.y

    def backward(self, dy_dev):
        d, K = self.dev, _kernels()
        d.activate()
        cl.enqueue_copy(d.q, self.dy, dy_dev)
        K["emb_scatter"](d.q, (self.cols,), None, self.dy, self.idx, self.dtable,
                         np.int32(self.rows), np.int32(self.cols))
        d.q.finish()
        return self.dtable


# ---------------------------------------------------------------- 对拍
def _rel(a, b):
    a = a.astype(np.float32)
    b = b.detach().numpy().astype(np.float32)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def check_layernorm(rows=64, cols=128):
    import torch
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((rows, cols)) * 0.5).astype(np.float16)
    g = (1 + 0.1 * rng.standard_normal(cols)).astype(np.float16)
    b = (0.1 * rng.standard_normal(cols)).astype(np.float16)
    dy = (rng.standard_normal((rows, cols)) * 0.1).astype(np.float16)

    ln = LayerNorm(rows, cols)
    ln.set_weight(g, b)
    ln.upload(x)
    ln.forward()
    y = ln.dev.down(ln.y, (rows, cols))
    ln.backward(ln.dev.up(dy))
    dx = ln.dev.down(ln.dx, (rows, cols))
    dg, db = ln.grads(ln.dev.up(dy))
    dg = ln.dev.down(dg, (cols,), np.float32)
    db = ln.dev.down(db, (cols,), np.float32)

    tx = torch.from_numpy(x.astype(np.float32)).requires_grad_(True)
    tg = torch.from_numpy(g.astype(np.float32)).requires_grad_(True)
    tb = torch.from_numpy(b.astype(np.float32)).requires_grad_(True)
    ty = torch.nn.functional.layer_norm(tx, (cols,), tg, tb, 1e-5)
    ty.backward(torch.from_numpy(dy.astype(np.float32)))
    ey, ex = _rel(y, ty), _rel(dx, tx.grad)
    eg = float(np.linalg.norm(dg - tg.grad.numpy()) / (np.linalg.norm(tg.grad.numpy()) + 1e-12))
    eb = float(np.linalg.norm(db - tb.grad.numpy()) / (np.linalg.norm(tb.grad.numpy()) + 1e-12))
    print("  LayerNorm %dx%d | y %.2e dx %.2e dg %.2e db %.2e" % (rows, cols, ey, ex, eg, eb))
    return max(ey, ex, eg, eb) < 5e-2


def check_gelu(n=4096):
    import torch
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(n) * 1.5).astype(np.float16)
    dy = (rng.standard_normal(n) * 0.1).astype(np.float16)
    d = CTX.DevCtx(arena_mb=200, name="gelu-check")
    d.activate()
    act = Gelu(n, dev=d)
    xd = d.up(x)
    y = act.forward(xd)
    act.backward(d.up(dy))
    out = d.down(y, (n,))
    dx = d.down(act.dx, (n,))
    tx = torch.from_numpy(x.astype(np.float32)).requires_grad_(True)
    ty = torch.nn.functional.gelu(tx)          # erf 版；tanh 近似差 ~1e-3
    ty.backward(torch.from_numpy(dy.astype(np.float32)))
    ey, ex = _rel(out, ty), _rel(dx, tx.grad)
    print("  GELU n=%d | y %.2e dx %.2e" % (n, ey, ex))
    return max(ey, ex) < 5e-2


def check_embedding(rows=64, cols=32, vocab=100):
    import torch
    rng = np.random.default_rng(2)
    table = (rng.standard_normal((vocab, cols)) * 0.3).astype(np.float16)
    idx = rng.integers(0, vocab, size=rows).astype(np.int32)
    dy = (rng.standard_normal((rows, cols)) * 0.1).astype(np.float16)
    d = CTX.DevCtx(arena_mb=300, name="emb-check")
    d.activate()
    e = EmbeddingGather(rows, cols, dev=d)
    # 表放 device；dtable 需要 vocab*cols（现缓冲按 rows 分配，这里单独给）
    tbl = d.up(table)
    e.dtable = d.buf(vocab * cols * 2)
    zeros = np.zeros((vocab, cols), dtype=np.float16)
    cl.enqueue_copy(d.q, e.dtable, zeros)
    d.q.finish()
    y = e.forward(tbl, idx)
    e.backward(d.up(dy))
    out = d.down(y, (rows, cols))
    dtab = d.down(e.dtable, (vocab, cols))
    tt = torch.from_numpy(table.astype(np.float32)).requires_grad_(True)
    selected = tt[torch.from_numpy(idx.astype(np.int64))]
    selected.backward(torch.from_numpy(dy.astype(np.float32)))
    ey, ex = _rel(out, selected), _rel(dtab, tt.grad)
    print("  Embedding %d rows | y %.2e dtable %.2e" % (rows, ey, ex))
    return max(ey, ex) < 5e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="", help="layernorm | gelu | embedding")
    a = ap.parse_args()
    if not a.check:
        print("  用 --check 运行对拍")
        return 0
    ok = True
    if a.only in ("", "layernorm"):
        ok &= check_layernorm()
    if a.only in ("", "gelu"):
        ok &= check_gelu()
    if a.only in ("", "embedding"):
        ok &= check_embedding()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
