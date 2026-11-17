# -*- coding: utf-8 -*-
"""igpu_ops2.py — dev13 P2 追加算子：**bias 广播 / 列求和 / 交叉熵（融合 softmax）**。

和 `igpu_nn_ops` 一样：内核在本 DevCtx 内编译并按 context 缓存；纯设备侧。
供 `igpu_fx` 的 Linear(bias=True) 与 F.cross_entropy 使用。
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// y[r, c] += b[c]
__kernel void bias_add_h(__global half* y, __global const half* b, int rows, int cols) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __global half* line = y + (long)r * cols;
    for (int c = lid; c < cols; c += lsz) line[c] = (half)((float)line[c] + (float)b[c]);
}

// db[c] = sum_r dy[r, c]
__kernel void col_sum_h(__global const half* dy, __global half* db, int rows, int cols) {
    int c = get_global_id(0);
    if (c >= cols) return;
    float a = 0.f;
    for (int r = 0; r < rows; ++r) a += (float)dy[(long)r * cols + c];
    db[c] = (half)a;
}

// 交叉熵：per-row softmax + NLL；同时写出**已缩放**的 logits 梯度
__kernel void ce_fwd(__global const half* logits, __global const int* tgt,
                     __global float* loss_per_row, __global half* grad,
                     int rows, int cols, float inv_n, float scale) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* line = logits + (long)r * cols;
    __global half* gline = grad + (long)r * cols;
    float mx = -1e30f;
    for (int c = lid; c < cols; c += lsz) mx = fmax(mx, (float)line[c]);
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] = fmax(red[lid], red[lid + off]); barrier(CLK_LOCAL_MEM_FENCE); }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.f;
    for (int c = lid; c < cols; c += lsz) s += exp((float)line[c] - m);
    red[lid] = s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float Z = red[0];
    int t = tgt[r];
    if (lid == 0) loss_per_row[r] = (m + log(Z)) - (float)line[t];
    for (int c = lid; c < cols; c += lsz) {
        float p = exp((float)line[c] - m) / Z;
        float onehot = (c == t) ? 1.f : 0.f;
        gline[c] = (half)((p - onehot) * inv_n * scale);
    }
}
"""

_CACHE = {}


def kernels(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _CACHE:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _CACHE[key] = {n: cl.Kernel(prog, n) for n in
                       ("bias_add_h", "col_sum_h", "ce_fwd")}
    return _CACHE[key]


def bias_add(dev, y_buf, b_buf, rows, cols):
    kernels(dev)["bias_add_h"](dev.q, (rows * 256,), (256,), y_buf, b_buf,
                               np.int32(rows), np.int32(cols))


def col_sum(dev, dy_buf, db_buf, rows, cols):
    kernels(dev)["col_sum_h"](dev.q, (cols,), None, dy_buf, db_buf,
                              np.int32(rows), np.int32(cols))


def ce_forward(dev, logits_buf, tgt_buf, loss_buf, grad_buf, rows, cols, inv_n, scale):
    """loss_buf: fp32 (rows,)；grad_buf: fp16 (rows, cols)，已乘 inv_n*scale。"""
    kernels(dev)["ce_fwd"](dev.q, (rows * 256,), (256,), logits_buf, tgt_buf, loss_buf,
                           grad_buf, np.int32(rows), np.int32(cols),
                           np.float32(inv_n), np.float32(scale))


def check(rows=32, cols=1024):
    import io as _io
    import gt1030.igpu_ctx as CTX
    d = CTX.DevCtx(arena_mb=300, name="ops2-check")
    d.activate()
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((rows, cols)) * 2).astype(np.float16)
    tgt = rng.integers(0, cols, size=rows).astype(np.int32)
    lb = d.up(logits)
    tb = d.up(tgt)
    loss = d.buf(max(4096, rows * 4))
    grad = d.buf(max(4096, rows * cols * 2))
    scale = 65536.0
    ce_forward(d, lb, tb, loss, grad, rows, cols, 1.0 / rows, scale)
    d.q.finish()
    l = d.down(loss, (rows,), np.float32)
    g = d.down(grad, (rows, cols)).astype(np.float32) / scale

    import torch
    t = torch.from_numpy(logits.astype(np.float32)).requires_grad_(True)
    l_t = torch.nn.functional.cross_entropy(t, torch.from_numpy(tgt.astype(np.int64)),
                                            reduction="none")
    l_t.mean().backward()
    el = float(np.linalg.norm(l - l_t.detach().numpy()) / (np.linalg.norm(l_t.detach().numpy()) + 1e-12))
    # t.grad 已是 mean 的梯度 = (p - onehot)/rows；设备侧同样乘了 inv_n ⇒ 直接比，不再除
    eg = float(np.linalg.norm(g - t.grad.numpy())
               / (np.linalg.norm(t.grad.numpy()) + 1e-12))
    print("  CE %dx%d | loss rel %.2e | grad rel %.2e" % (rows, cols, el, eg))
    return max(el, eg) < 5e-2


if __name__ == "__main__":
    ok = check()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)
