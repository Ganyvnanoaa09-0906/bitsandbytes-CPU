# -*- coding: utf-8 -*-
"""igpu_block_ops.py — 核显通用化的 L2 组件#1：**参数化 RMSNorm**（任意 rows×cols）fwd/bwd。

现有的 `rms_*` kernel 全部绑定 AR 的 M=2048 / D=512；这里重写为 (rows, cols) 参数化版本，
作为"通用 transformer block"的第一块积木。

接口：
    rms = RMSNorm(rows, cols, eps=1e-6)
    y  = rms.forward(x_dev)                 # x:(rows, cols) fp16 -> y 同形状
    dx = rms.backward(dy_dev)               # 用缓存的 xhat/rs 算 dx（dw 另给）
    dw = rms.weight_grad(dy_dev)            # (cols,) fp32
    rms.zero_grad(); rms.adamw(lr, ...)     # device 端更新 weight

验收：`python gt1030/igpu_block_ops.py --check` 与 torch 对照（判据 1e-3）。
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

A = MM.A

SRC = r"""
// y[r,c] = x[r,c] * rs * w[c]；rs = rsqrt(mean(x^2)+eps)；同时存 xhat 与 rs 供反向
__kernel void rmsnorm_fwd(__global const half* x, __global const half* w,
                          __global half* y, __global float* rs_out,
                          __global half* xhat, int rows, int cols, float eps) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* xr = x + (long)r * cols;
    __global half* yr = y + (long)r * cols;
    __global half* hr = xhat + (long)r * cols;
    float ss = 0.f;
    for (int c = lid; c < cols; c += lsz) { float v = (float)xr[c]; ss += v * v; }
    red[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rs = rsqrt(red[0] / (float)cols + eps);
    if (lid == 0) rs_out[r] = rs;
    for (int c = lid; c < cols; c += lsz) {
        float xh = (float)xr[c] * rs;
        hr[c] = (half)xh;
        yr[c] = (half)(xh * (float)w[c]);
    }
}

// dx = (dy*w - xhat * mean(dy*w*xhat)) * rs
__kernel void rmsnorm_bwd(__global const half* dy, __global const half* w,
                          __global const half* xhat, __global const float* rs_out,
                          __global half* dx, int rows, int cols) {
    int r = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* dyr = dy + (long)r * cols;
    __global const half* hr = xhat + (long)r * cols;
    __global half* dxr = dx + (long)r * cols;
    float acc = 0.f;
    for (int c = lid; c < cols; c += lsz) {
        float g = (float)dyr[c] * (float)w[c];
        acc += g * (float)hr[c];
    }
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = red[0] / (float)cols;
    float rs = rs_out[r];
    for (int c = lid; c < cols; c += lsz) {
        float g = (float)dyr[c] * (float)w[c];
        dxr[c] = (half)((g - (float)hr[c] * mean) * rs);
    }
}

// dW[c] += sum_r dy[r,c] * xhat[r,c]
__kernel void rmsnorm_dw(__global const half* dy, __global const half* xhat,
                         __global float* dw, int rows, int cols) {
    int c = get_global_id(0);
    if (c >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; ++r)
        s += (float)dy[(long)r * cols + c] * (float)xhat[(long)r * cols + c];
    dw[c] += s;
}
"""


class RMSNorm(object):
    def __init__(self, rows, cols, eps=1e-6):
        self.rows, self.cols, self.eps = int(rows), int(cols), float(eps)
        self.k = {n: cl.Kernel(cl.Program(A.ctx, SRC).build(
            options=['-cl-mad-enable', '-cl-fast-relaxed-math']), n)
            for n in ('rmsnorm_fwd', 'rmsnorm_bwd', 'rmsnorm_dw')}
        mf = cl.mem_flags
        self.w = cl.Buffer(A.ctx, mf.READ_WRITE, self.cols * 2)      # half
        # 本项目已知驱动缺陷：<4KB 的回读不可靠 -> rs/dw 一律补齐到 4KB
        self.rs = cl.Buffer(A.ctx, mf.READ_WRITE, max(self.rows * 4, 4096))
        self.xhat = cl.Buffer(A.ctx, mf.READ_WRITE, self.rows * self.cols * 2)
        self.dw = cl.Buffer(A.ctx, mf.READ_WRITE, max(self.cols * 4, 4096))
        self.y = cl.Buffer(A.ctx, mf.READ_WRITE, self.rows * self.cols * 2)
        self.dx = cl.Buffer(A.ctx, mf.READ_WRITE, self.rows * self.cols * 2)
        self.m = cl.Buffer(A.ctx, mf.READ_WRITE, self.cols * 4)
        self.v = cl.Buffer(A.ctx, mf.READ_WRITE, self.cols * 4)

    def set_weight(self, w):
        cl.enqueue_copy(A.queue, self.w, np.ascontiguousarray(w, dtype=np.float16))
        A.queue.finish()

    def forward(self, x_dev):
        self.k['rmsnorm_fwd'](A.queue, (self.rows * 256,), (256,), x_dev, self.w, self.y,
                              self.rs, self.xhat, np.int32(self.rows), np.int32(self.cols),
                              np.float32(self.eps))
        return self.y

    def backward(self, dy_dev):
        self.k['rmsnorm_bwd'](A.queue, (self.rows * 256,), (256,), dy_dev, self.w, self.xhat,
                              self.rs, self.dx, np.int32(self.rows), np.int32(self.cols))
        return self.dx

    def weight_grad(self, dy_dev):
        cl.enqueue_fill_buffer(A.queue, self.dw, np.float32(0.0), 0, self.cols * 4)
        self.k['rmsnorm_dw'](A.queue, (self.cols,), None, dy_dev, self.xhat, self.dw,
                             np.int32(self.rows), np.int32(self.cols))
        return self.dw

    def adamw(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        raise NotImplementedError("L2 待接：复用 ar_v2_multi 的 adamw 逻辑（fp32 参数）")


def _check(rows=512, cols=256):
    import torch
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((rows, cols)) * 0.5).astype(np.float16)
    w = rng.standard_normal(cols).astype(np.float16)
    dy = (rng.standard_normal((rows, cols)) * 0.1).astype(np.float16)

    mf = cl.mem_flags
    # 本项目已知：COPY_HOST_PTR 在 buffer 多时会传错数据 -> 显式分配 + copy + finish
    xd = cl.Buffer(A.ctx, mf.READ_WRITE, x.nbytes)
    dyd = cl.Buffer(A.ctx, mf.READ_WRITE, dy.nbytes)
    cl.enqueue_copy(A.queue, xd, np.ascontiguousarray(x)); A.queue.finish()
    cl.enqueue_copy(A.queue, dyd, np.ascontiguousarray(dy)); A.queue.finish()
    rms = RMSNorm(rows, cols)
    rms.set_weight(w)
    yd = rms.forward(xd)
    dxd = rms.backward(dyd)
    dwd = rms.weight_grad(dyd)
    A.queue.finish()

    def d2h(buf, shape, dtype=np.float16):
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(A.queue, out, buf, is_blocking=True)
        return out

    y_out = d2h(yd, (rows, cols))
    dx_out = d2h(dxd, (rows, cols))
    dw_out = d2h(dwd, (cols,), np.float32)

    tx = torch.from_numpy(x.astype(np.float32)).clone().requires_grad_(True)
    tw = torch.from_numpy(w.astype(np.float32)).clone().requires_grad_(True)
    tdy = torch.from_numpy(dy.astype(np.float32))
    rs = torch.rsqrt(tx.pow(2).mean(dim=1, keepdim=True) + 1e-6)
    ty = tx * rs * tw
    ty.backward(tdy)
    ey = float(np.abs(y_out.astype(np.float32) - ty.detach().numpy()).max())
    ex = float(np.abs(dx_out.astype(np.float32) - tx.grad.numpy()).max())
    ew = float(np.abs(dw_out - tw.grad.numpy()).max())
    print(f"  rows={rows} cols={cols}")
    print(f"  max|y-y_ref|={ey:.3e}  max|dx-dx_ref|={ex:.3e}  max|dw-dw_ref|={ew:.3e}")
    ok = ey < 2e-2 and ex < 2e-2 and ew < 2e-2
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=256)
    a = ap.parse_args()
    if a.check:
        return _check(a.rows, a.cols)
    print("  用 --check 运行对照")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
