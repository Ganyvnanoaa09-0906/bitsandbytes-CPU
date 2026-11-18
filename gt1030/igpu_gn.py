# -*- coding: utf-8 -*-
"""igpu_gn.py — GroupNorm（NCHW，分组归一化 + 逐通道仿射），前向与反向都在设备上。

    python gt1030/igpu_gn.py --check      # 与 torch 对拍 fwd/dx/dgamma/dbeta

布局：x (N, C, H, W) 连续；组 g 覆盖通道 [g*Cg, (g+1)*Cg)，每组内按 Cg*H*W 个元素归一化。
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

// 一个 workgroup 处理 (n,g)，长度 L = Cg*HW
__kernel void gn_fwd(__global const half* x, __global const half* gamma,
                     __global const half* beta, __global half* y,
                     __global float* mean_out, __global float* rstd_out,
                     int N, int C, int HW, int G, float eps) {
    int row = get_group_id(0);              // row = n*G + g
    int lid = get_local_id(0), lsz = get_local_size(0);
    int Cg = C / G;
    long L = (long)Cg * HW;
    long base = (long)row * L;
    __local float red[256];
    float acc = 0.f;
    for (long i = lid; i < L; i += lsz) acc += (float)x[base + i];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float mean = red[0] / (float)L;
    barrier(CLK_LOCAL_MEM_FENCE);
    acc = 0.f;
    for (long i = lid; i < L; i += lsz) { float d = (float)x[base + i] - mean; acc += d * d; }
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float rstd = rsqrt(red[0] / (float)L + eps);
    if (lid == 0) { mean_out[row] = mean; rstd_out[row] = rstd; }
    barrier(CLK_LOCAL_MEM_FENCE);
    int g = row % G;
    for (long i = lid; i < L; i += lsz) {
        int c = g * Cg + (int)(i / HW);
        float xh = ((float)x[base + i] - mean) * rstd;
        y[base + i] = (half)(xh * (float)gamma[c] + (float)beta[c]);
    }
}

// 反向：dx（dy 与 x 已知），组内先求两个归约量
__kernel void gn_bwd(__global const half* dy, __global const half* x,
                     __global const half* gamma, __global const float* mean,
                     __global const float* rstd, __global half* dx,
                     int N, int C, int HW, int G) {
    int row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int Cg = C / G;
    long L = (long)Cg * HW;
    long base = (long)row * L;
    float m = mean[row], r = rstd[row];
    int g = row % G;
    __local float r1[256];
    __local float r2[256];
    float a1 = 0.f, a2 = 0.f;
    for (long i = lid; i < L; i += lsz) {
        int c = g * Cg + (int)(i / HW);
        float d = (float)dy[base + i] * (float)gamma[c];
        a1 += d;
        a2 += d * ((float)x[base + i] - m) * r;
    }
    r1[lid] = a1; r2[lid] = a2; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { r1[lid] += r1[lid + off]; r2[lid] += r2[lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float s1 = r1[0], s2 = r2[0];
    for (long i = lid; i < L; i += lsz) {
        int c = g * Cg + (int)(i / HW);
        float d = (float)dy[base + i] * (float)gamma[c];
        float xh = ((float)x[base + i] - m) * r;
        dx[base + i] = (half)((d - (s1 + xh * s2) / (float)L) * r);
    }
}

// dgamma/dbeta：每个通道一个 work-item 遍历所有样本（OpenCL 没有 float 原子加）
__kernel void gn_param_grads(__global const half* dy, __global const half* x,
                             __global const float* mean, __global const float* rstd,
                             __global float* dgamma, __global float* dbeta,
                             int N, int C, int HW, int G) {
    int c = get_global_id(0);
    if (c >= C) return;
    int Cg = C / G;
    int g = c / Cg;
    int c_local = c % Cg;
    long L = (long)Cg * HW;
    float ag = 0.f, ab = 0.f;
    for (int n = 0; n < N; ++n) {
        int row = n * G + g;
        long base = (long)row * L + (long)c_local * HW;
        float m = mean[row], r = rstd[row];
        for (int i = 0; i < HW; ++i) {
            float d = (float)dy[base + i];
            float xh = ((float)x[base + i] - m) * r;
            ag += d * xh;
            ab += d;
        }
    }
    dgamma[c] = ag;
    dbeta[c] = ab;
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in
                   ("gn_fwd", "gn_bwd", "gn_param_grads")}
    return _C[key]


class GroupNorm(object):
    def __init__(self, dev, n, c, h, w, groups, eps=1e-5):
        self.dev = dev
        self.N, self.C, self.H, self.W = int(n), int(c), int(h), int(w)
        self.G = int(groups)
        self.eps = float(eps)
        self.HW = self.H * self.W
        self.rows = self.N * self.G
        self.gamma = dev.buf(max(4096, self.C * 2))
        self.beta = dev.buf(max(4096, self.C * 2))
        self.mean = dev.buf(max(4096, self.rows * 4))
        self.rstd = dev.buf(max(4096, self.rows * 4))
        self.dgamma = dev.buf(max(4096, self.C * 4))
        self.dbeta = dev.buf(max(4096, self.C * 4))

    def forward(self, x, y):
        K(self.dev)["gn_fwd"](self.dev.q, (self.rows * 256,), (256,), x, self.gamma, self.beta,
                              y, self.mean, self.rstd,
                              np.int32(self.N), np.int32(self.C), np.int32(self.HW),
                              np.int32(self.G), np.float32(self.eps))
        return y

    def backward(self, dy, x, dx):
        K(self.dev)["gn_bwd"](self.dev.q, (self.rows * 256,), (256,), dy, x, self.gamma,
                              self.mean, self.rstd, dx,
                              np.int32(self.N), np.int32(self.C), np.int32(self.HW),
                              np.int32(self.G))
        K(self.dev)["gn_param_grads"](self.dev.q, (self.C,), None, dy, x,
                                      self.mean, self.rstd, self.dgamma, self.dbeta,
                                      np.int32(self.N), np.int32(self.C), np.int32(self.HW),
                                      np.int32(self.G))
        return dx


def check(n=2, c=8, h=6, w=6, groups=4, eps=1e-5, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, c, h, w)) * 0.8).astype(np.float32)
    gamma = (rng.standard_normal(c) * 0.2 + 1).astype(np.float32)
    beta = (rng.standard_normal(c) * 0.2).astype(np.float32)
    dy = (rng.standard_normal((n, c, h, w)) * 0.3).astype(np.float32)

    dev = CTX.DevCtx(arena_mb=400, name="gn")
    dev.activate()
    gn = GroupNorm(dev, n, c, h, w, groups, eps)
    xd = dev.up(x.astype(np.float16))
    cl.enqueue_copy(dev.q, gn.gamma, gamma.astype(np.float16))
    cl.enqueue_copy(dev.q, gn.beta, beta.astype(np.float16))
    y = dev.buf(max(4096, x.size * 2))
    dyd = dev.up(dy.astype(np.float16))
    dx = dev.buf(max(4096, x.size * 2))
    gn.forward(xd, y)
    dev.q.finish()
    yv = dev.down(y, (n, c, h, w)).astype(np.float32)
    gn.backward(dyd, xd, dx)
    dev.q.finish()
    dxv = dev.down(dx, (n, c, h, w)).astype(np.float32)
    dgv = dev.down(gn.dgamma, (c,), np.float32)
    dbv = dev.down(gn.dbeta, (c,), np.float32)

    tx = torch.from_numpy(x).requires_grad_(True)
    tg = torch.from_numpy(gamma).requires_grad_(True)
    tb = torch.from_numpy(beta).requires_grad_(True)
    ty = torch.nn.functional.group_norm(tx, groups, tg, tb, eps)
    ty.backward(torch.from_numpy(dy))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    r = (rel(yv, ty.detach().numpy()), rel(dxv, tx.grad.numpy()),
         rel(dgv, tg.grad.numpy()), rel(dbv, tb.grad.numpy()))
    print("  groupnorm N%d C%d HxW%dx%d G%d | fwd %.2e dx %.2e dgamma %.2e dbeta %.2e"
          % (n, c, h, w, groups, *r))
    return max(r) < 3e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    ok &= check(n=1, c=32, h=8, w=8, groups=8)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
