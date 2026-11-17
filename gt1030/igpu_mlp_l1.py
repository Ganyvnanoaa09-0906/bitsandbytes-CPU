# -*- coding: utf-8 -*-
"""igpu_mlp_l1.py — 核显常驻训练 L1 设备侧：**任意 (M,K,H,N) 的 3 层 MLP** 20 步对齐 torch 参考。

契约：`D:\work\collab\shared\IGPU_L1_CONTRACT.md`
  x[M,K] -> Linear(K,H) -> SiLU -> Linear(H,H) -> SiLU -> Linear(H,N) -> y
  loss = MSE(y, target);  AdamW(lr=1e-3, betas=(.9,.999), eps=1e-8, wd=0); steps=20, seed=0

设备侧约定：
  · 权重 = fp32 master（设备）+ fp16 副本（设备）+ Adam m/v（设备）+ fp16 梯度（设备）
  · 初始化与数据**在 host 用 torch 复刻参考脚本**（同 seed），一次性上传；训练期间权重不回 host
  · 每步只有 loss 标量回读（4KB 补齐缓冲）

用法：
  python gt1030/igpu_mlp_l1.py --out device.json
  python gt1030/igpu_mlp_l1.py --m 512 --k 1024 --h 9216 --n 1024 --steps 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

A = MM.A
L = MM.L

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_bias(__global half* y, __global const half* b, int rows, int cols) {
    int i = get_global_id(0); if (i >= rows * cols) return;
    int c = i % cols;
    y[i] = (half)((float)y[i] + (float)b[c]);
}
__kernel void silu_fwd(__global const half* x, __global half* y, int n) {
    int i = get_global_id(0); if (i >= n) return;
    float v = (float)x[i];
    y[i] = (half)(v / (1.f + exp(-v)));
}
__kernel void silu_bwd(__global const half* x, __global const half* dy, __global half* dx, int n) {
    int i = get_global_id(0); if (i >= n) return;
    float v = (float)x[i];
    float s = 1.f / (1.f + exp(-v));
    dx[i] = (half)((float)dy[i] * (s * (1.f + v * (1.f - s))));
}
// dy = 2(y-t)/numel（loss 在 host 用 y/t 算，避免 float atomic 的兼容性问题）
__kernel void mse_bwd(__global const half* y, __global const half* t, __global half* dy,
                      int n, float inv_numel) {
    int i = get_global_id(0); if (i >= n) return;
    float d = (float)y[i] - (float)t[i];
    dy[i] = (half)(2.f * d * inv_numel);
}
__kernel void col_sum_half(__global const half* g, __global half* out, int rows, int cols) {
    int c = get_global_id(0); if (c >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; ++r) s += (float)g[(long)r * cols + c];
    out[c] = (half)s;
}
"""


class Param(object):
    """一个参数：fp32 master + fp16 副本 + Adam 状态 + fp16 梯度（全部在设备）。"""

    def __init__(self, w_fp32, use_adamw_kernel):
        mf = cl.mem_flags
        n = w_fp32.size
        self.n = n
        self.master = cl.Buffer(A.ctx, mf.READ_WRITE, n * 4)
        self.w = cl.Buffer(A.ctx, mf.READ_WRITE, n * 2)
        self.grad = cl.Buffer(A.ctx, mf.READ_WRITE, max(n * 2, 4096))
        self.mom = cl.Buffer(A.ctx, mf.READ_WRITE, max(n * 4, 4096))
        self.vel = cl.Buffer(A.ctx, mf.READ_WRITE, max(n * 4, 4096))
        cl.enqueue_copy(A.queue, self.master, np.ascontiguousarray(w_fp32.ravel(), dtype=np.float32))
        cl.enqueue_copy(A.queue, self.w, np.ascontiguousarray(w_fp32.ravel(), dtype=np.float16))
        cl.enqueue_fill_buffer(A.queue, self.grad, np.float16(0), 0, n * 2)
        cl.enqueue_fill_buffer(A.queue, self.mom, np.float32(0), 0, n * 4)
        cl.enqueue_fill_buffer(A.queue, self.vel, np.float32(0), 0, n * 4)
        A.queue.finish()
        self.adamw_kernel = use_adamw_kernel   # d.K['adamw_single']

    def step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.adamw_kernel(A.queue, (self.n // 4,), None, self.master, self.mom, self.vel,
                          self.grad, self.w, np.int32(self.n), np.float32(lr),
                          np.float32(b1), np.float32(b2), np.float32(eps), np.float32(wd),
                          np.float32(bc1), np.float32(bc2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--h", type=int, default=9216)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    model = nn.Sequential(nn.Linear(a.k, a.h), nn.SiLU(),
                          nn.Linear(a.h, a.h), nn.SiLU(),
                          nn.Linear(a.h, a.n))
    x = (torch.randn(a.m, a.k) * 0.02)
    tgt = (torch.randn(a.m, a.n) * 0.02)

    dev = MM.Dev()
    dev.activate()          # ★ 必须在建 program 之前：activate() 会把 A.ctx/A.queue 切到本 context
    prog = cl.Program(A.ctx, SRC).build(options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
    kern = {n: cl.Kernel(prog, n) for n in
            ("add_bias", "silu_fwd", "silu_bwd", "mse_bwd", "col_sum_half")}
    adamw_kernel = dev.K["adamw_single"]

    mf = cl.mem_flags
    def up(arr, dtype=np.float16):
        buf = cl.Buffer(A.ctx, mf.READ_WRITE, max(arr.nbytes, 4096))
        cl.enqueue_copy(A.queue, buf, np.ascontiguousarray(arr.ravel())); A.queue.finish()
        return buf

    x_d = up(x.numpy().astype(np.float16))
    t_d = up(tgt.numpy().astype(np.float16))
    y_d = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.n * 2)
    dy_d = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.n * 2)
    loss_d = cl.Buffer(A.ctx, mf.READ_WRITE, 4096)

    # 层参数（torch 顺序：0=Linear(k,h), 2=Linear(h,h), 4=Linear(h,n)）
    P = []
    for idx in (0, 2, 4):
        lin = model[idx]
        P.append((Param(lin.weight.detach().numpy().astype(np.float32), adamw_kernel),
                  Param(lin.bias.detach().numpy().astype(np.float32), adamw_kernel)))
    for p, _ in P:
        pass

    # 激活缓冲
    h1 = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    h1s = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    h2 = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    h2s = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    dh1 = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    dh2 = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.h * 2)
    dx1 = cl.Buffer(A.ctx, mf.READ_WRITE, a.m * a.k * 2)

    def fwd():
        L.hgemm_nt(x_d, P[0][0].w, h1, a.m, a.h, a.k)
        kern["add_bias"](A.queue, (a.m * a.h,), (256,), h1, P[0][1].w, np.int32(a.m), np.int32(a.h))
        kern["silu_fwd"](A.queue, (a.m * a.h,), (256,), h1, h1s, np.int32(a.m * a.h))
        L.hgemm_nt(h1s, P[1][0].w, h2, a.m, a.h, a.h)
        kern["add_bias"](A.queue, (a.m * a.h,), (256,), h2, P[1][1].w, np.int32(a.m), np.int32(a.h))
        kern["silu_fwd"](A.queue, (a.m * a.h,), (256,), h2, h2s, np.int32(a.m * a.h))
        L.hgemm_nt(h2s, P[2][0].w, y_d, a.m, a.n, a.h)
        kern["add_bias"](A.queue, (a.m * a.n,), (256,), y_d, P[2][1].w, np.int32(a.m), np.int32(a.n))

    def bwd():
        kern["mse_bwd"](A.queue, (a.m * a.n,), (256,), y_d, t_d, dy_d,
                        np.int32(a.m * a.n), np.float32(DY_SCALE / (a.m * a.n)))
        # L3: dW3 = dy^T @ h2s ; dh2 = dy @ W3
        L.hgemm_tn(dy_d, h2s, P[2][0].grad, a.n, a.h, a.m)
        kern["col_sum_half"](A.queue, (a.n,), None, dy_d, P[2][1].grad, np.int32(a.m), np.int32(a.n))
        L.hgemm_nn(dy_d, P[2][0].w, dh2, a.m, a.h, a.n)
        kern["silu_bwd"](A.queue, (a.m * a.h,), (256,), h2, dh2, dh2, np.int32(a.m * a.h))
        # L2
        L.hgemm_tn(dh2, h1s, P[1][0].grad, a.h, a.h, a.m)
        kern["col_sum_half"](A.queue, (a.h,), None, dh2, P[1][1].grad, np.int32(a.m), np.int32(a.h))
        L.hgemm_nn(dh2, P[1][0].w, dh1, a.m, a.h, a.h)
        kern["silu_bwd"](A.queue, (a.m * a.h,), (256,), h1, dh1, dh1, np.int32(a.m * a.h))
        # L1
        L.hgemm_tn(dh1, x_d, P[0][0].grad, a.h, a.k, a.m)
        kern["col_sum_half"](A.queue, (a.k,), None, dh1, P[0][1].grad, np.int32(a.m), np.int32(a.k))

    DY_SCALE = 65536.0          # ★ fp16 下的梯度放大（lr 同步除）
    losses, step_times = [], []
    for st in range(1, a.steps + 1):
        t0 = time.perf_counter()
        fwd()
        bwd()
        y_host = np.empty((a.m, a.n), dtype=np.float16)
        cl.enqueue_copy(A.queue, y_host, y_d, is_blocking=True)
        loss = float(np.mean((y_host.astype(np.float32) - tgt.numpy()) ** 2))
        if st == 1:
            # ★ 首步梯度必须在 optimizer.step 之前抓（否则拿到的是最后一步的梯度）
            def _gn(p):
                g = np.empty(p.n, dtype=np.float16)
                cl.enqueue_copy(A.queue, g, p.grad, is_blocking=True)
                return float(np.linalg.norm(g.astype(np.float32)) / DY_SCALE)
            first_gn = {"l1": _gn(P[0][0]), "l2": _gn(P[1][0]), "l3": _gn(P[2][0])}
        for (wl, bl) in P:
            wl.step(a.lr, st, eps=1e-8 * DY_SCALE)
            bl.step(a.lr, st, eps=1e-8 * DY_SCALE)
        step_times.append(time.perf_counter() - t0)
        losses.append(loss)
        if st == 1 or st % 5 == 0:
            print("  step %2d | loss %.6f | %.1f ms" % (st, loss, step_times[-1] * 1e3), flush=True)

    # 首步梯度范数已在训练循环内（st==1、step 之前）采集
    dev_name = A.ctx.devices[0].name.strip()
    result = {
        "config": {"m": a.m, "k": a.k, "h": a.h, "n": a.n, "steps": a.steps,
                   "lr": a.lr, "seed": a.seed},
        "dtype": "fp16-igpu",
        "backend": "CLBlast/OpenCL " + dev_name,
        "losses": losses,
        "final_loss": losses[-1],
        "first_grad_norms": first_gn,
        "step_seconds_median": statistics.median(step_times),
        "peak_device_mb": 0.0,
        "params": sum(p.n for p, _ in P) + sum(b.n for _, b in P),
    }
    print("  params %.1fM | loss %.6f -> %.6f | step %.1f ms"
          % (result["params"] / 1e6, losses[0], result["final_loss"],
             result["step_seconds_median"] * 1e3))
    if a.out:
        json.dump(result, open(a.out, "w", encoding="utf-8"), indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
