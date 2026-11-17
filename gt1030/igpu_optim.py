# -*- coding: utf-8 -*-
"""igpu_optim.py — dev13：核显优化器家族扩展（SGD / Lion / Adagrad），状态常驻 device。

约定：每个优化器提供 `step(dev, p, grad_buf, lr, ...)`，p 为 igpu_fx.P（fp32 master + fp16 副本 + fp16 梯度）。
调度（lr/wd/momentum）由调用方传；后续再加 8/4-bit 状态变体。

    python gt1030/igpu_optim.py --check    # SGD / Lion / Adagrad 与 torch 对拍（loss 轨迹）
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

// ---- SGD（momentum / Nesterov / 解耦 wd）----
__kernel void sgd_step(__global float* w, __global float* buf, __global const half* g,
                       __global half* wh, long n, float lr, float mom, float wd, int nesterov) {
    long i = get_global_id(0);
    if (i >= n) return;
    float gt = (float)g[i];
    buf[i] = mom * buf[i] + gt;
    float d = nesterov ? (gt + mom * buf[i]) : buf[i];
    w[i] = w[i] * (1.f - lr * wd) - lr * d;
    wh[i] = (half)w[i];
}

// ---- Lion（sign(momentum 混合)，单状态）----
__kernel void lion_step(__global float* w, __global float* m, __global const half* g,
                        __global half* wh, long n, float lr, float b1, float b2,
                        float wd, float sign_scale) {
    long i = get_global_id(0);
    if (i >= n) return;
    float gt = (float)g[i];
    float upd = b1 * m[i] + (1.f - b1) * gt;      // 用旧 m 的方向
    m[i] = b2 * m[i] + (1.f - b2) * gt;           // 再更新 m
    float s = (upd > 0.f) ? sign_scale : ((upd < 0.f) ? -sign_scale : 0.f);
    w[i] = w[i] * (1.f - lr * wd) - lr * s;
    wh[i] = (half)w[i];
}

// ---- Adagrad（累加平方梯度，单状态）----
__kernel void adagrad_step(__global float* w, __global float* acc, __global const half* g,
                           __global half* wh, long n, float lr, float eps, float wd) {
    long i = get_global_id(0);
    if (i >= n) return;
    float gt = (float)g[i];
    acc[i] += gt * gt;
    w[i] = w[i] * (1.f - lr * wd) - lr * gt / (sqrt(acc[i]) + eps);
    wh[i] = (half)w[i];
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in ("sgd_step", "lion_step", "adagrad_step")}
    return _C[key]


def _state(dev, p, nbytes=4):
    """单状态缓冲（fp32）；p.state 复用。"""
    if getattr(p, "state", None) is None:
        p.state = dev.buf(max(4096, p.n * nbytes))
        cl.enqueue_copy(dev.q, p.state, np.zeros(p.n, np.float32))
        dev.q.finish()
    return p.state


def sgd_step(dev, p, grad_buf, lr, momentum=0.9, wd=0.0, nesterov=True):
    K(dev)["sgd_step"](dev.q, (p.n,), None, p.master, _state(dev, p), grad_buf, p.buf,
                       np.int64(p.n), np.float32(lr), np.float32(momentum),
                       np.float32(wd), np.int32(1 if nesterov else 0))
    dev.q.finish()


def lion_step(dev, p, grad_buf, lr, b1=0.9, b2=0.99, wd=0.0):
    K(dev)["lion_step"](dev.q, (p.n,), None, p.master, _state(dev, p), grad_buf, p.buf,
                        np.int64(p.n), np.float32(lr), np.float32(b1), np.float32(b2),
                        np.float32(wd), np.float32(1.0))
    dev.q.finish()


def adagrad_step(dev, p, grad_buf, lr, eps=1e-8, wd=0.0):
    K(dev)["adagrad_step"](dev.q, (p.n,), None, p.master, _state(dev, p), grad_buf, p.buf,
                           np.int64(p.n), np.float32(lr), np.float32(eps), np.float32(wd))
    dev.q.finish()


def check(kind="sgd", steps=20, n=64, lr=None, seed=0):
    """用 loss 轨迹对拍 torch 的同名优化器（单层线性回归）。"""
    import torch
    import gt1030.ar_v2_multi as MM
    import gt1030.igpu_fx as FX

    GEMM = MM.L
    lr = lr or {"sgd": 0.05, "lion": 1e-3, "adagrad": 0.05}[kind]
    dev = CTX.DevCtx(arena_mb=300, name="opt-" + kind)
    dev.activate()
    rng = np.random.default_rng(seed)
    w0 = (rng.standard_normal((n, n)) * 0.1).astype(np.float32)
    x = (rng.standard_normal((32, n)) * 0.3).astype(np.float16)
    t = (rng.standard_normal((32, n)) * 0.1).astype(np.float32)

    p = FX.P(dev, w0, name="w")
    xd, td = dev.up(x), dev.up(t.astype(np.float16))
    yb = dev.buf(max(4096, 32 * n * 2))
    dyb = dev.buf(max(4096, 32 * n * 2))
    dwb = dev.buf(max(4096, n * n * 2))
    losses = []
    for step in range(1, steps + 1):
        GEMM.hgemm_nt(xd, p.buf, yb, 32, n, n)
        dev.q.finish()
        yv = dev.down(yb, (32, n)).astype(np.float32)
        losses.append(float(((yv - t) ** 2).mean()))
        dy = (2.0 / (32 * n)) * (yv - t) * 65536.0
        cl.enqueue_copy(dev.q, dyb, np.ascontiguousarray(dy, np.float16))
        GEMM.hgemm_tn(dyb, xd, dwb, n, n, 32)
        dev.q.finish()
        if kind == "sgd":
            sgd_step(dev, p, dwb, lr / 65536.0)   # 梯度是缩放过的，SGD 不像 Adam 那样归一化
        elif kind == "lion":
            lion_step(dev, p, dwb, lr)
        else:
            adagrad_step(dev, p, dwb, lr)

    # torch 参考
    tw = torch.from_numpy(w0.copy()).requires_grad_(True)
    xf = torch.from_numpy(x.astype(np.float32))
    tf = torch.from_numpy(t)
    if kind == "sgd":
        opt = torch.optim.SGD([tw], lr=lr, momentum=0.9, nesterov=True)
    elif kind == "lion":
        opt = torch.optim.Lion([tw], lr=lr) if hasattr(torch.optim, "Lion") else None
    else:
        opt = torch.optim.Adagrad([tw], lr=lr, eps=1e-8)
    tl = []
    for step in range(steps):
        if kind == "lion" and opt is None:
            break
        opt.zero_grad()
        loss = ((xf @ tw.T - tf) ** 2).mean()
        loss.backward()
        opt.step()
        tl.append(float(loss.detach()))
    print("  %-8s loss %.6f -> %.6f" % (kind, losses[0], losses[-1]))
    if tl:
        rel = abs(losses[-1] - tl[-1]) / max(abs(tl[-1]), 1e-12)
        print("            torch %.6f -> %.6f | 末步相对差 %.2e" % (tl[0], tl[-1], rel))
        return rel < 5e-2
    print("            (torch 无此优化器，跳过对拍)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = True
    for k in (["sgd", "lion", "adagrad"] if not a.only else [a.only]):
        ok &= check(k)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
