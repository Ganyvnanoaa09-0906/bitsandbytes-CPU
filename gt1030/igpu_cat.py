# -*- coding: utf-8 -*-
"""igpu_cat.py — 沿通道的 concat / split（UNet 的 skip 连接要用）。

    cat2(dst, a, b, N, C1, C2, H, W)   dst[(N,C1+C2,H,W)] = [a ; b]
    split2(src, a, b, N, C1, C2, H, W) a = src[:, :C1]，b = src[:, C1:]
    反向：split2 的反向就是 cat2（梯度按通道拼回），cat2 的反向就是 split2。

纯拷贝，没有算术 ⇒ 对拍应当是**逐位一致**（rel = 0），比常规 kernel 的门槛更严。

    python gt1030/igpu_cat.py --check
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

// dst 前 C1 通道来自 a，后 C2 通道来自 b（NCHW 连续）
__kernel void cat2(__global const half* a, __global const half* b,
                   __global half* dst,
                   int N, int C1, int C2, int HW) {
    long i = get_global_id(0);
    int C = C1 + C2;
    if (i >= (long)N * C * HW) return;
    long n = i / ((long)C * HW);
    long rest = i % ((long)C * HW);
    int c = (int)(rest / HW);
    long off = rest % HW;
    dst[i] = (c < C1) ? a[n * (long)C1 * HW + c * (long)HW + off]
                      : b[n * (long)C2 * HW + (c - C1) * (long)HW + off];
}

// src(N,C1+C2) → a(N,C1)、b(N,C2)
__kernel void split2(__global const half* src, __global half* a, __global half* b,
                     int N, int C1, int C2, int HW) {
    long i = get_global_id(0);
    int C = C1 + C2;
    if (i >= (long)N * C * HW) return;
    long n = i / ((long)C * HW);
    long rest = i % ((long)C * HW);
    int c = (int)(rest / HW);
    long off = rest % HW;
    if (c < C1) a[n * (long)C1 * HW + c * (long)HW + off] = src[i];
    else b[n * (long)C2 * HW + (c - C1) * (long)HW + off] = src[i];
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in ("cat2", "split2")}
    return _C[key]


def cat2(dev, dst, a, b, N, C1, C2, H, W):
    HW = H * W
    K(dev)["cat2"](dev.q, (N * (C1 + C2) * HW,), None, a, b, dst,
                   np.int32(N), np.int32(C1), np.int32(C2), np.int32(HW))
    return dst


def split2(dev, src, a, b, N, C1, C2, H, W):
    HW = H * W
    K(dev)["split2"](dev.q, (N * (C1 + C2) * HW,), None, src, a, b,
                     np.int32(N), np.int32(C1), np.int32(C2), np.int32(HW))
    return a, b


def check(N=2, C1=3, C2=5, H=4, W=6, seed=0):
    import torch

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((N, C1, H, W)).astype(np.float16)
    b = rng.standard_normal((N, C2, H, W)).astype(np.float16)
    dev = CTX.DevCtx(arena_mb=300, name="cat")
    dev.activate()
    ab, bb = dev.buf(max(4096, a.size * 2)), dev.buf(max(4096, b.size * 2))
    cl.enqueue_copy(dev.q, ab, np.ascontiguousarray(a.reshape(-1)))
    cl.enqueue_copy(dev.q, bb, np.ascontiguousarray(b.reshape(-1)))
    dst = dev.buf(max(4096, (a.size + b.size) * 2))
    cat2(dev, dst, ab, bb, N, C1, C2, H, W)
    dev.q.finish()
    dv = dev.down(dst, (N, C1 + C2, H, W))
    ref = torch.cat([torch.from_numpy(a), torch.from_numpy(b)], dim=1).numpy()
    ok_cat = bool(np.array_equal(dv, ref))

    sa, sb = dev.buf(max(4096, a.size * 2)), dev.buf(max(4096, b.size * 2))
    split2(dev, dst, sa, sb, N, C1, C2, H, W)
    dev.q.finish()
    oka = bool(np.array_equal(dev.down(sa, (N, C1, H, W)), a))
    okb = bool(np.array_equal(dev.down(sb, (N, C2, H, W)), b))
    print("  cat2/split2 N%d C%d+%d %dx%d: cat 逐位一致=%s · split_a=%s · split_b=%s"
          % (N, C1, C2, H, W, ok_cat, oka, okb))
    return ok_cat and oka and okb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    ok &= check(N=1, C1=320, C2=320, H=8, W=8, seed=1)     # UNet skip 的真实量级
    ok &= check(N=2, C1=1, C2=1280, H=4, W=4, seed=2)      # 极不均衡
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
