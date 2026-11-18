# -*- coding: utf-8 -*-
"""igpu_attn_full.py — **非因果**（全）注意力：SD1.5 decoder `mid_block.attentions` 需要它。

目前只实现**前向**并与 torch 的 `scaled_dot_product_attention` 对拍；
反向与 FX 分派是下一步（见文件末尾 TODO）。

    python gt1030/igpu_attn_full.py --check

实现要点（避开 batched GEMM）：
  把 (N,C,H,W) 转成 (N*T, C) 的 token 布局（复用 nchw2nhwc），一次算出**整流**的
  S = Q·Kᵀ （N*T, N*T）；softmax 只在**本样本的窗口**内做，窗口外写 0；
  于是 P·V 一次 GEMM 就等价于逐样本计算（因为跨样本块已被清零），省掉 batched GEMM。
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
import gt1030.igpu_conv as CV  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 每个样本占 T 个 token：只在 [s*T, (s+1)*T) 的窗口内做 softmax，其余写 0
__kernel void softmax_window(__global const half* s, __global half* p,
                             int N, int T, float scale) {
    long row = get_global_id(0);
    long rows = (long)N * T;
    if (row >= rows) return;
    int sample = (int)(row / T);
    long base = row * rows + (long)sample * T;   // S 是 (rows, rows)
    float mx = -INFINITY;
    for (int j = 0; j < T; ++j) {
        float v = (float)s[base + j] * scale;
        mx = fmax(mx, v);
    }
    float sum = 0.f;
    for (int j = 0; j < T; ++j) sum += exp((float)s[base + j] * scale - mx);
    float inv = 1.f / sum;
    for (int j = 0; j < T; ++j)
        p[base + j] = (half)(exp((float)s[base + j] * scale - mx) * inv);
}

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in ("softmax_window", "mask_outside")}
    return _C[key]


class FullAttn(object):
    """单头全注意力：(N,C,H,W) → (N,C,H,W)。q/k/v/o 由外部提供（真实权重）。"""

    def __init__(self, dev, n, c, h, w):
        self.dev = dev
        self.N, self.C, self.H, self.W = int(n), int(c), int(h), int(w)
        self.T = self.H * self.W
        self.rows = self.N * self.T
        self.scale = 1.0 / np.sqrt(float(self.C))
        self.q = dev.buf(max(4096, self.C * self.C * 2))
        self.k = dev.buf(max(4096, self.C * self.C * 2))
        self.v = dev.buf(max(4096, self.C * self.C * 2))
        self.o = dev.buf(max(4096, self.C * self.C * 2))
        self.tok = dev.buf(max(4096, self.rows * self.C * 2))
        self.qt = dev.buf(max(4096, self.rows * self.C * 2))
        self.kt = dev.buf(max(4096, self.rows * self.C * 2))
        self.vt = dev.buf(max(4096, self.rows * self.C * 2))
        self.s = dev.buf(max(4096, self.rows * self.rows * 2))
        self.p = dev.buf(max(4096, self.rows * self.rows * 2))
        self.ot = dev.buf(max(4096, self.rows * self.C * 2))
        self.out = dev.buf(max(4096, self.rows * self.C * 2))

    def set_weights(self, q, k, v, o):
        for buf, w in ((self.q, q), (self.k, k), (self.v, v), (self.o, o)):
            cl.enqueue_copy(self.dev.q, buf,
                            np.ascontiguousarray(w.reshape(-1), np.float16))

    def forward(self, x):
        d = self.dev
        N, C, H, W, T, R = self.N, self.C, self.H, self.W, self.T, self.rows
        CV.K(d)["nchw2nhwc"](d.q, (R * C,), None, x, self.tok,
                             np.int32(N), np.int32(C), np.int32(H), np.int32(W))
        GEMM.hgemm_nt(self.tok, self.q, self.qt, R, C, C)
        GEMM.hgemm_nt(self.tok, self.k, self.kt, R, C, C)
        GEMM.hgemm_nt(self.tok, self.v, self.vt, R, C, C)
        GEMM.hgemm_nt(self.qt, self.kt, self.s, R, R, C)   # S = Q Kᵀ（含跨样本块）
        K(d)["softmax_window"](d.q, (R,), None, self.s, self.p,
                               np.int32(N), np.int32(T), np.float32(self.scale))
        GEMM.hgemm_nn(self.p, self.vt, self.ot, R, C, R)   # P·V（跨样本块已清零）
        GEMM.hgemm_nt(self.ot, self.o, self.out, R, C, C)
        CV.K(d)["nhwc2nchw"](d.q, (R * C,), None, self.out, self._dst(),
                             np.int32(N), np.int32(C), np.int32(H), np.int32(W))
        return self._dst()

    _d = None

    def _dst(self):
        if self._d is None:
            self._d = self.dev.buf(max(4096, self.rows * self.C * 2))
        return self._d


def check(n=2, c=32, h=4, w=4, seed=0):
    import torch

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, c, h, w)) * 0.7).astype(np.float32)
    wq, wk, wv, wo = [(rng.standard_normal((c, c)) * 0.1).astype(np.float32)
                      for _ in range(4)]

    dev = CTX.DevCtx(arena_mb=600, name="attnfull")
    dev.activate()
    att = FullAttn(dev, n, c, h, w)
    att.set_weights(wq, wk, wv, wo)
    xd = dev.up(x.astype(np.float16))
    y = att.forward(xd)
    dev.q.finish()
    yv = dev.down(y, (n, c, h, w)).astype(np.float32)

    tx = torch.from_numpy(x)
    tq = tx.flatten(2).transpose(1, 2) @ torch.from_numpy(wq).T
    tk = tx.flatten(2).transpose(1, 2) @ torch.from_numpy(wk).T
    tv = tx.flatten(2).transpose(1, 2) @ torch.from_numpy(wv).T
    to = torch.nn.functional.scaled_dot_product_attention(tq, tk, tv)
    ref = (to @ torch.from_numpy(wo).T).transpose(1, 2).reshape(n, c, h, w).numpy()

    rel = float(np.linalg.norm(yv - ref) / np.linalg.norm(ref))
    print("  全注意力 N%d C%d %dx%d（T=%d）: forward rel = %.2e" % (n, c, h, w, h * w, rel))
    return rel < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    ok &= check(n=2, c=128, h=4, w=4, seed=1)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
