# -*- coding: utf-8 -*-
"""igpu_attn_cross.py — rectangular cross-attention forward for the UNet path.

    Q: (N, Cq, H, W) latent           -> Rq = N*Tq tokens
    K/V: (N, Tk, Ck) text tokens      -> Rk = N*Tk tokens

Current prototype is forward-only (no backward); it reuses CLBlast Hgemm and
one softmax kernel. S is (Rq, Rk), tiny for the UNet cross-attn (Rq=8192, Rk=154).

    python gt1030/igpu_attn_cross.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl

import gt1030.igpu_ctx as CTX
import gt1030.igpu_conv as CV
from gt1030.ar_v2_multi import L as GEMM

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// S: (Rq, Rk); row belongs to sample s, only K window [s*Tk,(s+1)*Tk) is active.
__kernel void softmax_cross(__global const half* s, __global half* p,
                            int rows, int cols, int tq, int tk, float scale) {
    long row = get_global_id(0);
    if (row >= rows) return;
    int sample = (int)(row / tq);
    long base = row * cols;
    long start = (long)sample * tk;
    float mx = -INFINITY;
    for (int j = 0; j < tk; ++j) mx = fmax(mx, (float)s[base + start + j] * scale);
    float sum = 0.f;
    for (int j = 0; j < tk; ++j) sum += exp((float)s[base + start + j] * scale - mx);
    float inv = 1.f / sum;
    for (int j = 0; j < cols; ++j) {
        if (j >= start && j < start + tk)
            p[base + j] = (half)(exp((float)s[base + j] * scale - mx) * inv);
        else
            p[base + j] = (half)0.f;
    }
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {"softmax_cross": cl.Kernel(prog, "softmax_cross")}
    return _C[key]


class CrossAttn(object):
    def __init__(self, dev, n, cq, ck, h, w, tk):
        self.dev = dev
        self.N, self.Cq, self.Ck = int(n), int(cq), int(ck)
        self.H, self.W, self.Tk = int(h), int(w), int(tk)
        self.Tq = self.H * self.W
        self.Rq, self.Rk = self.N * self.Tq, self.N * self.Tk
        self.scale = 1.0 / np.sqrt(float(cq))
        self.q = dev.buf(max(4096, self.Cq * self.Cq * 2))
        self.k = dev.buf(max(4096, self.Cq * self.Ck * 2))
        self.v = dev.buf(max(4096, self.Cq * self.Ck * 2))
        self.o = dev.buf(max(4096, self.Cq * self.Cq * 2))
        self.qb = dev.buf(max(4096, self.Cq * 2))
        self.kb = dev.buf(max(4096, self.Cq * 2))
        self.vb = dev.buf(max(4096, self.Cq * 2))
        self.ob = dev.buf(max(4096, self.Cq * 2))
        self.xq = dev.buf(max(4096, self.Rq * self.Cq * 2))
        self.xk = dev.buf(max(4096, self.Rk * self.Ck * 2))
        self.qt = dev.buf(max(4096, self.Rq * self.Cq * 2))
        self.kt = dev.buf(max(4096, self.Rk * self.Cq * 2))
        self.vt = dev.buf(max(4096, self.Rk * self.Cq * 2))
        self.s = dev.buf(max(4096, self.Rq * self.Rk * 2))
        self.p = dev.buf(max(4096, self.Rq * self.Rk * 2))
        self.ot = dev.buf(max(4096, self.Rq * self.Cq * 2))
        self.out = dev.buf(max(4096, self.Rq * self.Cq * 2))

    def set_weights(self, q, k, v, o, qb=None, kb=None, vb=None, ob=None):
        for buf, w in ((self.q, q), (self.k, k), (self.v, v), (self.o, o)):
            cl.enqueue_copy(self.dev.q, buf,
                            np.ascontiguousarray(np.asarray(w).reshape(-1), np.float16))
        for buf, b, n in ((self.qb, qb, self.Cq), (self.kb, kb, self.Cq),
                          (self.vb, vb, self.Cq), (self.ob, ob, self.Cq)):
            arr = np.zeros(n, np.float16) if b is None else np.asarray(b, np.float16)
            cl.enqueue_copy(self.dev.q, buf, np.ascontiguousarray(arr.reshape(-1)))

    def forward(self, x_nchw, k_tokens, v_tokens):
        """x_nchw: (N,Cq,H,W) half; k/v_tokens: (N*Tk,Ck) half."""
        d = self.dev
        N, Cq, Ck, Tq, Tk = self.N, self.Cq, self.Ck, self.Tq, self.Tk
        Rq, Rk = self.Rq, self.Rk
        CV.K(d)["nchw2nhwc"](d.q, (Rq * Cq,), None, x_nchw, self.xq,
                             np.int32(N), np.int32(Cq), np.int32(self.H), np.int32(self.W))
        cl.enqueue_copy(d.q, self.xk, k_tokens)
        cl.enqueue_copy(d.q, self.xk, k_tokens)  # keep explicit; v uses same input via caller
        # k and v have the same text source but different weights; copy once to xk
        # and compute both from it.
        GEMM.hgemm_nt(self.xq, self.q, self.qt, Rq, Cq, Cq)
        GEMM.hgemm_nt(self.xk, self.k, self.kt, Rk, Cq, Ck)
        GEMM.hgemm_nt(self.xk, self.v, self.vt, Rk, Cq, Ck)
        for y, b in ((self.qt, self.qb), (self.kt, self.kb), (self.vt, self.vb)):
            CV.K(d)["bias_add_oc"](d.q, (y is self.qt and Rq or Rk, Cq), None,
                                   y, b, np.int32(y is self.qt and Rq or Rk), np.int32(Cq))
        GEMM.hgemm_nt(self.qt, self.kt, self.s, Rq, Rk, Cq)
        K(d)["softmax_cross"](d.q, (Rq,), None, self.s, self.p,
                              np.int32(Rq), np.int32(Rk), np.int32(Tq), np.int32(Tk),
                              np.float32(self.scale))
        GEMM.hgemm_nn(self.p, self.vt, self.ot, Rq, Cq, Rk)
        GEMM.hgemm_nt(self.ot, self.o, self.out, Rq, Cq, Cq)
        CV.K(d)["bias_add_oc"](d.q, (Rq, Cq), None, self.out, self.ob,
                               np.int32(Rq), np.int32(Cq))
        dst = d.buf(max(4096, Rq * Cq * 2))
        CV.K(d)["nhwc2nchw"](d.q, (Rq * Cq,), None, self.out, dst,
                             np.int32(N), np.int32(Cq), np.int32(self.H), np.int32(self.W))
        return dst


def check(n=2, cq=320, ck=768, h=4, w=4, tk=7, seed=0):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    dev = CTX.DevCtx(arena_mb=800, name="cross")
    dev.activate()
    ca = CrossAttn(dev, n, cq, ck, h, w, tk)
    wq = (rng.standard_normal((cq, cq)) * 0.05).astype(np.float32)
    wk = (rng.standard_normal((cq, ck)) * 0.05).astype(np.float32)
    wv = (rng.standard_normal((cq, ck)) * 0.05).astype(np.float32)
    wo = (rng.standard_normal((cq, cq)) * 0.05).astype(np.float32)
    ca.set_weights(wq, wk, wv, wo)
    x = rng.standard_normal((n, cq, h, w)).astype(np.float32)
    txt = rng.standard_normal((n, tk, ck)).astype(np.float32)
    xd = dev.up(x.astype(np.float16))
    txtd = dev.up(txt.astype(np.float16))
    before = dev._off
    import time
    t0 = time.time()
    y = ca.forward(xd, txtd, txtd)
    dev.q.finish()
    yv = dev.down(y, (n, cq, h, w)).astype(np.float32)

    xt = torch.from_numpy(x).flatten(2).transpose(1, 2)
    tt = torch.from_numpy(txt)
    q = xt @ torch.from_numpy(wq).T
    k = tt @ torch.from_numpy(wk).T
    v = tt @ torch.from_numpy(wv).T
    # torch SDPA requires (N, heads, T, d); use 1 head
    o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / np.sqrt(cq))
    ref = (o @ torch.from_numpy(wo).T).transpose(1, 2).reshape(n, cq, h, w).numpy()
    dt = time.time() - t0
    rel = np.linalg.norm(yv - ref) / np.linalg.norm(ref)
    print("  cross N%d Cq%d Ck%d Tq%d Tk%d: rel %.2e · %.2fs · arena +%.0f MB" %
          (n, cq, ck, h * w, tk, rel, dt,
           (dev._off - before) / 1048576))
    return rel < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
