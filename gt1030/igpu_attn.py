# -*- coding: utf-8 -*-
"""igpu_attn.py — 核显通用化的 L2 组件#2：**参数化 causal attention**（任意 B/T/nh/hd）fwd+bwd。

设计：Q/K/V 各自独立缓冲，布局 (BH, T, HD) 连续，BH = B*nh；不依赖 AR 的融合 QKV 布局。
复用 attention_gpu 的两个通用 softmax kernel（T 作为运行期参数），GEMM 用 CLBlast
strided-batched（自己带 batch 数，不用 attention_gpu 里写死的 BH）。

用法/验收：
    python gt1030/igpu_attn.py --check --b 2 --t 32 --nh 4 --hd 16
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import ctypes  # noqa: E402

import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.attention_gpu as AG  # noqa: E402

A = MM.A


def hgemm_batched_ex(a_dev, b_dev, c_dev, m, n, k, ta, tb, lda, ldb, ldc,
                     a_stride, b_stride, c_stride, nbatches):
    """CLBlast strided-batched hgemm（显式 batch 数，不依赖模块级 BH）。"""
    rc = A.bl.CLBlastHgemmStridedBatched(
        A.ROW, ta, tb, m, n, k, A.ONE,
        ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda, a_stride,
        ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb, b_stride,
        A.ZERO,
        ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc, c_stride,
        int(nbatches), ctypes.byref(A.QP), None)
    if rc != 0:
        raise RuntimeError("HgemmStridedBatched rc=%d m%d n%d k%d" % (rc, m, n, k))


_PROG = None
_K = {}


def _kernels():
    global _PROG
    if _PROG is None:
        _PROG = cl.Program(A.ctx, AG.KERNELS).build(
            options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
        for n in ('softmax_causal', 'softmax_bwd_causal'):
            _K[n] = cl.Kernel(_PROG, n)
    return _K


class Attention(object):
    def __init__(self, b, t, nh, hd, scale=None):
        self.B, self.T, self.NH, self.HD = int(b), int(t), int(nh), int(hd)
        self.BH = self.B * self.NH
        self.scale = float(scale) if scale is not None else 1.0 / np.sqrt(self.HD)
        mf = cl.mem_flags
        nq = self.BH * self.T * self.HD
        ns = self.BH * self.T * self.T
        self.q = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.k = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.v = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.o = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.dq = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.dk = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.dv = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.do = cl.Buffer(A.ctx, mf.READ_WRITE, nq * 2)
        self.s = cl.Buffer(A.ctx, mf.READ_WRITE, ns * 2)
        self.p = cl.Buffer(A.ctx, mf.READ_WRITE, ns * 2)
        self.dp = cl.Buffer(A.ctx, mf.READ_WRITE, ns * 2)
        self.ds = cl.Buffer(A.ctx, mf.READ_WRITE, ns * 2)

    def upload(self, q, k, v):
        for buf, arr in ((self.q, q), (self.k, k), (self.v, v)):
            cl.enqueue_copy(A.queue, buf, np.ascontiguousarray(arr, dtype=np.float16))
        A.queue.finish()

    def download(self, buf, shape):
        out = np.empty(shape, dtype=np.float16)
        cl.enqueue_copy(A.queue, out, buf, is_blocking=True)
        return out

    # ---------------- forward ----------------
    def forward(self):
        BH, T, HD = self.BH, self.T, self.HD
        hgemm_batched_ex(self.q, self.k, self.s, T, T, HD, A.NO, A.YES,
                         HD, HD, T, T * HD, T * HD, T * T, BH)          # S = Q Kᵀ
        _kernels()["softmax_causal"](A.queue, (T * 256, BH), (256, 1), self.s, self.p,
                                     np.int32(T), np.float32(self.scale))  # P = softmax(S·scale)
        hgemm_batched_ex(self.p, self.v, self.o, T, HD, T, A.NO, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)          # O = P V
        return self.o

    # ---------------- backward ----------------
    def backward(self, do_dev=None):
        BH, T, HD = self.BH, self.T, self.HD
        if do_dev is not None:
            cl.enqueue_copy(A.queue, self.do, do_dev) if hasattr(self.do, "shape") else None
            A.queue.finish()
        # dV = Pᵀ dO
        hgemm_batched_ex(self.p, self.do, self.dv, T, HD, T, A.YES, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)
        # dP = dO Vᵀ
        hgemm_batched_ex(self.do, self.v, self.dp, T, T, HD, A.NO, A.YES,
                         HD, HD, T, T * HD, T * HD, T * T, BH)
        # dS = P ⊙ (dP − rowsum(dP⊙P)) · scale
        _kernels()["softmax_bwd_causal"](A.queue, (T * 256, BH), (256, 1), self.p, self.dp,
                                         self.ds, np.int32(T), np.float32(self.scale))
        # dQ = dS K ; dK = dSᵀ Q
        hgemm_batched_ex(self.ds, self.k, self.dq, T, HD, T, A.NO, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)
        hgemm_batched_ex(self.ds, self.q, self.dk, T, HD, T, A.YES, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)
        A.queue.finish()
        return self.dq, self.dk, self.dv


def _check(b=2, t=32, nh=4, hd=16):
    import torch
    BH, D = b * nh, nh * hd
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((b, nh, t, hd)) * 0.3)
    k = (rng.standard_normal((b, nh, t, hd)) * 0.3)
    v = (rng.standard_normal((b, nh, t, hd)) * 0.3)
    do = (rng.standard_normal((b, nh, t, hd)) * 0.1)

    att = Attention(b, t, nh, hd)
    att.upload(q.reshape(BH, t, hd), k.reshape(BH, t, hd), v.reshape(BH, t, hd))
    att.forward()
    o_gpu = att.download(att.o, (BH, t, hd)).reshape(b, nh, t, hd)
    cl.enqueue_copy(A.queue, att.do, np.ascontiguousarray(do.reshape(BH, t, hd), dtype=np.float16))
    A.queue.finish()
    att.backward()
    dq = att.download(att.dq, (BH, t, hd)).reshape(b, nh, t, hd)
    dk = att.download(att.dk, (BH, t, hd)).reshape(b, nh, t, hd)
    dv = att.download(att.dv, (BH, t, hd)).reshape(b, nh, t, hd)

    tq = torch.from_numpy(q.astype(np.float32)).requires_grad_(True)
    tk = torch.from_numpy(k.astype(np.float32)).requires_grad_(True)
    tv = torch.from_numpy(v.astype(np.float32)).requires_grad_(True)
    scale = 1.0 / (hd ** 0.5)
    s = torch.einsum("bhtd,bhud->bhtu", tq, tk) * scale
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("bhtu,bhud->bhtd", p, tv)
    o.backward(torch.from_numpy(do.astype(np.float32)))

    def rel(a, bb):
        a = a.astype(np.float32); bb = bb.numpy()
        return float(np.linalg.norm(a - bb) / (np.linalg.norm(bb) + 1e-12))

    eo = rel(o_gpu, o.detach())
    eq = rel(dq, tq.grad)
    ek = rel(dk, tk.grad)
    ev = rel(dv, tv.grad)
    print("  b=%d t=%d nh=%d hd=%d | relL2  o=%.2e dq=%.2e dk=%.2e dv=%.2e" % (b, t, nh, hd, eo, eq, ek, ev))
    ok = all(x < 5e-2 for x in (eo, eq, ek, ev))
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--hd", type=int, default=16)
    a = ap.parse_args()
    if a.check:
        return _check(a.b, a.t, a.nh, a.hd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
