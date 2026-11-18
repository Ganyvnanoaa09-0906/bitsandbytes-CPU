# -*- coding: utf-8 -*-
"""igpu_cl_ffn_train2.py — 直接用 CLBlast 做 Transformer FFN 的 fwd+bwd（训练步），对比 CPU。

形状：x(bs,dim) → h1(bs,ff) → gelu → h2(bs,dim)。GEMM 全在核显，逐元素在 CPU。
所有缓冲一次分配复用（避免 per-call 分配开销）。转置用 CLBlast 的 trans 标志，不用额外核。
"""
from __future__ import annotations

import ctypes
import json
import sys
import time

import numpy as np
import pyopencl as cl
import torch

DLL = r"D:\work\clblast_build\CLBlast-master\build\clblast.dll"
OUT = r"D:\work\research\bench_results\audio_lab"
ROW, NO, YES = 101, 111, 112
torch.set_num_threads(6)


class GEMM:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx)
        self.bl = ctypes.CDLL(DLL)
        self.bl.CLBlastSgemm.restype = ctypes.c_int
        self.bl.CLBlastSgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.qp = ctypes.c_void_p(int(self.q.int_ptr))
        self.mf = cl.mem_flags
        self.bufs = {}

    def buf(self, key, nbytes):
        b = self.bufs.get(key)
        if b is None or b.size < nbytes:
            b = cl.Buffer(self.ctx, self.mf.READ_WRITE, nbytes)
            self.bufs[key] = b
        return b

    def upl(self, key, arr):
        b = self.buf(key, arr.nbytes)
        cl.enqueue_copy(self.q, b, np.ascontiguousarray(arr, np.float32))
        return b

    def gemm(self, a_key, b_key, c_key, m, n, k, lda, ldb, ldc, ta=NO, tb=NO):
        a = self.buf(a_key, m * k * 4 if ta == NO else k * m * 4)
        b = self.buf(b_key, k * n * 4 if tb == NO else n * k * 4)
        c = self.buf(c_key, m * n * 4)
        rc = self.bl.CLBlastSgemm(ROW, ta, tb, m, n, k, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                                  ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(c.int_ptr)), 0, ldc, self.qp, None)
        if rc:
            raise RuntimeError(f"Sgemm rc={rc}")
        return c

    def down(self, key, shape):
        out = np.empty(shape, np.float32)
        cl.enqueue_copy(self.q, out, self.buf(key, int(np.prod(shape)) * 4)).wait()
        return out


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * c * (1 + 3 * 0.044715 * x ** 2)


def main():
    g = GEMM()
    res = {}
    for dim, ff, bs in ((768, 3072, 32), (1024, 4096, 32), (768, 3072, 128)):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((bs, dim)) * 0.1).astype(np.float32)
        W1 = (rng.standard_normal((dim, ff)) * 0.05).astype(np.float32)
        W2 = (rng.standard_normal((ff, dim)) * 0.05).astype(np.float32)
        tgt = rng.standard_normal((bs, dim)).astype(np.float32)

        g.upl("x", x); g.upl("W1", W1); g.upl("W2", W2); g.upl("tgt", tgt)

        def igpu_step():
            g.gemm("x", "W1", "h1", bs, ff, dim, dim, ff, ff)                 # h1 = x@W1
            h1 = g.down("h1", (bs, ff))
            a = gelu(h1)
            g.upl("a", a)
            g.gemm("a", "W2", "h2", bs, dim, ff, ff, dim, dim)                # h2 = a@W2
            h2 = g.down("h2", (bs, dim))
            diff = h2 - tgt
            dh2 = (2.0 * diff / diff.size).astype(np.float32)
            g.upl("dh2", dh2)
            g.gemm("a", "dh2", "dW2", ff, dim, bs, ff, dim, dim, ta=YES)      # dW2 = a^T@dh2
            g.gemm("dh2", "W2", "da", bs, ff, dim, dim, dim, ff, tb=YES)      # da = dh2@W2^T
            da = g.down("da", (bs, ff))
            dh1 = (da * gelu_grad(h1)).astype(np.float32)
            g.upl("dh1", dh1)
            g.gemm("x", "dh1", "dW1", dim, ff, bs, dim, ff, ff, ta=YES)       # dW1 = x^T@dh1
            return float((diff ** 2).mean())

        # CPU 对照
        W1t = torch.from_numpy(W1.copy()).requires_grad_(True)
        W2t = torch.from_numpy(W2.copy()).requires_grad_(True)
        xt, tgtt = torch.from_numpy(x), torch.from_numpy(tgt)

        def cpu_step():
            h1 = xt @ W1t
            a = torch.nn.functional.gelu(h1)
            h2 = a @ W2t
            loss = ((h2 - tgtt) ** 2).mean()
            grads = torch.autograd.grad(loss, (W1t, W2t))
            with torch.no_grad():
                W1t.data.add_(grads[0], alpha=-1e-3)
                W2t.data.add_(grads[1], alpha=-1e-3)
            return float(loss)

        igpu_step(); cpu_step()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); igpu_step(); ts.append(time.perf_counter() - t)
        gpu = min(ts) * 1e3
        ts = []
        for _ in range(3):
            t = time.perf_counter(); cpu_step(); ts.append(time.perf_counter() - t)
        cpu = min(ts) * 1e3
        key = f"ffn_{dim}x{ff}_bs{bs}"
        res[key] = dict(cpu_ms=round(cpu, 2), igpu_ms=round(gpu, 2), speedup=round(cpu / gpu, 2))
        print(f"  {key:<22} CPU {cpu:7.2f} ms | iGPU {gpu:7.2f} ms | {cpu/gpu:5.2f}×", flush=True)

    with open(rf"{OUT}\cl_ffn_train2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
