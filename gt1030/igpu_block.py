# -*- coding: utf-8 -*-
"""igpu_block.py — 通用 transformer block 装配（L2 的第二块积木）。

分工：LEAD 写 attention kernel（`gt1030/igpu_attn.py`），这里只做
**RMSNorm + attention 接口 + SwiGLU MLP + 残差** 的设备常驻装配，以及
attention 未就绪时的 `attn=None`（零分支）自检。

设计
----
* `Linear` / `MLP` / `Block` 全部持有 device buffer；权重、梯度、AdamW 状态、
  激活都不回 host（`--check` 只在结束时回读比对）。
* attention 只要满足 `forward(x_dev)->dev` / `backward(dy_dev)->dev` /
  `step(lr,t)` / `zero_grad()` 即可接入；`attn=None` 时残差分支为 0。
* 形状全部参数化：`rows`（M）、`d_model`、`hidden`、`heads`（交给 attention）。

自检：
    python gt1030/igpu_block.py --check
    python gt1030/igpu_block.py --check --rows 64 --d 32 --hidden 64
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.igpu_attn as IA  # noqa: E402
from gt1030.igpu_ctx import DevCtx  # noqa: E402
from gt1030.igpu_block_ops import RMSNorm  # noqa: E402
from gt1030.igpu_block_ops import SRC as NORM_SRC  # noqa: E402

A = MM.A

KSRC = r"""
__kernel void zero_half(__global half* x, int n) {
    int i = get_global_id(0);
    if (i < n) x[i] = (half)0;
}
__kernel void add_half(__global const half* a, __global const half* b, __global half* y, int n) {
    int i = get_global_id(0);
    if (i < n) y[i] = (half)((float)a[i] + (float)b[i]);
}
__kernel void bias_add(__global const half* x, __global const float* bias,
                       __global half* y, int rows, int cols) {
    int i = get_global_id(0);
    if (i >= rows * cols) return;
    y[i] = (half)((float)x[i] + bias[i % cols]);
}
__kernel void bias_grad(__global const half* dy, __global float* db, int rows, int cols) {
    int c = get_global_id(0);
    if (c >= cols) return;
    float s = 0.f;
    for (int r = 0; r < rows; ++r) s += (float)dy[(long)r * cols + c];
    db[c] = s;
}
__kernel void silu_gate(__global const half* a, __global const half* b,
                        __global half* g, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)a[i];
    g[i] = (half)(x / (1.f + exp(-x)) * (float)b[i]);
}
__kernel void silu_bwd(__global const half* a, __global const half* b, __global const half* dg,
                       __global half* da, __global half* db, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)a[i];
    float s = 1.f / (1.f + exp(-x));
    float d = (float)dg[i];
    da[i] = (half)(d * (s + x * s * (1.f - s)) * (float)b[i]);
    db[i] = (half)(d * x * s);
}
__kernel void split_qkv(__global const half* x, __global half* q, __global half* k,
                        __global half* v, int B, int T, int NH, int HD) {
    int i = get_global_id(0);
    int total = B * NH * T * HD;
    if (i >= total) return;
    int hd = i % HD;
    int t = (i / HD) % T;
    int bh = i / (HD * T);
    int b = bh / NH;
    int nh = bh % NH;
    int D = NH * HD;
    int src = (b * T + t) * 3 * D + nh * HD + hd;
    q[i] = x[src];
    k[i] = x[src + D];
    v[i] = x[src + 2 * D];
}
__kernel void merge_qkv(__global const half* dq, __global const half* dk,
                        __global const half* dv, __global half* x,
                        int B, int T, int NH, int HD) {
    int i = get_global_id(0);
    int total = B * T * NH * HD;
    if (i >= total) return;
    int hd = i % HD;
    int nh = (i / HD) % NH;
    int t = (i / (HD * NH)) % T;
    int b = i / (HD * NH * T);
    int D = NH * HD;
    int dst = (b * T + t) * 3 * D + nh * HD + hd;
    int src = ((b * NH + nh) * T + t) * HD + hd;
    x[dst] = dq[src];
    x[dst + D] = dk[src];
    x[dst + 2 * D] = dv[src];
}
__kernel void heads_to_tok(__global const half* h, __global half* x,
                           int B, int T, int NH, int HD) {
    int i = get_global_id(0);
    int total = B * T * NH * HD;
    if (i >= total) return;
    int hd = i % HD;
    int nh = (i / HD) % NH;
    int t = (i / (HD * NH)) % T;
    int b = i / (HD * NH * T);
    x[(b * T + t) * (NH * HD) + nh * HD + hd] = h[((b * NH + nh) * T + t) * HD + hd];
}
__kernel void tok_to_heads(__global const half* x, __global half* h,
                           int B, int T, int NH, int HD) {
    int i = get_global_id(0);
    int total = B * T * NH * HD;
    if (i >= total) return;
    int hd = i % HD;
    int nh = (i / HD) % NH;
    int t = (i / (HD * NH)) % T;
    int b = i / (HD * NH * T);
    h[((b * NH + nh) * T + t) * HD + hd] = x[(b * T + t) * (NH * HD) + nh * HD + hd];
}
__kernel void adamw_half(__global half* w, __global const half* g,
                         __global float* m, __global float* v,
                         int n, float lr, float b1, float b2, float eps, float wd,
                         float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float grad = (float)g[i];
    float mm = b1 * m[i] + (1.f - b1) * grad;
    float vv = b2 * v[i] + (1.f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    float wv = (float)w[i] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * (float)w[i];
    w[i] = (half)wv;
}
__kernel void adamw_master(__global half* w, __global float* master, __global const half* g,
                            __global float* m, __global float* v, int n,
                            float lr, float b1, float b2, float eps, float wd,
                            float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float grad = (float)g[i];
    float mm = b1 * m[i] + (1.f - b1) * grad;
    float vv = b2 * v[i] + (1.f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    float wv = master[i] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * master[i];
    master[i] = wv;
    w[i] = (half)wv;
}
__kernel void adamw_norm_master(__global half* w, __global float* master, __global const float* g,
                                 __global float* m, __global float* v, int n,
                                 float lr, float b1, float b2, float eps, float wd,
                                 float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float grad = g[i];
    float mm = b1 * m[i] + (1.f - b1) * grad;
    float vv = b2 * v[i] + (1.f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    float wv = master[i] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * master[i];
    master[i] = wv;
    w[i] = (half)wv;
}
__kernel void adamw_halfgrad(__global half* w, __global const float* g,
                            __global float* m, __global float* v,
                            int n, float lr, float b1, float b2, float eps, float wd,
                            float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float grad = g[i];
    float mm = b1 * m[i] + (1.f - b1) * grad;
    float vv = b2 * v[i] + (1.f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    float wv = (float)w[i] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * (float)w[i];
    w[i] = (half)wv;
}
__kernel void adamw_float(__global float* w, __global const float* g,
                          __global float* m, __global float* v,
                          int n, float lr, float b1, float b2, float eps, float wd,
                          float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    float grad = g[i];
    float mm = b1 * m[i] + (1.f - b1) * grad;
    float vv = b2 * v[i] + (1.f - b2) * grad * grad;
    m[i] = mm;
    v[i] = vv;
    w[i] = w[i] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * w[i];
}
"""

_prog = cl.Program(A.ctx, KSRC).build(
    options=["-cl-mad-enable", "-cl-fast-relaxed-math"]
)
_K = {
    name: cl.Kernel(_prog, name)
    for name in (
        "zero_half",
        "add_half",
        "bias_add",
        "bias_grad",
        "silu_gate",
        "silu_bwd",
        "adamw_half",
        "adamw_halfgrad",
        "adamw_float",
        "adamw_master",
        "adamw_norm_master",
        "split_qkv",
        "merge_qkv",
        "heads_to_tok",
        "tok_to_heads",
    )
}


class LayerCtx(object):
    """Thin wrapper over LEAD's DevCtx: per-layer context + arena + per-context kernels."""

    def __init__(self, arena_mb=400, name="layer", profiling=False):
        self.dev = DevCtx(arena_mb=arena_mb, name=name, profiling=profiling)
        self.ctx, self.queue = self.dev.ctx, self.dev.q
        names = (
            "zero_half",
            "add_half",
            "bias_add",
            "bias_grad",
            "silu_gate",
            "silu_bwd",
            "adamw_half",
            "adamw_halfgrad",
            "adamw_float",
            "adamw_master",
            "adamw_norm_master",
            "split_qkv",
            "merge_qkv",
            "heads_to_tok",
            "tok_to_heads",
        )
        self.K = self.dev.program(KSRC, names)
        self.norm_K = self.dev.program(
            NORM_SRC, ("rmsnorm_fwd", "rmsnorm_bwd", "rmsnorm_dw")
        )

    def activate(self):
        global _K, _CURRENT_DEV
        self.dev.activate()
        _CURRENT_DEV = self.dev
        _K = self.K


_CURRENT_DEV = None


def _buf(nbytes):
    if _CURRENT_DEV is not None:
        return _CURRENT_DEV.buf(nbytes)
    return cl.Buffer(A.ctx, cl.mem_flags.READ_WRITE, max(int(nbytes), 4))


def _upload(arr):
    # Do NOT use COPY_HOST_PTR: with many live buffers this driver silently uploads zeros.
    if _CURRENT_DEV is not None:
        return _CURRENT_DEV.up(np.ascontiguousarray(arr))
    a = np.ascontiguousarray(arr)
    buf = cl.Buffer(A.ctx, cl.mem_flags.READ_WRITE, max(a.nbytes, 4))
    cl.enqueue_copy(A.queue, buf, a)
    A.queue.finish()
    return buf


def _read(buf, shape, dtype=np.float16):
    if _CURRENT_DEV is not None:
        return _CURRENT_DEV.down(buf, shape, dtype)
    out = np.empty(shape, dtype=dtype)
    cl.enqueue_copy(A.queue, out, buf)
    A.queue.finish()
    return out


class Linear(object):
    """y = x @ W^T (+ bias), all device-resident; fp16 params/grads, fp32 AdamW state."""

    def __init__(self, rows, n_in, n_out, bias=True, rng=None):
        self.rows, self.n_in, self.n_out = int(rows), int(n_in), int(n_out)
        rng = rng or np.random.default_rng(0)
        w = (rng.standard_normal((self.n_out, self.n_in)) * 0.02).astype(np.float16)
        self.w = _upload(w)
        self.master = _buf(self.n_out * self.n_in * 4)
        cl.enqueue_copy(
            A.queue, self.master, np.ascontiguousarray(w.astype(np.float32))
        )
        self.dw = _buf(self.n_out * self.n_in * 2)
        self.m = _buf(self.n_out * self.n_in * 4)
        self.v = _buf(self.n_out * self.n_in * 4)
        cl.enqueue_fill_buffer(A.queue, self.m, np.float32(0), 0, self.m.size)
        cl.enqueue_fill_buffer(A.queue, self.v, np.float32(0), 0, self.v.size)
        self.bias = None
        self.db = None
        if bias:
            self.bias = _upload(np.zeros(self.n_out, dtype=np.float32))
            self.db = _buf(self.n_out * 4)
            self.bm = _buf(self.n_out * 4)
            self.bv = _buf(self.n_out * 4)
            for b in (self.bm, self.bv, self.db):
                cl.enqueue_fill_buffer(A.queue, b, np.float32(0), 0, self.n_out * 4)
        self.y = _buf(self.rows * self.n_out * 2)
        self.dx = _buf(self.rows * self.n_in * 2)
        self._x = None
        A.queue.finish()

    def set_weight(self, w_fp32):
        w = np.ascontiguousarray(
            np.asarray(w_fp32, dtype=np.float32).astype(np.float16)
        )
        assert w.shape == (self.n_out, self.n_in), (w.shape, self.n_out, self.n_in)
        cl.enqueue_copy(A.queue, self.w, w)
        cl.enqueue_copy(
            A.queue,
            self.master,
            np.ascontiguousarray(np.asarray(w_fp32, dtype=np.float32)),
        )
        cl.enqueue_fill_buffer(
            A.queue, self.m, np.float32(0), 0, self.n_out * self.n_in * 4
        )
        cl.enqueue_fill_buffer(
            A.queue, self.v, np.float32(0), 0, self.n_out * self.n_in * 4
        )
        A.queue.finish()

    def forward(self, x_dev):
        self._x = x_dev
        MM.L.hgemm_nt(x_dev, self.w, self.y, self.rows, self.n_out, self.n_in)
        if self.bias is not None:
            _K["bias_add"](
                A.queue,
                (self.rows * self.n_out,),
                None,
                self.y,
                self.bias,
                self.y,
                np.int32(self.rows),
                np.int32(self.n_out),
            )
        return self.y

    def backward(self, dy_dev):
        _K["zero_half"](
            A.queue,
            (self.n_out * self.n_in,),
            None,
            self.dw,
            np.int32(self.n_out * self.n_in),
        )
        MM.L.hgemm_tn(dy_dev, self._x, self.dw, self.n_out, self.n_in, self.rows)
        MM.L.hgemm_nn(dy_dev, self.w, self.dx, self.rows, self.n_in, self.n_out)
        if self.bias is not None:
            _K["bias_grad"](
                A.queue,
                (self.n_out,),
                None,
                dy_dev,
                self.db,
                np.int32(self.rows),
                np.int32(self.n_out),
            )
        return self.dx

    def step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        n = self.n_out * self.n_in
        bc1 = 1.0 - b1**t
        bc2 = 1.0 - b2**t
        _K["adamw_master"](
            A.queue,
            ((n + 255) // 256 * 256,),
            (256,),
            self.w,
            self.master,
            self.dw,
            self.m,
            self.v,
            np.int32(n),
            np.float32(lr),
            np.float32(b1),
            np.float32(b2),
            np.float32(eps),
            np.float32(wd),
            np.float32(bc1),
            np.float32(bc2),
        )
        if self.bias is not None:
            _K["adamw_float"](
                A.queue,
                (self.n_out,),
                None,
                self.bias,
                self.db,
                self.bm,
                self.bv,
                np.int32(self.n_out),
                np.float32(lr),
                np.float32(b1),
                np.float32(b2),
                np.float32(eps),
                np.float32(wd),
                np.float32(bc1),
                np.float32(bc2),
            )


class MLP(object):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, rows, d_model, hidden, rng=None):
        self.rows, self.d_model, self.hidden = int(rows), int(d_model), int(hidden)
        rng = rng or np.random.default_rng(1)
        self.gate = Linear(rows, d_model, hidden, bias=False, rng=rng)
        self.up = Linear(rows, d_model, hidden, bias=False, rng=rng)
        self.down = Linear(rows, hidden, d_model, bias=False, rng=rng)
        self.a = self.gate.y
        self.b = self.up.y
        self.g = _buf(rows * hidden * 2)
        self.da = _buf(rows * hidden * 2)
        self.db = _buf(rows * hidden * 2)
        self.dx = _buf(rows * d_model * 2)

    def forward(self, x_dev):
        self.gate.forward(x_dev)
        self.up.forward(x_dev)
        _K["silu_gate"](
            A.queue,
            (self.rows * self.hidden,),
            None,
            self.a,
            self.b,
            self.g,
            np.int32(self.rows * self.hidden),
        )
        self._last_y = self.down.forward(self.g)
        return self._last_y

    def backward(self, dy_dev):
        self.down.backward(dy_dev)
        _K["silu_bwd"](
            A.queue,
            (self.rows * self.hidden,),
            None,
            self.a,
            self.b,
            self.down.dx,
            self.da,
            self.db,
            np.int32(self.rows * self.hidden),
        )
        self.gate.backward(self.da)
        self.up.backward(self.db)
        _K["add_half"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            self.gate.dx,
            self.up.dx,
            self.dx,
            np.int32(self.rows * self.d_model),
        )
        return self.dx

    def step(self, lr, t, eps=1e-8):
        for m in (self.gate, self.up, self.down):
            m.step(lr, t, eps=eps)


class SelfAttention(object):
    """QKV/out projections + LEAD's causal attention kernel, assembled into the Block interface.

    The attention kernel (`gt1030/igpu_attn.py`) takes host q/k/v in (B*NH, T, HD) layout;
    this adapter does the (B,T,NH,HD) <-> (B*NH,T,HD) reshaping on host. That is a first
    integration: QKV is 3*d_model*2 bytes per token (e.g. 6 MB at M=2048,D=512), still far
    below the model state, but the follow-up should keep the reshape on device.
    """

    def __init__(self, rows, d_model, nh, hd, t, bias=False, rng=None):
        from gt1030.igpu_attn import Attention

        rng = rng or np.random.default_rng(2)
        self.rows, self.d_model, self.nh, self.hd = (
            int(rows),
            int(d_model),
            int(nh),
            int(hd),
        )
        self.t = int(t)
        assert self.rows % self.t == 0, "rows must be B*T"
        self.b = self.rows // self.t
        assert self.nh * self.hd == self.d_model, "nh*hd must equal d_model"
        self.qkv = Linear(rows, d_model, 3 * d_model, bias=bias, rng=rng)
        self.wo = Linear(rows, d_model, d_model, bias=bias, rng=rng)
        self.att = Attention(self.b, self.t, self.nh, self.hd)
        self.o_tokens = _buf(rows * d_model * 2)
        self.dqkv = _buf(rows * 3 * d_model * 2)
        self._last = None

    def _to_bh(self, arr):
        # (B,T,NH,HD) -> (B*NH,T,HD)
        return np.ascontiguousarray(
            arr.reshape(self.b, self.t, self.nh, self.hd)
            .transpose(0, 2, 1, 3)
            .reshape(self.b * self.nh, self.t, self.hd)
        )

    def _from_bh(self, arr):
        # (B*NH,T,HD) -> (B,T,NH,HD)
        return np.ascontiguousarray(
            arr.reshape(self.b, self.nh, self.t, self.hd).transpose(0, 2, 1, 3)
        )

    def forward(self, x_dev):
        qkv = self.qkv.forward(x_dev)
        _K["split_qkv"](
            A.queue,
            (self.b * self.nh * self.t * self.hd,),
            None,
            qkv,
            self.att.q,
            self.att.k,
            self.att.v,
            np.int32(self.b),
            np.int32(self.t),
            np.int32(self.nh),
            np.int32(self.hd),
        )
        o = self.att.forward()
        _K["heads_to_tok"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            o,
            self.o_tokens,
            np.int32(self.b),
            np.int32(self.t),
            np.int32(self.nh),
            np.int32(self.hd),
        )
        return self.wo.forward(self.o_tokens)

    def backward(self, dy_dev):
        do_tokens = self.wo.backward(dy_dev)
        _K["tok_to_heads"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            do_tokens,
            self.att.do,
            np.int32(self.b),
            np.int32(self.t),
            np.int32(self.nh),
            np.int32(self.hd),
        )
        dq, dk, dv = self.att.backward()
        _K["merge_qkv"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            dq,
            dk,
            dv,
            self.dqkv,
            np.int32(self.b),
            np.int32(self.t),
            np.int32(self.nh),
            np.int32(self.hd),
        )
        return self.qkv.backward(self.dqkv)

    def step(self, lr, t, eps=1e-8):
        self.qkv.step(lr, t, eps=eps)
        self.wo.step(lr, t, eps=eps)


class NormDev(object):
    """RMSNorm whose buffers all come from the layer arena (no direct cl.Buffer)."""

    def __init__(self, rows, cols, dev, eps=1e-6):
        self.rows, self.cols, self.eps = int(rows), int(cols), float(eps)
        self.dev = dev
        self.k = dev.norm_K
        self.w = _buf(self.cols * 2)
        self.rs = _buf(max(self.rows * 4, 4096))
        self.xhat = _buf(self.rows * self.cols * 2)
        self.dw = _buf(max(self.cols * 4, 4096))
        self.y = _buf(self.rows * self.cols * 2)
        self.dx = _buf(self.rows * self.cols * 2)
        self.m = _buf(self.cols * 4)
        self.v = _buf(self.cols * 4)
        self.master = _buf(self.cols * 4)

    def set_weight(self, w):
        cl.enqueue_copy(A.queue, self.w, np.ascontiguousarray(w, dtype=np.float16))
        cl.enqueue_copy(
            A.queue, self.master, np.ascontiguousarray(np.asarray(w, dtype=np.float32))
        )
        cl.enqueue_fill_buffer(A.queue, self.m, np.float32(0), 0, self.cols * 4)
        cl.enqueue_fill_buffer(A.queue, self.v, np.float32(0), 0, self.cols * 4)
        A.queue.finish()

    def forward(self, x_dev):
        self.k["rmsnorm_fwd"](
            A.queue,
            (self.rows * 256,),
            (256,),
            x_dev,
            self.w,
            self.y,
            self.rs,
            self.xhat,
            np.int32(self.rows),
            np.int32(self.cols),
            np.float32(self.eps),
        )
        return self.y

    def backward(self, dy_dev):
        self.k["rmsnorm_bwd"](
            A.queue,
            (self.rows * 256,),
            (256,),
            dy_dev,
            self.w,
            self.xhat,
            self.rs,
            self.dx,
            np.int32(self.rows),
            np.int32(self.cols),
        )
        return self.dx

    def weight_grad(self, dy_dev):
        cl.enqueue_fill_buffer(A.queue, self.dw, np.float32(0.0), 0, self.cols * 4)
        self.k["rmsnorm_dw"](
            A.queue,
            (self.cols,),
            None,
            dy_dev,
            self.xhat,
            self.dw,
            np.int32(self.rows),
            np.int32(self.cols),
        )
        return self.dw


class Block(object):
    """Pre-norm transformer block: x + attn(norm1(x)), then + mlp(norm2(...))."""

    def __init__(self, rows, d_model, hidden, attn=None, eps=1e-6, seed=None, dev=None):
        self.rows, self.d_model, self.hidden = int(rows), int(d_model), int(hidden)
        self.dev = dev
        if dev is not None:
            dev.activate()
        if dev is not None:
            self.norm1 = NormDev(rows, d_model, dev, eps=eps)
            self.norm2 = NormDev(rows, d_model, dev, eps=eps)
        else:
            self.norm1 = RMSNorm(rows, d_model, eps=eps)
            self.norm2 = RMSNorm(rows, d_model, eps=eps)
        self.norm1.set_weight(np.ones(d_model, dtype=np.float16))
        self.norm2.set_weight(np.ones(d_model, dtype=np.float16))
        self.norm1.m = _buf(d_model * 4)
        self.norm1.v = _buf(d_model * 4)
        self.norm2.m = _buf(d_model * 4)
        self.norm2.v = _buf(d_model * 4)
        self.norm1.master = _buf(d_model * 4)
        self.norm2.master = _buf(d_model * 4)
        for b in (self.norm1.m, self.norm1.v, self.norm2.m, self.norm2.v):
            cl.enqueue_fill_buffer(A.queue, b, np.float32(0), 0, d_model * 4)
        for nm in (self.norm1, self.norm2):
            cl.enqueue_copy(A.queue, nm.master, np.ones(d_model, dtype=np.float32))
        A.queue.finish()
        self.attn = attn
        self.mlp = MLP(
            rows,
            d_model,
            hidden,
            rng=np.random.default_rng(seed + 1) if seed is not None else None,
        )
        self.zero = _buf(rows * d_model * 2)
        _K["zero_half"](
            A.queue, (rows * d_model,), None, self.zero, np.int32(rows * d_model)
        )
        self.x1 = _buf(rows * d_model * 2)
        self.y = _buf(rows * d_model * 2)
        self.dx = _buf(rows * d_model * 2)
        self.dx1 = _buf(rows * d_model * 2)
        self._x = None

    def activate(self):
        if self.dev is not None:
            self.dev.activate()

    def forward(self, x_dev):
        self.activate()
        self._x = x_dev
        if self.attn is None:
            # Zero attention branch: x1 = x, and norm1 is unused (skip the pass).
            _K["add_half"](
                A.queue,
                (self.rows * self.d_model,),
                None,
                x_dev,
                self.zero,
                self.x1,
                np.int32(self.rows * self.d_model),
            )
        else:
            h1 = self.norm1.forward(x_dev)
            a = self.attn.forward(h1)
            _K["add_half"](
                A.queue,
                (self.rows * self.d_model,),
                None,
                x_dev,
                a,
                self.x1,
                np.int32(self.rows * self.d_model),
            )
        h2 = self.norm2.forward(self.x1)
        self.mlp.forward(h2)
        _K["add_half"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            self.x1,
            self.mlp._last_y,
            self.y,
            np.int32(self.rows * self.d_model),
        )
        return self.y

    def backward(self, dy_dev):
        self.activate()
        # y = x1 + MLP(norm2(x1)); g2 = dL/d(norm2 output), then dL/dx1 = dy + J_norm2^T g2
        g2 = self.mlp.backward(dy_dev)
        self.norm2.weight_grad(g2)
        d_norm2 = self.norm2.backward(g2)
        _K["add_half"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            dy_dev,
            d_norm2,
            self.dx1,
            np.int32(self.rows * self.d_model),
        )
        if self.attn is None:
            return self.dx1
        # x1 = x + attn(norm1(x)); dL/dattn_out = dL/dx1
        a_dx = self.attn.backward(self.dx1)
        self.norm1.weight_grad(a_dx)
        dh1 = self.norm1.backward(a_dx)
        _K["add_half"](
            A.queue,
            (self.rows * self.d_model,),
            None,
            self.dx1,
            dh1,
            self.dx,
            np.int32(self.rows * self.d_model),
        )
        return self.dx

    def _norm_step(self, norm, lr, t, eps=1e-8):
        cols = self.d_model
        bc1 = 1.0 - 0.9**t
        bc2 = 1.0 - 0.999**t
        _K["adamw_norm_master"](
            A.queue,
            ((cols + 63) // 64 * 64,),
            (64,),
            norm.w,
            norm.master,
            norm.dw,
            norm.m,
            norm.v,
            np.int32(cols),
            np.float32(lr),
            np.float32(0.9),
            np.float32(0.999),
            np.float32(eps),
            np.float32(0.0),
            np.float32(bc1),
            np.float32(bc2),
        )

    def step(self, lr, t, eps=1e-8):
        self.activate()
        self._norm_step(self.norm1, lr, t, eps=eps)
        self._norm_step(self.norm2, lr, t, eps=eps)
        self.mlp.step(lr, t, eps=eps)
        if self.attn is not None:
            self.attn.step(lr, t, eps=eps)


class Stack(object):
    """N pre-norm blocks + final RMSNorm; all device-resident."""

    def __init__(
        self,
        rows,
        d_model,
        hidden,
        nh,
        hd,
        t,
        layers,
        seed=0,
        eps=1e-6,
        final_norm=True,
        per_layer=False,
        profiling=False,
        layers_per_ctx=1,
    ):
        self.rows, self.d_model, self.hidden = int(rows), int(d_model), int(hidden)
        self.layers = int(layers)
        self.per_layer = bool(per_layer)
        self.layers_per_ctx = max(1, int(layers_per_ctx))
        self.blocks = []
        self.groups = []
        for start in range(0, self.layers, self.layers_per_ctx):
            dev = LayerCtx(profiling=profiling) if self.per_layer else None
            if dev is not None:
                dev.activate()
            group = []
            for i in range(start, min(start + self.layers_per_ctx, self.layers)):
                attn = SelfAttention(
                    rows, d_model, nh, hd, t, rng=np.random.default_rng(seed + 100 + i)
                )
                block = Block(
                    rows,
                    d_model,
                    hidden,
                    attn=attn,
                    eps=eps,
                    seed=seed + 200 + i,
                    dev=dev,
                )
                group.append(block)
                self.blocks.append(block)
            if dev is not None:
                self.groups.append((dev, group))
        if self.per_layer:
            for block in self.blocks:
                block.activate()
                block.bridge_x = _buf(self.rows * self.d_model * 2)
                block.bridge_dy = _buf(self.rows * self.d_model * 2)
        self.norm = None
        if final_norm:
            self.norm = RMSNorm(rows, d_model, eps=eps)
            self.norm.set_weight(np.ones(d_model, dtype=np.float16))
            self.norm.m = _buf(d_model * 4)
            self.norm.v = _buf(d_model * 4)
            for b in (self.norm.m, self.norm.v):
                cl.enqueue_fill_buffer(A.queue, b, np.float32(0), 0, d_model * 4)

    def forward(self, x_dev):
        x = x_dev
        for block in self.blocks:
            x = block.forward(x)
        return self.norm.forward(x) if self.norm is not None else x

    def backward(self, dy_dev):
        dy = dy_dev
        if self.norm is not None:
            self.norm.weight_grad(dy)
            dy = self.norm.backward(dy)
        for block in reversed(self.blocks):
            dy = block.backward(dy)
        return dy

    def forward_np(self, x_np):
        """Host bridge only at context-group boundaries; device pass inside a group."""
        h = np.ascontiguousarray(x_np, dtype=np.float16)
        for dev, group in self.groups:
            dev.activate()
            xb = group[0].bridge_x
            cl.enqueue_copy(A.queue, xb, np.ascontiguousarray(h))
            x = xb
            for block in group:
                block.activate()
                x = block.forward(x)
            h = _read(x, (self.rows, self.d_model))
        return h.astype(np.float32)

    def backward_np(self, dy_np):
        """Host bridge only at context-group boundaries; device pass inside a group."""
        d = np.ascontiguousarray(dy_np, dtype=np.float16)
        for dev, group in reversed(self.groups):
            dev.activate()
            dyb = group[-1].bridge_dy
            cl.enqueue_copy(A.queue, dyb, np.ascontiguousarray(d))
            x = dyb
            for block in reversed(group):
                block.activate()
                x = block.backward(x)
            d = _read(x, (self.rows, self.d_model))
        return d.astype(np.float32)

    def step(self, lr, t, eps=1e-8):
        if self.norm is not None:
            cols = self.d_model
            bc1 = 1.0 - 0.9**t
            bc2 = 1.0 - 0.999**t
            _K["adamw_norm_master"](
                A.queue,
                ((cols + 63) // 64 * 64,),
                (64,),
                self.norm.w,
                self.norm.master,
                self.norm.dw,
                self.norm.m,
                self.norm.v,
                np.int32(cols),
                np.float32(lr),
                np.float32(0.9),
                np.float32(0.999),
                np.float32(1e-8),
                np.float32(0.0),
                np.float32(bc1),
                np.float32(bc2),
            )
        for block in self.blocks:
            block.activate()
            block.step(lr, t, eps=eps)


def check_stack(b=2, t=8, nh=2, hd=16, hidden=64, layers=2, seed=0):
    import torch
    import torch.nn.functional as F

    rows, d = b * t, nh * hd
    rng = np.random.default_rng(seed)
    xh = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    x = _upload(xh)
    st = Stack(rows, d, hidden, nh, hd, t, layers, seed=seed)

    params = []
    for blk in st.blocks:
        params.append(
            dict(
                wq=torch.from_numpy(
                    _read(blk.attn.qkv.w, (3 * d, d)).astype(np.float32)
                ).requires_grad_(True),
                wo=torch.from_numpy(
                    _read(blk.attn.wo.w, (d, d)).astype(np.float32)
                ).requires_grad_(True),
                wg=torch.from_numpy(
                    _read(blk.mlp.gate.w, (hidden, d)).astype(np.float32)
                ).requires_grad_(True),
                wu=torch.from_numpy(
                    _read(blk.mlp.up.w, (hidden, d)).astype(np.float32)
                ).requires_grad_(True),
                wd=torch.from_numpy(
                    _read(blk.mlp.down.w, (d, hidden)).astype(np.float32)
                ).requires_grad_(True),
                bqkv=torch.zeros(3 * d, requires_grad=True),
                bwo=torch.zeros(d, requires_grad=True),
                bg=torch.zeros(hidden, requires_grad=True),
                bu=torch.zeros(hidden, requires_grad=True),
                bd=torch.zeros(d, requires_grad=True),
                n1=torch.ones(d, requires_grad=True),
                n2=torch.ones(d, requires_grad=True),
            )
        )
    nf = torch.ones(d, requires_grad=True)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def block_fwd(x, p):
        h1 = rms(x, p["n1"])
        qkv = h1 @ p["wq"].t() + p["bqkv"]
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        sh = lambda a: a.reshape(b, t, nh, hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        score = (q @ k.transpose(-1, -2)) / np.sqrt(hd)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        prob = torch.softmax(score.masked_fill(mask, float("-inf")), dim=-1)
        a = (prob @ v).transpose(1, 2).reshape(rows, d) @ p["wo"].t() + p["bwo"]
        x1 = x + a
        h2 = rms(x1, p["n2"])
        m = (F.silu(h2 @ p["wg"].t() + p["bg"]) * (h2 @ p["wu"].t() + p["bu"])) @ p[
            "wd"
        ].t() + p["bd"]
        return x1 + m

    xt = torch.from_numpy(xh.astype(np.float32)).requires_grad_(True)
    xr = xt
    for p in params:
        xr = block_fwd(xr, p)
    y_ref = rms(xr, nf)
    (y_ref**2).mean().backward()

    y = st.forward(x)
    yh = _read(y, (rows, d)).astype(np.float32)
    dy = _upload((2 * yh / (rows * d)).astype(np.float16))
    dx = st.backward(dy)
    dxh = _read(dx, (rows, d)).astype(np.float32)
    y_rel = float(
        np.linalg.norm(yh - y_ref.detach().numpy())
        / max(np.linalg.norm(y_ref.detach().numpy()), 1e-9)
    )
    dx_max = float(np.abs(dxh - xt.grad.numpy()).max())
    y_max = float(np.abs(yh - y_ref.detach().numpy()).max())
    print(
        "  stack layers=%d b=%d t=%d nh=%d hd=%d | y rel %.3e | y maxabs %.3e | dx max %.3e"
        % (layers, b, t, nh, hd, y_rel, y_max, dx_max)
    )
    ok = y_rel < 3e-2 and dx_max < 2e-3
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def check(rows=64, d=32, hidden=64, steps=3):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    xh = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    th = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    x = _upload(xh)
    t = _upload(th)
    blk = Block(rows, d, hidden, attn=None)

    class Ref(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = torch.nn.Parameter(torch.ones(d))
            self.gate = torch.nn.Linear(d, hidden)
            self.up = torch.nn.Linear(d, hidden)
            self.down = torch.nn.Linear(hidden, d)

        def forward(self, x):
            h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.n1
            h = torch.nn.functional.silu(self.gate(h)) * self.up(h)
            return x + self.down(h)

    ref = Ref()
    with torch.no_grad():
        ref.gate.weight.copy_(
            torch.from_numpy(_read(blk.mlp.gate.w, (hidden, d)).astype(np.float32))
        )
        ref.up.weight.copy_(
            torch.from_numpy(_read(blk.mlp.up.w, (hidden, d)).astype(np.float32))
        )
        ref.down.weight.copy_(
            torch.from_numpy(_read(blk.mlp.down.w, (d, hidden)).astype(np.float32))
        )
        ref.gate.bias.zero_()
        ref.up.bias.zero_()
        ref.down.bias.zero_()

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))

    xt = torch.from_numpy(xh.astype(np.float32)).requires_grad_(True)
    tt = torch.from_numpy(th.astype(np.float32))
    y_ref = ref(xt)
    loss = F.mse_loss(y_ref, tt)
    loss.backward()

    y = blk.forward(x)
    yh = _read(y, (rows, d))
    y_max = float(np.abs(yh.astype(np.float32) - y_ref.detach().numpy()).max())
    y_rel = rel(yh.astype(np.float32), y_ref.detach().numpy())

    dy = _upload(
        ((2.0 * (yh.astype(np.float32) - tt.numpy()) / (rows * d)).astype(np.float16))
    )
    dx = blk.backward(dy)
    dxh = _read(dx, (rows, d))
    dx_rel = rel(dxh.astype(np.float32), xt.grad.numpy())
    dx_max = float(np.abs(dxh.astype(np.float32) - xt.grad.numpy()).max())
    dwh = _read(blk.mlp.down.dw, (d, hidden)).astype(np.float32)
    dw_rel = rel(dwh, ref.down.weight.grad.numpy())
    dw_max = float(np.abs(dwh - ref.down.weight.grad.numpy()).max())

    opt = torch.optim.AdamW(
        ref.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    blk.step(1e-3, 1)
    opt.step()
    w_err = float(
        np.abs(
            _read(blk.mlp.down.w, (d, hidden)).astype(np.float32)
            - ref.down.weight.detach().numpy()
        ).max()
    )

    print(
        "  rows=%d d=%d hidden=%d | y rel %.3e (max %.3e) | dx rel %.3e | "
        "dW rel %.3e | step dW %.3e"
        % (rows, d, hidden, y_rel, y_max, dx_rel, dw_rel, w_err)
    )
    # CLBlast Hgemm accumulates in fp16 on gfx902: ~1% relative is expected for K~512-1368.
    # fp16 gradients are tiny at the AR shape; accept by absolute error, not rel-to-a-tiny-norm.
    ok = y_rel < 3e-2 and dx_max < 1e-3 and dw_max < 1e-4 and w_err < 5e-3
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def check_attention(b=2, t=8, nh=2, hd=16, hidden=64, seed=0):
    import torch
    import torch.nn.functional as F

    rows, d = b * t, nh * hd
    rng = np.random.default_rng(seed)
    xh = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    x = _upload(xh)
    sa = SelfAttention(rows, d, nh, hd, t, rng=np.random.default_rng(seed + 1))

    qkv_w = _read(sa.qkv.w, (3 * d, d)).astype(np.float32)
    wo_w = _read(sa.wo.w, (d, d)).astype(np.float32)
    xt = torch.from_numpy(xh.astype(np.float32)).requires_grad_(True)
    wq = torch.from_numpy(qkv_w.copy()).requires_grad_(True)
    wo = torch.from_numpy(wo_w.copy()).requires_grad_(True)
    qkv = xt @ wq.t()
    q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
    sh = lambda a: a.reshape(b, t, nh, hd).transpose(1, 2)
    q, k, v = sh(q), sh(k), sh(v)
    score = (q @ k.transpose(-1, -2)) / np.sqrt(hd)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    score = score.masked_fill(mask, float("-inf"))
    prob = torch.softmax(score, dim=-1)
    o = (prob @ v).transpose(1, 2).reshape(rows, d)
    y_ref = o @ wo.t()
    loss = (y_ref**2).mean()
    loss.backward()

    y = sa.forward(x)
    yh = _read(y, (rows, d)).astype(np.float32)
    dy = _upload((2 * yh / (rows * d)).astype(np.float16))
    dx = sa.backward(dy)
    dxh = _read(dx, (rows, d)).astype(np.float32)
    dwh = _read(sa.qkv.dw, (3 * d, d)).astype(np.float32)
    y_rel = float(
        np.linalg.norm(yh - y_ref.detach().numpy())
        / max(np.linalg.norm(y_ref.detach().numpy()), 1e-9)
    )
    dx_max = float(np.abs(dxh - xt.grad.numpy()).max())
    dw_max = float(np.abs(dwh - wq.grad.numpy()).max())
    print(
        "  b=%d t=%d nh=%d hd=%d | y rel %.3e | dx max %.3e | dWqkv max %.3e"
        % (b, t, nh, hd, y_rel, dx_max, dw_max)
    )
    ok = y_rel < 3e-2 and dx_max < 2e-3 and dw_max < 1e-4
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def check_block_attn(b=2, t=8, nh=2, hd=16, hidden=64, seed=0):
    import torch

    rows, d = b * t, nh * hd
    rng = np.random.default_rng(seed)
    xh = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    x = _upload(xh)
    blk = Block(rows, d, hidden, attn=SelfAttention(rows, d, nh, hd, t))

    qkv_w = _read(blk.attn.qkv.w, (3 * d, d)).astype(np.float32)
    wo_w = _read(blk.attn.wo.w, (d, d)).astype(np.float32)
    gate_w = _read(blk.mlp.gate.w, (hidden, d)).astype(np.float32)
    up_w = _read(blk.mlp.up.w, (hidden, d)).astype(np.float32)
    down_w = _read(blk.mlp.down.w, (d, hidden)).astype(np.float32)

    xt = torch.from_numpy(xh.astype(np.float32)).requires_grad_(True)
    wq = torch.from_numpy(qkv_w.copy()).requires_grad_(True)
    wo = torch.from_numpy(wo_w.copy()).requires_grad_(True)
    wg = torch.from_numpy(gate_w.copy()).requires_grad_(True)
    wu = torch.from_numpy(up_w.copy()).requires_grad_(True)
    wd = torch.from_numpy(down_w.copy()).requires_grad_(True)
    bqkv = torch.zeros(3 * d, requires_grad=True)
    bwo = torch.zeros(d, requires_grad=True)
    bg = torch.zeros(hidden, requires_grad=True)
    bu = torch.zeros(hidden, requires_grad=True)
    bd = torch.zeros(d, requires_grad=True)
    n1 = torch.ones(d, requires_grad=True)
    n2 = torch.ones(d, requires_grad=True)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def att(x):
        qkv = x @ wq.t()
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        sh = lambda a: a.reshape(b, t, nh, hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        score = (q @ k.transpose(-1, -2)) / np.sqrt(hd)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        prob = torch.softmax(score.masked_fill(mask, float("-inf")), dim=-1)
        return (prob @ v).transpose(1, 2).reshape(rows, d) @ wo.t() + bwo

    h1 = rms(xt, n1)
    x1 = xt + att(h1)
    h2 = rms(x1, n2)
    y_ref = x1 + (torch.nn.functional.silu(h2 @ wg.t()) * (h2 @ wu.t())) @ wd.t()
    (y_ref**2).mean().backward()

    y = blk.forward(x)
    yh = _read(y, (rows, d)).astype(np.float32)
    dy = _upload((2 * yh / (rows * d)).astype(np.float16))
    dx = blk.backward(dy)
    dxh = _read(dx, (rows, d)).astype(np.float32)
    dwq = _read(blk.attn.qkv.dw, (3 * d, d)).astype(np.float32)
    y_rel = float(
        np.linalg.norm(yh - y_ref.detach().numpy())
        / max(np.linalg.norm(y_ref.detach().numpy()), 1e-9)
    )
    dx_max = float(np.abs(dxh - xt.grad.numpy()).max())
    dw_max = float(np.abs(dwq - wq.grad.numpy()).max())
    print(
        "  block b=%d t=%d nh=%d hd=%d hidden=%d | y rel %.3e | dx max %.3e | dWqkv max %.3e"
        % (b, t, nh, hd, hidden, y_rel, dx_max, dw_max)
    )
    ok = y_rel < 3e-2 and dx_max < 2e-3 and dw_max < 1e-4
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def check_block_train(b=2, t=8, nh=2, hd=16, hidden=64, steps=20, lr=1e-5, seed=0):
    import torch
    import torch.nn.functional as F

    rows, d = b * t, nh * hd
    rng = np.random.default_rng(seed)
    xh = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    th = (rng.standard_normal((rows, d)) * 0.05).astype(np.float16)
    x = _upload(xh)
    blk = Block(rows, d, hidden, attn=SelfAttention(rows, d, nh, hd, t))

    qkv_w = _read(blk.attn.qkv.w, (3 * d, d)).astype(np.float32)
    wo_w = _read(blk.attn.wo.w, (d, d)).astype(np.float32)
    gate_w = _read(blk.mlp.gate.w, (hidden, d)).astype(np.float32)
    up_w = _read(blk.mlp.up.w, (hidden, d)).astype(np.float32)
    down_w = _read(blk.mlp.down.w, (d, hidden)).astype(np.float32)

    xt = torch.from_numpy(xh.astype(np.float32))
    tt = torch.from_numpy(th.astype(np.float32))
    wq = torch.from_numpy(qkv_w.copy()).requires_grad_(True)
    wo = torch.from_numpy(wo_w.copy()).requires_grad_(True)
    wg = torch.from_numpy(gate_w.copy()).requires_grad_(True)
    wu = torch.from_numpy(up_w.copy()).requires_grad_(True)
    wd = torch.from_numpy(down_w.copy()).requires_grad_(True)
    bqkv = torch.zeros(3 * d, requires_grad=True)
    bwo = torch.zeros(d, requires_grad=True)
    bg = torch.zeros(hidden, requires_grad=True)
    bu = torch.zeros(hidden, requires_grad=True)
    bd = torch.zeros(d, requires_grad=True)
    n1 = torch.ones(d, requires_grad=True)
    n2 = torch.ones(d, requires_grad=True)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def att(x):
        qkv = x @ wq.t() + bqkv
        q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        sh = lambda a: a.reshape(b, t, nh, hd).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        score = (q @ k.transpose(-1, -2)) / np.sqrt(hd)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        prob = torch.softmax(score.masked_fill(mask, float("-inf")), dim=-1)
        return (prob @ v).transpose(1, 2).reshape(rows, d) @ wo.t() + bwo

    def ref_forward(x):
        x1 = x + att(rms(x, n1))
        h2 = rms(x1, n2)
        return x1 + (F.silu(h2 @ wg.t() + bg) * (h2 @ wu.t() + bu)) @ wd.t() + bd

    opt = torch.optim.AdamW(
        [wq, wo, wg, wu, wd, bqkv, bwo, bg, bu, bd, n1, n2],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    losses, refs = [], []
    for step in range(1, steps + 1):
        yh = _read(blk.forward(x), (rows, d)).astype(np.float32)
        loss = float(((yh - th.astype(np.float32)) ** 2).mean())
        losses.append(loss)
        dy = _upload(
            (2.0 * (yh - th.astype(np.float32)) / (rows * d)).astype(np.float16)
        )
        blk.backward(dy)
        blk.step(lr, step)
        yref = ref_forward(xt)
        loss_ref = ((yref - tt) ** 2).mean()
        opt.zero_grad()
        loss_ref.backward()
        opt.step()
        refs.append(float(loss_ref.detach()))
    worst = 0.0
    for i, (a, r) in enumerate(zip(losses, refs)):
        rel = abs(a - r) / max(abs(r), 1e-12)
        worst = max(worst, rel)
        if i in (0, steps - 1):
            print(
                "  step %2d: device %.6g | torch %.6g | rel %.3e" % (i + 1, a, r, rel)
            )
    print("  worst loss rel over %d steps: %.3e" % (steps, worst))
    ok = worst < 1e-2
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--check-attn", action="store_true")
    ap.add_argument("--check-block-attn", action="store_true")
    ap.add_argument("--check-train", action="store_true")
    ap.add_argument("--check-stack", action="store_true")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--nh", type=int, default=2)
    ap.add_argument("--hd", type=int, default=16)
    a = ap.parse_args()
    if a.check_stack:
        return check_stack(a.b, a.t, a.nh, a.hd, a.hidden, a.layers)
    if a.check_train:
        return check_block_train(a.b, a.t, a.nh, a.hd, a.hidden, a.steps, a.lr)
    if a.check_block_attn:
        return check_block_attn(a.b, a.t, a.nh, a.hd, a.hidden)
    if a.check_attn:
        return check_attention(a.b, a.t, a.nh, a.hd, a.hidden)
    if a.check:
        return check(a.rows, a.d, a.hidden, a.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
