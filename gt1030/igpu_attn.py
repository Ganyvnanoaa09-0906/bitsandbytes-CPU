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

# ★ 向量化（half8）softmax：来自 ar_v2_multi，实测比标量版快 2-4x
VSRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void softmax_causal_v(__global const half* s, __global half* p, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float vals[1024];
    __local float red[256];
    __global const half* line = s + ((long)bh * T + row) * T;
    __global half* out = p + ((long)bh * T + row) * T;
    float mx = -1e30f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float a0 = (float)v.s0 * scale, a1 = (float)v.s1 * scale;
            float a2 = (float)v.s2 * scale, a3 = (float)v.s3 * scale;
            float a4 = (float)v.s4 * scale, a5 = (float)v.s5 * scale;
            float a6 = (float)v.s6 * scale, a7 = (float)v.s7 * scale;
            mx = fmax(mx, fmax(fmax(fmax(a0, a1), fmax(a2, a3)),
                              fmax(fmax(a4, a5), fmax(a6, a7))));
        } else {
            for (int t = k; t < n; ++t) mx = fmax(mx, (float)line[t] * scale);
        }
    }
    red[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 v = vload8(0, line + k);
            float e0 = exp((float)v.s0 * scale - m), e1 = exp((float)v.s1 * scale - m);
            float e2 = exp((float)v.s2 * scale - m), e3 = exp((float)v.s3 * scale - m);
            float e4 = exp((float)v.s4 * scale - m), e5 = exp((float)v.s5 * scale - m);
            float e6 = exp((float)v.s6 * scale - m), e7 = exp((float)v.s7 * scale - m);
            vals[k + 0] = e0; vals[k + 1] = e1; vals[k + 2] = e2; vals[k + 3] = e3;
            vals[k + 4] = e4; vals[k + 5] = e5; vals[k + 6] = e6; vals[k + 7] = e7;
            sum += (e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7);
        } else {
            for (int t = k; t < n; ++t) {
                float e = exp((float)line[t] * scale - m);
                vals[t] = e; sum += e;
            }
        }
    }
    red[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        o.s0 = (k + 0 < n) ? (half)(vals[k + 0] * inv) : (half)0.f;
        o.s1 = (k + 1 < n) ? (half)(vals[k + 1] * inv) : (half)0.f;
        o.s2 = (k + 2 < n) ? (half)(vals[k + 2] * inv) : (half)0.f;
        o.s3 = (k + 3 < n) ? (half)(vals[k + 3] * inv) : (half)0.f;
        o.s4 = (k + 4 < n) ? (half)(vals[k + 4] * inv) : (half)0.f;
        o.s5 = (k + 5 < n) ? (half)(vals[k + 5] * inv) : (half)0.f;
        o.s6 = (k + 6 < n) ? (half)(vals[k + 6] * inv) : (half)0.f;
        o.s7 = (k + 7 < n) ? (half)(vals[k + 7] * inv) : (half)0.f;
        vstore8(o, 0, out + k);
    }
}

__kernel void softmax_bwd_causal_v(__global const half* p, __global const half* dp,
                                   __global half* ds, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float red[256];
    __global const half* pl = p + ((long)bh * T + row) * T;
    __global const half* dl = dp + ((long)bh * T + row) * T;
    __global half* ol = ds + ((long)bh * T + row) * T;
    float sacc = 0.f;
    for (int k = lid * 8; k < n; k += lsz * 8) {
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            sacc += (float)a.s0 * (float)b.s0 + (float)a.s1 * (float)b.s1
                  + (float)a.s2 * (float)b.s2 + (float)a.s3 * (float)b.s3
                  + (float)a.s4 * (float)b.s4 + (float)a.s5 * (float)b.s5
                  + (float)a.s6 * (float)b.s6 + (float)a.s7 * (float)b.s7;
        } else {
            for (int t = k; t < n; ++t) sacc += (float)pl[t] * (float)dl[t];
        }
    }
    red[lid] = sacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tot = red[0];
    for (int k = lid * 8; k < T; k += lsz * 8) {
        half8 o;
        if (k + 8 <= n) {
            half8 a = vload8(0, pl + k);
            half8 b = vload8(0, dl + k);
            o.s0 = (half)((float)a.s0 * ((float)b.s0 - tot) * scale);
            o.s1 = (half)((float)a.s1 * ((float)b.s1 - tot) * scale);
            o.s2 = (half)((float)a.s2 * ((float)b.s2 - tot) * scale);
            o.s3 = (half)((float)a.s3 * ((float)b.s3 - tot) * scale);
            o.s4 = (half)((float)a.s4 * ((float)b.s4 - tot) * scale);
            o.s5 = (half)((float)a.s5 * ((float)b.s5 - tot) * scale);
            o.s6 = (half)((float)a.s6 * ((float)b.s6 - tot) * scale);
            o.s7 = (half)((float)a.s7 * ((float)b.s7 - tot) * scale);
        } else {
            o.s0 = (k + 0 < n) ? (half)((float)pl[k + 0] * ((float)dl[k + 0] - tot) * scale) : (half)0.f;
            o.s1 = (k + 1 < n) ? (half)((float)pl[k + 1] * ((float)dl[k + 1] - tot) * scale) : (half)0.f;
            o.s2 = (k + 2 < n) ? (half)((float)pl[k + 2] * ((float)dl[k + 2] - tot) * scale) : (half)0.f;
            o.s3 = (k + 3 < n) ? (half)((float)pl[k + 3] * ((float)dl[k + 3] - tot) * scale) : (half)0.f;
            o.s4 = (k + 4 < n) ? (half)((float)pl[k + 4] * ((float)dl[k + 4] - tot) * scale) : (half)0.f;
            o.s5 = (k + 5 < n) ? (half)((float)pl[k + 5] * ((float)dl[k + 5] - tot) * scale) : (half)0.f;
            o.s6 = (k + 6 < n) ? (half)((float)pl[k + 6] * ((float)dl[k + 6] - tot) * scale) : (half)0.f;
            o.s7 = (k + 7 < n) ? (half)((float)pl[k + 7] * ((float)dl[k + 7] - tot) * scale) : (half)0.f;
        }
        vstore8(o, 0, ol + k);
    }
}

"""


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


_PROG = {}
_K = {}


def _ctx_key():
    c = A.ctx
    return int(c.int_ptr) if hasattr(c, 'int_ptr') else id(c)


def _kernels():
    """★ 内核缓存必须 **按 context 分开**：多层 stack 每层一个 context 时，
    用单一全局缓存会让后一层拿前一层的内核（跨 context 入队）→ 静默算错/部分不执行。"""
    key = _ctx_key()
    if key not in _PROG:
        _PROG[key] = cl.Program(A.ctx, AG.KERNELS).build(
            options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
        _v = cl.Program(A.ctx, VSRC).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
        _K[key] = {n: cl.Kernel(_PROG[key], n)
                   for n in ('softmax_causal', 'softmax_bwd_causal')}
        _K[key]['softmax_causal_v'] = cl.Kernel(_v, 'softmax_causal_v')
        _K[key]['softmax_bwd_causal_v'] = cl.Kernel(_v, 'softmax_bwd_causal_v')
    return _K[key]


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
        _sm = "softmax_causal_v" if T >= 512 else "softmax_causal"   # T 小时向量版线程利用率不足
        _kernels()[_sm](A.queue, (T * 256, BH), (256, 1), self.s, self.p,
                        np.int32(T), np.float32(self.scale))  # P = softmax(S·scale)
        hgemm_batched_ex(self.p, self.v, self.o, T, HD, T, A.NO, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)          # O = P V
        return self.o

    # ---------------- backward ----------------
    def backward(self, do_dev=None):
        BH, T, HD = self.BH, self.T, self.HD
        if do_dev is not None:
            # ★ 修 bug：原来用 hasattr(self.do, "shape") 判断，cl.Buffer 没有 shape ⇒ 永远不拷贝
            src = do_dev if isinstance(do_dev, cl.Buffer) else np.ascontiguousarray(do_dev, dtype=np.float16)
            cl.enqueue_copy(A.queue, self.do, src)
            A.queue.finish()
        # dV = Pᵀ dO
        hgemm_batched_ex(self.p, self.do, self.dv, T, HD, T, A.YES, A.NO,
                         T, HD, HD, T * T, T * HD, T * HD, BH)
        # dP = dO Vᵀ
        hgemm_batched_ex(self.do, self.v, self.dp, T, T, HD, A.NO, A.YES,
                         HD, HD, T, T * HD, T * HD, T * T, BH)
        # dS = P ⊙ (dP − rowsum(dP⊙P)) · scale
        _smy = "softmax_bwd_causal_v" if T >= 512 else "softmax_bwd_causal"
        _kernels()[_smy](A.queue, (T * 256, BH), (256, 1), self.p, self.dp,
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
    att.backward(do.reshape(BH, t, hd))      # ★ 走 do_dev 参数路径（此前是静默忽略的 bug 路径）
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
