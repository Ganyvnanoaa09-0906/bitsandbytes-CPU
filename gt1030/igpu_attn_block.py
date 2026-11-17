# -*- coding: utf-8 -*-
"""igpu_attn_block.py — dev13 P3：**多头因果自注意力的设备实现**（(B*T, D) 布局，head 内 gather/scatter）。

与 `igpu_attn.Attention` 的区别：那个是"已经按 head 排好布局 (BH,T,HD)"的批处理版，
这里吃的是 transformer 里最常见的 `(B*T, nh*hd)` 连续布局，并按 head 切片计算，
便于和 `igpu_fx` 的 Linear 输出直接对接（不需要 host 侧 permute）。

全部 fp16；softmax 用 fp32 累加；缩放 = 1/sqrt(hd)。

    python gt1030/igpu_attn_block.py --check     # fwd/bwd 对拍 torch
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 从 (BT, D) 里取第 h 个 head 的 (BT, hd)  [gather]
__kernel void head_gather(__global const half* x, __global half* y, int bt, int hd, int D, int h) {
    int r = get_global_id(0), c = get_global_id(1);
    if (r >= bt || c >= hd) return;
    y[(long)r * hd + c] = x[(long)r * D + h * hd + c];
}

// 把 (BT, hd) 累加回 (BT, D) 的第 h 个 head  [scatter-add]
__kernel void head_scatter(__global const half* dy, __global half* dx, int bt, int hd, int D,
                           int h, int nh) {
    int r = get_global_id(0), c = get_global_id(1);
    if (r >= bt || c >= hd) return;
    long off = (long)r * (hd * nh) + h * hd + c;
    dx[off] = (half)((float)dx[off] + (float)dy[(long)r * hd + c]);
}

// 因果 softmax（逐行，T×T），out 用 fp16；同时不保存中间量（bwd 重算）
__kernel void causal_softmax(__global const half* s, __global half* p, int t, float scale) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* line = s + (long)r * t;
    __global half* out = p + (long)r * t;
    float mx = -1e30f;
    for (int c = lid; c < t; c += lsz) if (c <= r) mx = fmax(mx, (float)line[c] * scale);
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] = fmax(red[lid], red[lid + off]); barrier(CLK_LOCAL_MEM_FENCE); }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.f;
    for (int c = lid; c < t; c += lsz) if (c <= r) acc += exp((float)line[c] * scale - m);
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float z = red[0];
    for (int c = lid; c < t; c += lsz)
        out[c] = (half)((c <= r) ? exp((float)line[c] * scale - m) / z : 0.f);
}

// dS = P * (dP - rowsum(dP*P))，已含 scale；同时把 (c>r) 置零
__kernel void ds_from_dp(__global const half* p, __global const half* dp, __global half* ds,
                         int t, float scale) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* pl = p + (long)r * t;
    __global const half* dl = dp + (long)r * t;
    __global half* o = ds + (long)r * t;
    float acc = 0.f;
    for (int c = lid; c < t; c += lsz) if (c <= r) acc += (float)dl[c] * (float)pl[c];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float s = red[0];
    for (int c = lid; c < t; c += lsz) {
        float v = (c <= r) ? (float)pl[c] * ((float)dl[c] - s) * scale : 0.f;
        o[c] = (half)v;
    }
}

// rowsum(dP*P) 单独用（dQ 用不到，dS 需要）——已内联在 ds_from_dp
__kernel void row_sum_h(__global const half* x, __global float* out, int t) {
    int r = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    __local float red[256];
    __global const half* line = x + (long)r * t;
    float acc = 0.f;
    for (int c = lid; c < t; c += lsz) acc += (float)line[c];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) out[r] = red[0];
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in
                   ("head_gather", "head_scatter", "causal_softmax", "ds_from_dp", "row_sum_h")}
    return _C[key]


class AttnBlock(object):
    """一个多头因果自注意力（不含投影，投影由调用方的 Linear 负责）。

    输入/输出都是 (BT, D) fp16 的 device buffer，D = nh*hd。
    """

    def __init__(self, dev, bt, nh, hd):
        self.dev, self.bt, self.nh, self.hd = dev, int(bt), int(nh), int(hd)
        self.D = self.nh * self.hd
        self.T = self.bt                # 单序列（B=1）时的长度
        self.scale = 1.0 / np.sqrt(self.hd)
        d = dev
        self.q = self._buf((self.bt, self.D))
        self.k = self._buf((self.bt, self.D))
        self.v = self._buf((self.bt, self.D))
        self.o = self._buf((self.bt, self.D))
        self.do = self._buf((self.bt, self.D))
        self.dq = self._buf((self.bt, self.D))
        self.dk = self._buf((self.bt, self.D))
        self.dv = self._buf((self.bt, self.D))
        # 每 head 的工作区（复用，不按 head 分配）
        self.qh = self._buf((self.bt, self.hd))
        self.kh = self._buf((self.bt, self.hd))
        self.vh = self._buf((self.bt, self.hd))
        self.oh = self._buf((self.bt, self.hd))
        self.doh = self._buf((self.bt, self.hd))
        self.dqh = self._buf((self.bt, self.hd))
        self.dkh = self._buf((self.bt, self.hd))
        self.dvh = self._buf((self.bt, self.hd))
        self.s = self._buf((self.bt, self.bt))
        self.p = self._buf((self.bt, self.bt))
        self.dp = self._buf((self.bt, self.bt))
        self.ds = self._buf((self.bt, self.bt))
        # 保存每个 head 的 P（bwd 要用）——heads 少（nh≤8）时内存可接受
        self.p_keep = [self._buf((self.bt, self.bt)) for _ in range(self.nh)]

    def _buf(self, shape):
        n = 1
        for x in shape:
            n *= x
        return self.dev.buf(max(4096, n * 2))

    def _gather(self, x, xh, h):
        K(self.dev)["head_gather"](self.dev.q, (self.bt, self.hd), None, x, xh,
                                   np.int32(self.bt), np.int32(self.hd), np.int32(self.D),
                                   np.int32(h))

    def _scatter(self, dyh, dx, h):
        K(self.dev)["head_scatter"](self.dev.q, (self.bt, self.hd), None, dyh, dx,
                                    np.int32(self.bt), np.int32(self.hd), np.int32(self.D),
                                    np.int32(h), np.int32(self.nh))

    def forward(self, q_buf, k_buf, v_buf):
        """q/k/v: (BT, D) fp16 device buffers（调用方已做 qkv 投影）。"""
        d, kk = self.dev, K(self.dev)
        for bufsrc, bufdst in ((q_buf, self.q), (k_buf, self.k), (v_buf, self.v)):
            cl.enqueue_copy(d.q, bufdst, bufsrc)
        for h in range(self.nh):
            self._gather(self.q, self.qh, h)
            self._gather(self.k, self.kh, h)
            self._gather(self.v, self.vh, h)
            # S = Q Kᵀ : (BT,BT) = (BT,hd) × (BT,hd)ᵀ
            GEMM.hgemm_nt(self.qh, self.kh, self.s, self.bt, self.bt, self.hd)
            kk["causal_softmax"](d.q, (self.bt * 256,), (256,), self.s, self.p,
                                 np.int32(self.bt), np.float32(self.scale))
            cl.enqueue_copy(d.q, self.p_keep[h], self.p)
            # O = P V : (BT,hd) = (BT,BT) × (BT,hd)
            GEMM.hgemm_nn(self.p, self.vh, self.oh, self.bt, self.hd, self.bt)
            self._scatter(self.oh, self.o, h)
        d.q.finish()
        return self.o

    def backward(self, do_buf):
        """do: (BT, D) 输出梯度；返回 (dq, dk, dv) 三个 device buffer。"""
        d, kk = self.dev, K(self.dev)
        cl.enqueue_copy(d.q, self.do, do_buf)
        for b in (self.dq, self.dk, self.dv):
            cl.enqueue_copy(d.q, b, np.zeros(self.bt * self.D, np.float16))
        for h in range(self.nh):
            self._gather(self.do, self.doh, h)
            self._gather(self.q, self.qh, h)
            self._gather(self.k, self.kh, h)
            self._gather(self.v, self.vh, h)
            cl.enqueue_copy(d.q, self.p, self.p_keep[h])
            # dP = dO Vᵀ                       (bt,bt)
            GEMM.hgemm_nt(self.doh, self.vh, self.dp, self.bt, self.bt, self.hd)
            # dS = P*(dP - rowsum(dP*P)) * scale
            kk["ds_from_dp"](d.q, (self.bt * 256,), (256,), self.p, self.dp, self.ds,
                             np.int32(self.bt), np.float32(self.scale))
            # dQ = dS K                         (bt,hd)
            GEMM.hgemm_nn(self.ds, self.kh, self.dqh, self.bt, self.hd, self.bt)
            self._scatter(self.dqh, self.dq, h)
            # dK = dSᵀ Q                        (bt,hd)
            GEMM.hgemm_tn(self.ds, self.qh, self.dkh, self.bt, self.hd, self.bt)
            self._scatter(self.dkh, self.dk, h)
            # dV = Pᵀ dO                        (bt,hd)
            GEMM.hgemm_tn(self.p, self.doh, self.dvh, self.bt, self.hd, self.bt)
            self._scatter(self.dvh, self.dv, h)
        d.q.finish()
        return self.dq, self.dk, self.dv


def check(bt=16, nh=4, hd=8, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    D = nh * hd
    q = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    v = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    do = (rng.standard_normal((bt, D)) * 0.1).astype(np.float16)

    d = CTX.DevCtx(arena_mb=400, name="attn-block")
    d.activate()
    blk = AttnBlock(d, bt, nh, hd)
    out = blk.forward(d.up(q), d.up(k), d.up(v))
    y = d.down(out, (bt, D))

    tq = torch.from_numpy(q.astype(np.float32)).requires_grad_(True)
    tk = torch.from_numpy(k.astype(np.float32)).requires_grad_(True)
    tv = torch.from_numpy(v.astype(np.float32)).requires_grad_(True)

    def heads(x):
        return x.view(bt, nh, hd).transpose(0, 1)

    s = (heads(tq) @ heads(tk).transpose(-1, -2)) / np.sqrt(hd)
    mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = (p @ heads(tv)).transpose(0, 1).reshape(bt, D)
    rel = float(np.linalg.norm(y.astype(np.float32) - o.detach().numpy())
                / (np.linalg.norm(o.detach().numpy()) + 1e-12))

    # 反传对拍
    o.backward(torch.from_numpy(do.astype(np.float32)))
    dq, dk, dv = blk.backward(d.up(do))
    gq = d.down(dq, (bt, D)).astype(np.float32)
    gk = d.down(dk, (bt, D)).astype(np.float32)
    gv = d.down(dv, (bt, D)).astype(np.float32)

    def r(a, t):
        return float(np.linalg.norm(a - t.numpy()) / (np.linalg.norm(t.numpy()) + 1e-12))

    eq, ek, ev = r(gq, tq.grad), r(gk, tk.grad), r(gv, tv.grad)
    print("  attn bt=%d nh=%d hd=%d | fwd %.2e dq %.2e dk %.2e dv %.2e"
          % (bt, nh, hd, rel, eq, ek, ev))
    return max(rel, eq, ek, ev) < 5e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
