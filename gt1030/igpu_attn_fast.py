# -*- coding: utf-8 -*-
"""igpu_attn_fast.py — **head-batched** 因果自注意力：(BT, nh*hd) 布局，一次算所有 head。

与 `igpu_attn_block.AttnBlock`（按 head 循环，每层 4×6 次 launch）相比：
  1. q/k/v 各自**一次**重排成 (nh*T, hd) 的批布局（3 次 launch，而不是 4*nh 次）；
  2. S = QKᵀ、O = PV 用 **CLBlast strided-batched hgemm**（每个方向 1 次）；
  3. 因果 softmax / dS 各 1 次（按行覆盖 nh*T 行）；
  4. 反传同样只有 ~6 次 launch。

    python gt1030/igpu_attn_fast.py --check     # 与 torch 对拍 fwd/bwd
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import A  # noqa: E402
from gt1030.igpu_attn import hgemm_batched_ex  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// (BT, D) 交错布局 → (nh, T, hd) 批布局：out[h*T + t, c] = x[t, h*hd + c]
__kernel void pack_heads(__global const half* x, __global half* y,
                         int T, int nh, int hd) {
    int h = get_global_id(0), t = get_global_id(1), c = get_global_id(2);
    if (h >= nh || t >= T || c >= hd) return;
    int D = nh * hd;
    y[((long)h * T + t) * hd + c] = x[(long)t * D + h * hd + c];
}

// 批布局 → 交错布局（每个目标元素唯一，直接覆盖，无需预清零）
__kernel void unpack_heads(__global const half* y, __global half* x,
                           int T, int nh, int hd) {
    int h = get_global_id(0), t = get_global_id(1), c = get_global_id(2);
    if (h >= nh || t >= T || c >= hd) return;
    int D = nh * hd;
    long o = (long)t * D + h * hd + c;
    x[o] = y[((long)h * T + t) * hd + c];
}

// 三路一次 pack：q/k/v 交错 (BT,D) → 三个 (nh*T,hd) 批布局
__kernel void pack_qkv(__global const half* q, __global const half* k,
                       __global const half* v, __global half* qp,
                       __global half* kp, __global half* vp,
                       int T, int nh, int hd) {
    int h = get_global_id(0), t = get_global_id(1), c = get_global_id(2);
    if (h >= nh || t >= T || c >= hd) return;
    int D = nh * hd;
    long src = (long)t * D + h * hd + c;
    long dst = ((long)h * T + t) * hd + c;
    qp[dst] = q[src];
    kp[dst] = k[src];
    vp[dst] = v[src];
}

// 三路一次 unpack（覆盖写，无需预清零）
__kernel void unpack_qkv(__global const half* dqp, __global const half* dkp,
                         __global const half* dvp, __global half* dq,
                         __global half* dk, __global half* dv,
                         int T, int nh, int hd) {
    int h = get_global_id(0), t = get_global_id(1), c = get_global_id(2);
    if (h >= nh || t >= T || c >= hd) return;
    int D = nh * hd;
    long dst = (long)t * D + h * hd + c;
    long src = ((long)h * T + t) * hd + c;
    dq[dst] = dqp[src];
    dk[dst] = dkp[src];
    dv[dst] = dvp[src];
}

// 因果 softmax：共 nh*T 行，每行长度 T（row 号 % T 即在序列中的位置）
__kernel void causal_softmax_batched(__global const half* s, __global half* p,
                                     int T, int nh, float scale) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int qpos = row % T;
    __local float red[256];
    __global const half* line = s + (long)row * T;
    __global half* out = p + (long)row * T;
    float mx = -1e30f;
    for (int c = lid; c < T; c += lsz) if (c <= qpos) mx = fmax(mx, (float)line[c] * scale);
    red[lid] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] = fmax(red[lid], red[lid + off]); barrier(CLK_LOCAL_MEM_FENCE); }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.f;
    for (int c = lid; c < T; c += lsz) if (c <= qpos) acc += exp((float)line[c] * scale - m);
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float z = red[0];
    for (int c = lid; c < T; c += lsz)
        out[c] = (half)((c <= qpos) ? exp((float)line[c] * scale - m) / z : 0.f);
}

// dS = P * (dP - rowsum(dP*P)) * scale，掩码清零
__kernel void ds_batched(__global const half* p, __global const half* dp, __global half* ds,
                         int T, float scale) {
    int row = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int qpos = row % T;
    __local float red[256];
    __global const half* pl = p + (long)row * T;
    __global const half* dl = dp + (long)row * T;
    __global half* o = ds + (long)row * T;
    float acc = 0.f;
    for (int c = lid; c < T; c += lsz) if (c <= qpos) acc += (float)dl[c] * (float)pl[c];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) { if (lid < off) red[lid] += red[lid + off]; barrier(CLK_LOCAL_MEM_FENCE); }
    float sum = red[0];
    for (int c = lid; c < T; c += lsz) {
        float v = (c <= qpos) ? (float)pl[c] * ((float)dl[c] - sum) * scale : 0.f;
        o[c] = (half)v;
    }
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in
                   ("pack_heads", "unpack_heads", "pack_qkv", "unpack_qkv",
                    "causal_softmax_batched", "ds_batched")}
    return _C[key]


class FastAttn(object):
    """多头因果注意力（head-batched）。q/k/v/o 的投影由调用方负责。"""

    def __init__(self, dev, bt, nh, hd):
        self.dev, self.bt, self.nh, self.hd = dev, int(bt), int(nh), int(hd)
        self.D = self.nh * self.hd
        self.T = self.bt
        self.scale = 1.0 / np.sqrt(self.hd)
        self.BH = self.nh * self.T
        d = dev
        self.qp = self._b(self.BH * self.hd)
        self.kp = self._b(self.BH * self.hd)
        self.vp = self._b(self.BH * self.hd)
        self.op = self._b(self.BH * self.hd)
        self.dop = self._b(self.BH * self.hd)
        self.s = self._b(self.BH * self.T)
        self.p = self._b(self.BH * self.T)
        self.dp = self._b(self.BH * self.T)
        self.ds = self._b(self.BH * self.T)
        self.o = self._b(self.bt * self.D)
        self.dx = self._b(self.bt * self.D)

    def _b(self, n):
        return self.dev.buf(max(4096, n * 2))

    def _pack(self, src, dst):
        K(self.dev)["pack_heads"](self.dev.q, (self.nh, self.T, self.hd), None,
                                  src, dst, np.int32(self.T), np.int32(self.nh),
                                  np.int32(self.hd))

    def _pack_qkv(self, q_buf, k_buf, v_buf):
        K(self.dev)["pack_qkv"](self.dev.q, (self.nh, self.T, self.hd), None,
                                q_buf, k_buf, v_buf, self.qp, self.kp, self.vp,
                                np.int32(self.T), np.int32(self.nh), np.int32(self.hd))

    def _unpack_qkv(self, dq_buf, dk_buf, dv_buf):
        K(self.dev)["unpack_qkv"](self.dev.q, (self.nh, self.T, self.hd), None,
                                  self.dqp, self.dkp, self.dvp,
                                  dq_buf, dk_buf, dv_buf,
                                  np.int32(self.T), np.int32(self.nh), np.int32(self.hd))

    def forward(self, q_buf, k_buf, v_buf):
        d = self.dev
        self._pack_qkv(q_buf, k_buf, v_buf)
        hgemm_batched_ex(self.qp, self.kp, self.s, self.T, self.T, self.hd,
                         A.NO, A.YES, self.hd, self.hd, self.T,
                         self.T * self.hd, self.T * self.hd, self.T * self.T, self.nh)
        K(d)["causal_softmax_batched"](d.q, (self.BH,), (1,), self.s, self.p,
                                       np.int32(self.T), np.int32(self.nh),
                                       np.float32(self.scale))
        hgemm_batched_ex(self.p, self.vp, self.op, self.T, self.hd, self.T,
                         A.NO, A.NO, self.T, self.hd, self.hd,
                         self.T * self.T, self.T * self.hd, self.T * self.hd, self.nh)
        K(d)["unpack_heads"](d.q, (self.nh, self.T, self.hd), None, self.op, self.o,
                             np.int32(self.T), np.int32(self.nh), np.int32(self.hd))
        d.q.finish()
        return self.o

    def backward(self, do_buf):
        d = self.dev
        self._pack(do_buf, self.dop)
        # dP = dO Vᵀ
        hgemm_batched_ex(self.dop, self.vp, self.dp, self.T, self.T, self.hd,
                         A.NO, A.YES, self.hd, self.hd, self.T,
                         self.T * self.hd, self.T * self.hd, self.T * self.T, self.nh)
        K(d)["ds_batched"](d.q, (self.BH,), (1,), self.p, self.dp, self.ds,
                           np.int32(self.T), np.float32(self.scale))
        if getattr(self, "dqk", None) is None:
            self.dqp = self._b(self.BH * self.hd)
            self.dkp = self._b(self.BH * self.hd)
            self.dvp = self._b(self.BH * self.hd)
            self.dqk = True
        dqp, dkp, dvp = self.dqp, self.dkp, self.dvp
        hgemm_batched_ex(self.ds, self.kp, dqp, self.T, self.hd, self.T,
                         A.NO, A.NO, self.T, self.hd, self.hd,
                         self.T * self.T, self.T * self.hd, self.T * self.hd, self.nh)
        hgemm_batched_ex(self.ds, self.qp, dkp, self.T, self.hd, self.T,
                         A.YES, A.NO, self.T, self.hd, self.hd,
                         self.T * self.T, self.T * self.hd, self.T * self.hd, self.nh)
        # dv = Pᵀ dO：直接用 batched GEMM 的 transpose-A，省掉 transpose_bt + 一块 T² buffer。
        hgemm_batched_ex(self.p, self.dop, dvp, self.T, self.hd, self.T,
                         A.YES, A.NO, self.T, self.hd, self.hd,
                         self.T * self.T, self.T * self.hd, self.T * self.hd, self.nh)
        outs = []
        for name in ("_dq", "_dk", "_dv"):
            buf = getattr(self, name, None)
            if buf is None:
                buf = self._b(self.bt * self.D)
                setattr(self, name, buf)
            outs.append(buf)
        self._unpack_qkv(outs[0], outs[1], outs[2])
        d.q.finish()
        return outs[0], outs[1], outs[2]


def check(bt=16, nh=4, hd=8, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    D = nh * hd
    q = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    v = (rng.standard_normal((bt, D)) * 0.5).astype(np.float16)
    do = (rng.standard_normal((bt, D)) * 0.1).astype(np.float16)

    dev = CTX.DevCtx(arena_mb=500, name="fast-attn")
    dev.activate()
    a = FastAttn(dev, bt, nh, hd)
    out = a.forward(dev.up(q), dev.up(k), dev.up(v))
    y = dev.down(out, (bt, D))

    tq = torch.from_numpy(q.astype(np.float32)).requires_grad_(True)
    tk = torch.from_numpy(k.astype(np.float32)).requires_grad_(True)
    tv = torch.from_numpy(v.astype(np.float32)).requires_grad_(True)

    def heads(x):
        return x.view(bt, nh, hd).transpose(0, 1)

    sc = (heads(tq) @ heads(tk).transpose(-1, -2)) / (hd ** 0.5)
    mask = torch.tril(torch.ones(bt, bt, dtype=torch.bool))
    pr = torch.softmax(sc.masked_fill(~mask, float("-inf")), dim=-1)
    o = (pr @ heads(tv)).transpose(0, 1).reshape(bt, D)
    rel_f = float(np.linalg.norm(y.astype(np.float32) - o.detach().numpy())
                  / np.linalg.norm(o.detach().numpy()))
    o.backward(torch.from_numpy(do.astype(np.float32)))
    dq, dk, dv = a.backward(dev.up(do))
    def rel(x, t):
        return float(np.linalg.norm(dev.down(x, (bt, D)).astype(np.float32) - t.numpy())
                     / np.linalg.norm(t.numpy()))
    print("  fast attn bt=%d nh=%d hd=%d | fwd %.2e dq %.2e dk %.2e dv %.2e"
          % (bt, nh, hd, rel_f, rel(dq, tq.grad), rel(dk, tk.grad), rel(dv, tv.grad)))
    return max(rel_f, rel(dq, tq.grad), rel(dk, tk.grad), rel(dv, tv.grad)) < 5e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    x = ap.parse_args()
    if not x.check:
        print(__doc__)
        return 0
    return 0 if check() else 1


if __name__ == "__main__":
    raise SystemExit(main())
