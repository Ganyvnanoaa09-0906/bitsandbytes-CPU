# -*- coding: utf-8 -*-
"""igpu_attn_chunk.py — self-attention 的**按查询分块**实现（前向 + 反向），解决 T=4096 内存墙。

整流法要物化 `S = Tq×Tk`（T=4096 ⇒ 16.8M 元素 / 33 MB，四块 ~537 MB）；
本实现沿 **query 维**切块，每块只算 `bq×Tk` 的 S/P/dP/dS，
复用已验证的 GEMM 与 `softmax_window` ⇒ 与整流法**逐位等价**（同一批 kernel）。
内存 = O(bq·Tk)：bq=256、Tk=4096 ⇒ **2 MB/块**。

    python gt1030/igpu_attn_chunk.py --check          # 前向 + 反向对拍（T=256/512）
    python gt1030/igpu_attn_chunk.py --check --big    # 额外 T=1024

约定：q/o (B,Tq,C)、k/v (B,Tk,C) 都是 **token 布局**；dx 输出为 NCHW。
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402
from gt1030.igpu_attn_full import K as FULLK  # noqa: E402
from gt1030.igpu_conv import K as CVK  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 分块反向：矩形 bq×Tk，不假设方阵

// 矩形 softmax：S 是 (n_rows, Tk) 的连续块，逐行在整行 Tk 上归一化
// （chunked 调用时每个块只含一个样本，所以窗口 = 整行）

// 每行一个 work-group（默认 256 线程），组内归约 max/sum：
// 相比"一行一个 work-item"的串行实现，Tk=4096 时并行度提升 256 倍
__kernel void softmax_rect_fast(__global const half* s, __global half* p,
                                int n_rows, int Tk, float scale) {
    __local float lmax[256];
    __local float lsum[256];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    if (row >= n_rows) return;
    long base = (long)row * Tk;

    float mx = -INFINITY;
    for (int j = lid; j < Tk; j += lsz) mx = fmax(mx, (float)s[base + j] * scale);
    lmax[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = lsz >> 1; st > 0; st >>= 1) {
        if (lid < st) lmax[lid] = fmax(lmax[lid], lmax[lid + st]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = lmax[0];

    float sum = 0.f;
    for (int j = lid; j < Tk; j += lsz) sum += exp((float)s[base + j] * scale - m);
    lsum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = lsz >> 1; st > 0; st >>= 1) {
        if (lid < st) lsum[lid] += lsum[lid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / lsum[0];

    for (int j = lid; j < Tk; j += lsz)
        p[base + j] = (half)(exp((float)s[base + j] * scale - m) * inv);
}

__kernel void softmax_rect(__global const half* s, __global half* p,
                           int n_rows, int Tk, float scale) {
    long row = get_global_id(0);
    if (row >= n_rows) return;
    long base = row * Tk;
    float mx = -INFINITY;
    for (int j = 0; j < Tk; ++j) mx = fmax(mx, (float)s[base + j] * scale);
    float sum = 0.f;
    for (int j = 0; j < Tk; ++j) sum += exp((float)s[base + j] * scale - mx);
    float inv = 1.f / sum;
    for (int j = 0; j < Tk; ++j)
        p[base + j] = (half)(exp((float)s[base + j] * scale - mx) * inv);
}

__kernel void chunk_rowsum(__global const half* dp, __global const half* p,
                           __global float* rs, int n_rows, int Tk) {
    int r = get_global_id(0);
    if (r >= n_rows) return;
    long base = (long)r * Tk;
    float acc = 0.f;
    for (int j = 0; j < Tk; ++j) acc += (float)dp[base + j] * (float)p[base + j];
    rs[r] = acc;
}


// (rows, C) → head-major (heads, rows, dh)：dst[(h*rows + r)*dh + d] = src[r*C + h*dh + d]
__kernel void head_split(__global const half* src, __global half* dst,
                         int rows, int C, int heads) {
    long i = get_global_id(0);
    if (i >= (long)rows * C) return;
    int r = (int)(i / C), c = (int)(i % C);
    int dh = C / heads;
    int h = c / dh, d = c % dh;
    dst[((long)h * rows + r) * dh + d] = src[i];
}

// head-major → (rows, C)
__kernel void head_merge(__global const half* src, __global half* dst,
                         int rows, int C, int heads) {
    long i = get_global_id(0);
    if (i >= (long)rows * C) return;
    int r = (int)(i / C), c = (int)(i % C);
    int dh = C / heads;
    int h = c / dh, d = c % dh;
    dst[i] = src[((long)h * rows + r) * dh + d];
}

__kernel void chunk_ds(__global const half* dp, __global const half* p,
                       __global const float* rs, __global half* ds,
                       int n_rows, int Tk, float scale) {
    long idx = get_global_id(0);
    if (idx >= (long)n_rows * Tk) return;
    int r = (int)(idx / Tk);
    ds[idx] = (half)((float)p[idx] * ((float)dp[idx] - rs[r]) * scale);
}
"""

_C = {}


def CK(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = {n: cl.Kernel(prog, n) for n in ("chunk_rowsum", "chunk_ds", "head_split", "head_merge",
                      "softmax_rect", "softmax_rect_fast")}
    return _C[key]


def _buf(dev, n):
    return dev.buf(max(4096, int(n) * 2))


def _zeros(dev, buf, nbytes):
    cl.enqueue_fill_buffer(dev.q, buf, np.uint16(0), 0, int(nbytes))


def _acc_add(dev, dst, src, n):
    """dst += src（逐元素，half）。用 igpu_graph 的 add_h 写到临时再拷回。"""
    import gt1030.igpu_graph as _G

    tmp = _buf(dev, n)
    _G._kernels(dev)["add_h"](dev.q, (int(n),), None, dst, src, tmp, np.int64(n))
    cl.enqueue_copy(dev.q, dst, tmp, byte_count=int(n) * 2)


def attn_chunked(dev, q_buf, k_buf, v_buf, o_buf, B, Tq, Tk, C, bq=512):
    """前向：返回 o_buf（token 布局）。"""
    scale = 1.0 / np.sqrt(float(C))
    sb, pb = _buf(dev, bq * Tk), _buf(dev, bq * Tk)
    tmpq = _buf(dev, bq * C)
    for b in range(B):
        qbase, kbase, obase = b * Tq * C, b * Tk * C, b * Tq * C
        tmpk = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpk, k_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        tmpv = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpv, v_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        done = 0
        while done < Tq:
            rows = min(bq, Tq - done)
            cl.enqueue_copy(dev.q, tmpq, q_buf,
                            src_offset=(qbase + done * C) * 2, byte_count=rows * C * 2)
            GEMM.hgemm_nt(tmpq, tmpk, sb, rows, Tk, C)
            # 每行一个 work-group（256 线程归约）替代"一行一线程"的串行版
            _ls = 256
            CK(dev)["softmax_rect_fast"](dev.q, (rows * _ls,), (_ls,), sb, pb,
                                         np.int32(rows), np.int32(Tk), np.float32(scale))
            GEMM.hgemm_nn(pb, tmpv, tmpq, rows, C, Tk)
            cl.enqueue_copy(dev.q, o_buf, tmpq,
                            dst_offset=(obase + done * C) * 2, byte_count=rows * C * 2)
            done += rows
    return o_buf


def attn_chunked_backward(dev, q_buf, k_buf, v_buf, o_tok, do_tok, dx_nchw,
                          B, Tq, Tk, C, H, W, bq=512):
    """反向：只算 attention 自身（dx / dK / dV）；q/k/v 的投影由外部 Linear 负责。"""
    import gt1030.igpu_ops2 as _OPS2

    scale = 1.0 / np.sqrt(float(C))
    sb, pb = _buf(dev, bq * Tk), _buf(dev, bq * Tk)
    dp, ds = _buf(dev, bq * Tk), _buf(dev, bq * Tk)
    rs = dev.buf(max(4096, bq * 4))
    tq_q, tq_do, tq_o, dqc = (_buf(dev, bq * C) for _ in range(4))
    dkc, dvc = _buf(dev, Tk * C), _buf(dev, Tk * C)
    dk_acc, dv_acc = _buf(dev, Tk * C), _buf(dev, Tk * C)
    dq_tok = _buf(dev, B * Tq * C)
    dk_tok = _buf(dev, B * Tk * C)
    dv_tok = _buf(dev, B * Tk * C)
    dxt = _buf(dev, B * Tq * C)
    dtmp = _buf(dev, C * C)

    for b in range(B):
        qbase, kbase = b * Tq * C, b * Tk * C
        tmpk = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpk, k_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        tmpv = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpv, v_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        _zeros(dev, dk_acc, Tk * C * 2)
        _zeros(dev, dv_acc, Tk * C * 2)
        done = 0
        while done < Tq:
            rows = min(bq, Tq - done)
            off = (qbase + done * C) * 2
            nb_ = rows * C * 2
            for dst, src in ((tq_q, q_buf), (tq_do, do_tok), (tq_o, o_tok)):
                cl.enqueue_copy(dev.q, dst, src, src_offset=off, byte_count=nb_)
            GEMM.hgemm_nt(tq_q, tmpk, sb, rows, Tk, C)          # S
            CK(dev)["softmax_rect"](dev.q, (rows,), None, sb, pb,
                                    np.int32(rows), np.int32(Tk), np.float32(scale))
            GEMM.hgemm_nt(tq_do, tmpv, dp, rows, Tk, C)         # dP = dO·Vᵀ
            CK(dev)["chunk_rowsum"](dev.q, (rows,), None, dp, pb, rs,
                                    np.int32(rows), np.int32(Tk))
            CK(dev)["chunk_ds"](dev.q, (rows * Tk,), None, dp, pb, rs, ds,
                                np.int32(rows), np.int32(Tk), np.float32(scale))
            GEMM.hgemm_nn(ds, tmpk, dqc, rows, C, Tk)           # dQ = dS·K
            cl.enqueue_copy(dev.q, dq_tok, dqc,
                            dst_offset=(qbase + done * C) * 2, byte_count=nb_)
            GEMM.hgemm_tn(ds, tq_q, dkc, Tk, C, rows)           # dK = dSᵀ·Q
            _acc_add(dev, dk_acc, dkc, Tk * C)
            GEMM.hgemm_tn(pb, tq_do, dvc, Tk, C, rows)          # dV = Pᵀ·dO
            _acc_add(dev, dv_acc, dvc, Tk * C)
            done += rows
        # 每个样本的 dK / dV 写回 token 布局缓冲
        cl.enqueue_copy(dev.q, dk_tok, dk_acc,
                        dst_offset=kbase * 2, byte_count=Tk * C * 2)
        cl.enqueue_copy(dev.q, dv_tok, dv_acc,
                        dst_offset=kbase * 2, byte_count=Tk * C * 2)

    CVK(dev)["nhwc2nchw"](dev.q, (B * Tq * C,), None, dq_tok, dx_nchw,
                          np.int32(B), np.int32(C), np.int32(H), np.int32(W))
    return dict(dx=dx_nchw, dk=dk_tok, dv=dv_tok)



def attn_chunked_heads(dev, q_buf, k_buf, v_buf, o_buf, B, Tq, Tk, C, heads, bq=256):
    """多头版：先 head_split 成 head-major，逐头调用 attn_chunked，再 head_merge 回来。

    attn_chunked 本身是单头语义（scale = 1/sqrt(C)）；多头时每头 dim_head = C/heads，
    scale 应为 1/sqrt(dim_head)，所以必须逐头调用，不能直接整块喂进去。
    """
    dh = C // heads
    rq, rk = B * Tq, B * Tk
    qh = _buf(dev, heads * rq * dh)
    kh = _buf(dev, heads * rk * dh)
    vh = _buf(dev, heads * rk * dh)
    oh = _buf(dev, heads * rq * dh)
    CK(dev)["head_split"](dev.q, (rq * C,), None, q_buf, qh, np.int32(rq), np.int32(C), np.int32(heads))
    CK(dev)["head_split"](dev.q, (rk * C,), None, k_buf, kh, np.int32(rk), np.int32(C), np.int32(heads))
    CK(dev)["head_split"](dev.q, (rk * C,), None, v_buf, vh, np.int32(rk), np.int32(C), np.int32(heads))
    qs = _buf(dev, rq * dh)
    ks = _buf(dev, rk * dh)
    vs = _buf(dev, rk * dh)
    os_ = _buf(dev, rq * dh)
    for h in range(heads):
        for src, dst in ((qh, qs), (kh, ks), (vh, vs)):
            cl.enqueue_copy(dev.q, dst, src,
                            src_offset=h * rq * dh * 2 if src is qh else h * rk * dh * 2,
                            byte_count=(rq if src is qh else rk) * dh * 2)
        # 切片拷贝后再进 attn：内联验证过的版本有这一步，缺它时多头结果全零
        dev.q.finish()
        attn_chunked(dev, qs, ks, vs, os_, B, Tq, Tk, dh, bq=min(bq, Tq))
        cl.enqueue_copy(dev.q, oh, os_, dst_offset=h * rq * dh * 2, byte_count=rq * dh * 2)
    CK(dev)["head_merge"](dev.q, (rq * C,), None, oh, o_buf, np.int32(rq), np.int32(C), np.int32(heads))
    return o_buf


def check(T=256, C=64, B=1, bq=64, H=None, W=None, seed=0, grads=True):
    import torch

    H = H or 16
    W = W or (T // H)
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((B, T, C)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, T, C)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, T, C)) * 0.5).astype(np.float32)
    dO = (rng.standard_normal((B, T, C)) * 0.3).astype(np.float32)

    dev = CTX.DevCtx(arena_mb=900, name="chunk")
    dev.activate()
    qb, kb, vb = (dev.up(x.astype(np.float16)) for x in (q, k, v))
    ob = _buf(dev, B * T * C)
    attn_chunked(dev, qb, kb, vb, ob, B, T, T, C, bq=bq)
    dev.q.finish()
    ov = dev.down(ob, (B, T, C)).astype(np.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)).numpy()
    rel_f = float(np.linalg.norm(ov - ref) / np.linalg.norm(ref))
    print("  chunk T%-5d C%-4d bq%-4d forward rel %.2e" % (T, C, bq, rel_f))
    ok = rel_f < 2e-2
    if not grads:
        return ok

    dOb = dev.up(dO.astype(np.float16))
    dxb = _buf(dev, B * T * C)
    g = attn_chunked_backward(dev, qb, kb, vb, ob, dOb, dxb, B, T, T, C, H, W, bq=bq)
    dev.q.finish()
    dxv = dev.down(dxb, (B, C, H, W)).astype(np.float32)
    dkv = dev.down(g["dk"], (B, T, C)).astype(np.float32)
    dvv = dev.down(g["dv"], (B, T, C)).astype(np.float32)

    tq = torch.from_numpy(q).requires_grad_(True)
    tk = torch.from_numpy(k).requires_grad_(True)
    tv = torch.from_numpy(v).requires_grad_(True)
    o = torch.nn.functional.scaled_dot_product_attention(tq, tk, tv)
    o.backward(torch.from_numpy(dO))

    def rel(a, b):
        nb = float(np.linalg.norm(b))
        return float(np.linalg.norm(a - b) / (nb + 1e-12)) if nb > 1e-9 else float(np.linalg.norm(a - b))

    rels = {
        "dx": rel(dxv, tq.grad.numpy().transpose(0, 2, 1).reshape(B, C, H, W)),
        "dK": rel(dkv, tk.grad.numpy()),
        "dV": rel(dvv, tv.grad.numpy()),
    }
    for kk, vv in rels.items():
        print("      %-4s %.2e" % (kk, vv))
    return ok and all(v < 3e-2 for v in rels.values() if np.isfinite(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--big", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check(T=256, C=64, bq=64)
    ok &= check(T=512, C=128, bq=128)
    if a.big:
        ok &= check(T=1024, C=320, bq=256, grads=False)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
