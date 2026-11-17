# -*- coding: utf-8 -*-
"""igpu_graph.py — dev13 P2 第一块：**设备侧执行器 + 算子注册表**（还没有 FX 前端）。

目标（见 DEV13_PLAN.md A1/A3）：权重、梯度、优化器状态全程在 device；
执行器只负责"按计划跑 op、逆序反传、常驻更新"，**不碰 host**。

本文件先做**最小闭环**：用一个手写计划跑 `Linear → GELU → Linear → MSE`，
与 torch 同参同数据对拍 5 步，验证 op 注册表与反传公式正确。
FX 前端（把任意 nn.Module 编成同一份计划）是下一步，二者解耦。

    python gt1030/igpu_graph.py --check        # 端到端对拍 5 步
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import ctypes  # noqa: E402

import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.igpu_ctx as CTX  # noqa: E402
import gt1030.igpu_nn_ops as NN  # noqa: E402

A = MM.A
L = MM.L

EXTRA = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 8-bit 优化器状态的数值护栏（实测：缺了会在大约第 9 步出 NaN）
//   1) v 的解码地板：小元素的码会舍入到 0，sqrt(0/bc2)=0 后分母只剩 eps，
//      更新被放大约 1e3 倍，一两步后权重就溢出；
//   2) m/v 的上下限：拦住 inf 进入动量（inf/inf 会直接产生 NaN）。
#define Q8_V_FLOOR 1e-6f
#define Q8_M_CLAMP 1e6f
#define Q8_V_MIN   1e-9f
#define Q8_V_MAX   1e6f
// y = a + b（逐元素）
__kernel void add_h(__global const half* a, __global const half* b, __global half* y, long n) {
    long i = get_global_id(0);
    if (i < n) y[i] = (half)((float)a[i] + (float)b[i]);
}
// dy 累加：g += d（逐元素）
__kernel void acc_h(__global half* g, __global const half* d, long n) {
    long i = get_global_id(0);
    if (i < n) g[i] = (half)((float)g[i] + (float)d[i]);
}
// MSE 对 y 的梯度：dy = 2*(y - t)/n，并用 S 缩放写回 fp16（避免下溢）
__kernel void mse_grad_h(__global const half* y, __global const half* t, __global half* dy,
                         long n, float inv_n, float scale) {
    long i = get_global_id(0);
    if (i < n) dy[i] = (half)(2.f * ((float)y[i] - (float)t[i]) * inv_n * scale);
}
// AdamW（fp32 master + fp16 副本，梯度可以是缩放过的）——自带实现，避免依赖 MM.KSRC
__kernel void adamw1(__global float* w, __global float* m, __global float* v,
                     __global const half* g, __global half* wh, int n,
                     float lr, float b1, float b2, float eps, float wd, int t) {
    int i = get_global_id(0);
    if (i >= n) return;
    float gt = (float)g[i];
    float bc1 = 1.f - pow(b1, (float)t);
    float bc2 = 1.f - pow(b2, (float)t);
    m[i] = b1 * m[i] + (1.f - b1) * gt;
    v[i] = b2 * v[i] + (1.f - b2) * gt * gt;
    float upd = lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + eps);
    w[i] = w[i] - upd - lr * wd * w[i];
    wh[i] = (half)w[i];
}


// 8-bit 状态 AdamW：每 workgroup 处理一个 block(256)，m/v 为 int8 码，absmax 为 fp32
__kernel void adamw1_q8(__global float* w, __global uchar* m8, __global uchar* v8,
                        __global float* amax_m, __global float* amax_v,
                        __global const half* g, __global half* wh, int n,
                        float lr, float b1, float b2, float eps, float wd, int t) {
    int blk = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int base = blk * 256;
    int i = base + lid;
    __local float redm[256];
    __local float redv[256];
    __local float sm, sv;
    float nm = 0.f, nv = 0.f, gt = 0.f;
    float mm = amax_m[blk], vv = amax_v[blk];
    if (i < n) {
        gt = (float)g[i];
        float m_old = (((float)m8[i]) - 128.f) / 127.f * mm;
        float v_old = (((float)v8[i]) - 128.f) / 127.f * vv;
        v_old = fmax(v_old, Q8_V_FLOOR);      // 见文件头：差 1e-12 时第 9 步必 NaN
        nm = b1 * m_old + (1.f - b1) * gt;
        nv = b2 * v_old + (1.f - b2) * gt * gt;
    }
    redm[lid] = fabs(nm);
    redv[lid] = fabs(nv);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { redm[lid] = fmax(redm[lid], redm[lid + off]);
                         redv[lid] = fmax(redv[lid], redv[lid + off]); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { sm = fmax(redm[0], 1e-12f); sv = fmax(redv[0], 1e-12f);
                    amax_m[blk] = sm; amax_v[blk] = sv; }
    barrier(CLK_LOCAL_MEM_FENCE);
    sm = amax_m[blk]; sv = amax_v[blk];
    if (i < n) {
        float bc1 = 1.f - pow(b1, (float)t);
        float bc2 = 1.f - pow(b2, (float)t);
        nm = clamp(nm, -Q8_M_CLAMP, Q8_M_CLAMP);   // 拦住 inf，避免下一步 NaN
        nv = clamp(nv, Q8_V_MIN, Q8_V_MAX);        // 下界同时保护 sqrt(nv/bc2)
        float upd = lr * (nm / bc1) / (sqrt(nv / bc2) + eps);
        w[i] = w[i] - upd - lr * wd * w[i];
        wh[i] = (half)w[i];
        // uchar 编码：0..255 表示 -1..+1（128 = 0）；NaN/Inf 一律落回中性值
        float cm = isfinite(nm) && isfinite(sm) ? round(nm / sm * 127.f) + 128.f : 128.f;
        float cv = isfinite(nv) && isfinite(sv) ? round(nv / sv * 127.f) + 128.f : 128.f;
        m8[i] = (uchar)clamp(cm, 0.f, 255.f);
        v8[i] = (uchar)clamp(cv, 0.f, 255.f);
    }
}


// 4-bit 状态 AdamW：m/v 各占半个字节（两个码一个字节），每 256 元素一个 fp32 absmax
inline float nib4_get(__global const uchar* buf, long i, float am) {
    uchar b = buf[i >> 1];
    int code = (i & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    return ((float)code - 8.f) / 7.f * am;
}

__kernel void adamw1_q4(__global float* w, __global uchar* m4, __global uchar* v4,
                        __global float* amax_m, __global float* amax_v,
                        __global const half* g, __global half* wh, int n,
                        float lr, float b1, float b2, float eps, float wd, int t) {
    int blk = get_group_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    int base = blk * 256;
    int i = base + lid;
    __local float redm[256];
    __local float redv[256];
    float nm = 0.f, nv = 0.f;
    if (i < n) {
        float gt = (float)g[i];
        float m_old = nib4_get(m4, i, amax_m[blk]);
        float v_old = nib4_get(v4, i, amax_v[blk]);
        v_old = fmax(v_old, Q8_V_FLOOR);      // 4-bit 码更粗，同样需要地板
        nm = b1 * m_old + (1.f - b1) * gt;
        nv = b2 * v_old + (1.f - b2) * gt * gt;
        float bc1 = 1.f - pow(b1, (float)t);
        float bc2 = 1.f - pow(b2, (float)t);
        nm = clamp(nm, -Q8_M_CLAMP, Q8_M_CLAMP);   /* Q4_CLAMPED */
        nv = clamp(nv, Q8_V_MIN, Q8_V_MAX);        /* Q4_CLAMPED */
        float upd = lr * (nm / bc1) / (sqrt(nv / bc2) + eps);
        w[i] = w[i] - upd - lr * wd * w[i];
        wh[i] = (half)w[i];
    }
    redm[lid] = fabs(nm);
    redv[lid] = fabs(nv);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) { redm[lid] = fmax(redm[lid], redm[lid + off]);
                         redv[lid] = fmax(redv[lid], redv[lid + off]); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { amax_m[blk] = fmax(redm[0], 1e-12f); amax_v[blk] = fmax(redv[0], 1e-12f); }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i < n) {
        float am = amax_m[blk], av = amax_v[blk];
        int cm = (int)clamp(round(nm / am * 7.f) + 8.f, 0.f, 15.f);
        int cv = (int)clamp(round(nv / av * 7.f) + 8.f, 0.f, 15.f);
        uchar bm = m4[i >> 1], bv = v4[i >> 1];
        m4[i >> 1] = (i & 1) ? (uchar)((bm & 0x0F) | (cm << 4)) : (uchar)((bm & 0xF0) | cm);
        v4[i >> 1] = (i & 1) ? (uchar)((bv & 0x0F) | (cv << 4)) : (uchar)((bv & 0xF0) | cv);
    }
}

// 逐元素乘（反向用）
__kernel void mul_h(__global const half* a, __global const half* b, __global half* y, long n) {
    long i = get_global_id(0);
    if (i < n) y[i] = (half)((float)a[i] * (float)b[i]);
}
"""

_K = {}


def _kernels(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _K:
        prog = cl.Program(dev.ctx, EXTRA).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _K[key] = {n: cl.Kernel(prog, n) for n in
                   ("add_h", "acc_h", "mse_grad_h", "mul_h")}
    return _K[key]


class T(object):
    """一个设备张量：buffer + 形状（fp16 激活）。"""

    __slots__ = ("buf", "shape", "grad", "requires_grad")

    def __init__(self, buf, shape, requires_grad=False):
        self.buf, self.shape = buf, tuple(shape)
        self.grad = None
        self.requires_grad = requires_grad

    @property
    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


class Param(T):
    """带 fp32 master + Adam 状态的参数（常驻）。"""

    __slots__ = ("master", "mom", "vel", "weight")

    def __init__(self, dev, buf, shape, master, mom, vel):
        super().__init__(buf, shape, requires_grad=True)
        self.master, self.mom, self.vel = master, mom, vel


class Session(object):
    """一个 DevCtx + arena + 参数字典 + 执行器。"""

    def __init__(self, arena_mb=900, name="graph"):
        self.dev = CTX.DevCtx(arena_mb=arena_mb, name=name)
        self.dev.activate()
        self.K = _kernels(self.dev)
        self.params = {}          # name -> Param
        self.adam_single = self._adam_kernel()

    # ---------------- 参数 ----------------
    def _adam_kernel_q8(self):
        prog = cl.Program(self.dev.ctx, EXTRA).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        return cl.Kernel(prog, "adamw1_q8")

    def _adam_kernel(self):
        # 自带 adamw1（见 EXTRA）：不再从 MM.KSRC 编译，避免源码里的非 GBK 字符在 Windows 上失败
        prog = cl.Program(self.dev.ctx, EXTRA).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        return cl.Kernel(prog, "adamw1")

    def add_param(self, name, w_np):
        """w_np: (out, in) 的 fp32 权重；建 fp32 master + fp16 副本 + m/v。"""
        d = self.dev
        w = np.ascontiguousarray(w_np, dtype=np.float32)
        flat = w.reshape(-1)
        n = flat.size
        nbytes4 = max(4096, n * 4)
        master = d.buf(nbytes4)
        mom = d.buf(nbytes4)
        vel = d.buf(nbytes4)
        half = d.buf(max(4096, n * 2))
        cl.enqueue_copy(d.q, master, flat)
        cl.enqueue_copy(d.q, mom, np.zeros(n, np.float32))
        cl.enqueue_copy(d.q, vel, np.zeros(n, np.float32))
        cl.enqueue_copy(d.q, half, flat.astype(np.float16))
        d.q.finish()
        p = Param(d, half, w.shape, master, mom, vel)
        p.grad = d.buf(max(4096, n * 2))       # fp16 梯度缓冲（常驻）
        self.params[name] = p
        return p

    def set_param(self, name, w_np):
        """覆盖参数值（master + fp16 副本），用于与 torch 同步初值。"""
        d = self.dev
        p = self.params[name]
        flat = np.ascontiguousarray(w_np, dtype=np.float32).reshape(-1)
        cl.enqueue_copy(d.q, p.master, flat)
        cl.enqueue_copy(d.q, p.buf, flat.astype(np.float16))
        cl.enqueue_copy(d.q, p.grad, np.zeros(flat.size, np.float16))
        d.q.finish()

    def get_param(self, name):
        p = self.params[name]
        out = np.empty(p.master.size // 4, np.float32)
        cl.enqueue_copy(self.dev.q, out, p.master, is_blocking=True)
        return out.reshape(p.shape)

    def zero_grads(self):
        d = self.dev
        for p in self.params.values():
            cl.enqueue_copy(d.q, p.grad, np.zeros(p.numel, np.float16))
        d.q.finish()

    def adam_step(self, lr, t, eps=1e-8, wd=0.0):
        d = self.dev
        if not hasattr(self, "_k8"):
            self._k8 = None
        for p in self.params.values():
            if getattr(p, "state_bits", 32) == 4:
                if getattr(self, "_k4", None) is None:
                    prog = cl.Program(self.dev.ctx, EXTRA).build(
                        options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
                    self._k4 = cl.Kernel(prog, "adamw1_q4")
                nblocks = (p.n + 255) // 256
                self._k4(self.dev.q, (nblocks * 256,), (256,), p.master, p.mom8, p.vel8,
                         p.mom, p.vel, p.grad, p.buf, np.int32(p.n), np.float32(lr),
                         np.float32(0.9), np.float32(0.999), np.float32(eps),
                         np.float32(wd), np.int32(t))
                continue
            if getattr(p, "state_bits", 32) == 8:
                if self._k8 is None:
                    self._k8 = self._adam_kernel_q8()
                nblocks = (p.numel + 255) // 256
                self._k8(d.q, (nblocks * 256,), (256,), p.master, p.mom8, p.vel8,
                         p.mom, p.vel, p.grad, p.buf, np.int32(p.numel),
                         np.float32(lr), np.float32(0.9), np.float32(0.999),
                         np.float32(eps), np.float32(wd), np.int32(t))
            else:
                self.adam_single(d.q, (p.numel,), None, p.master, p.mom, p.vel, p.grad,
                                 p.buf, np.int32(p.numel), np.float32(lr),
                                 np.float32(0.9), np.float32(0.999), np.float32(eps),
                                 np.float32(wd), np.int32(t))
        d.q.finish()


# ---------------------------------------------------------------- op 注册表
class Op(object):
    """一个 op 的 fwd/bwd。输入/输出都是 T（device buffer）。"""

    def forward(self, sess, *args, **kw):
        raise NotImplementedError

    def backward(self, sess, *args, **kw):
        raise NotImplementedError


class Linear(Op):
    def __init__(self, name, m):
        self.name, self.m = name, int(m)

    def forward(self, sess, x, w, b=None):
        out = T(sess.dev.buf(max(4096, x.shape[0] * self.m * 2)), (x.shape[0], self.m))
        L.hgemm_nt(x.buf, w.buf, out.buf, x.shape[0], self.m, w.shape[1])
        if b is not None:
            # bias 逐行加（复用 add_h：把 bias 视作 (1, m) 广播 → 用 kernel 循环实现）
            raise NotImplementedError("bias 广播待实现（先支持 bias=False）")
        sess.dev.q.finish()
        return out

    def backward(self, sess, x, w, dy):
        M, K, N = x.shape[0], w.shape[1], self.m
        dx = T(sess.dev.buf(max(4096, M * K * 2)), (M, K))
        dW = T(sess.dev.buf(max(4096, N * K * 2)), (N, K))
        L.hgemm_nn(dy.buf, w.buf, dx.buf, M, K, N)
        L.hgemm_tn(dy.buf, x.buf, dW.buf, N, K, M)
        sess.dev.q.finish()
        return dx, dW


class Gelu(Op):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def forward(self, sess, x):
        out = T(sess.dev.buf(max(4096, x.numel * 2)), x.shape)
        NN._kernels()["gelu_fwd"](sess.dev.q, (x.numel,), None, x.buf, out.buf,
                                   np.int64(x.numel))
        sess.dev.q.finish()
        return out

    def backward(self, sess, x, dy):
        dx = T(sess.dev.buf(max(4096, x.numel * 2)), x.shape)
        NN._kernels()["gelu_bwd"](sess.dev.q, (x.numel,), None, x.buf, dy.buf, dx.buf,
                                  np.int64(x.numel))
        sess.dev.q.finish()
        return (dx,)


class Add(Op):
    def forward(self, sess, a, b):
        out = T(sess.dev.buf(max(4096, a.numel * 2)), a.shape)
        _kernels(sess.dev)["add_h"](sess.dev.q, (a.numel,), None, a.buf, b.buf, out.buf,
                                    np.int64(a.numel))
        sess.dev.q.finish()
        return out

    def backward(self, sess, a, b, dy):
        return (dy, dy)      # 加法两侧梯度相同（上层再做累加）


def acc_grad(sess, target_buf, dy_buf, n):
    """把 dy 累加进 target 的 fp16 梯度缓冲。"""
    _kernels(sess.dev)["acc_h"](sess.dev.q, (n,), None, target_buf, dy_buf, np.int64(n))


# ---------------------------------------------------------------- 最小闭环测试
def cosine_lr(t, total, lr_max, lr_min=0.0, warmup=0):
    """warmup 线性升温 → 余弦退火到 lr_min。t 从 1 计。"""
    import math
    if warmup and t <= warmup:
        return lr_max * t / float(warmup)
    span = max(1, total - warmup)
    prog = min(1.0, (t - warmup) / float(span))
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))


def sgdr_lr(t, t0, lr_max, lr_min=0.0, t_mult=2, warmup=0):
    """SGDR：余弦退火 + 周期性热重启（Loshchilov & Hutter 2017）。

    t      : 当前步（从 1 计）
    t0     : 第一个周期的长度
    t_mult : 每个周期长度倍率（1 = 等长周期）
    warmup : 前 warmup 步线性升温（在第一个周期内）

    返回 (lr, cycle_index, t_in_cycle)。
    周期长度序列：t0, t0*t_mult, t0*t_mult^2, ...
    """
    import math
    if warmup and t <= warmup:
        return lr_max * t / float(warmup), 0, t
    t -= warmup
    cycle, length = 0, float(t0)
    while t > length:
        t -= length
        length *= t_mult
        cycle += 1
    # 周期内余弦：从 lr_max 降到 lr_min
    frac = (t - 1) / max(1.0, length - 1)
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * frac))
    return lr, cycle, int(t)


def check(steps=5, d_in=64, d_h=128, batch=32, scale=65536.0):
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((d_h, d_in)) * (1.0 / np.sqrt(d_in))).astype(np.float32)
    w2 = (rng.standard_normal((d_in, d_h)) * (1.0 / np.sqrt(d_h))).astype(np.float32)
    x = (rng.standard_normal((batch, d_in)) * 0.3).astype(np.float16)
    tgt = (rng.standard_normal((batch, d_in)) * 0.1).astype(np.float16)

    lr = 1e-3
    sess = Session(arena_mb=400, name="graph-check")
    p1 = sess.add_param("w1", w1)
    p2 = sess.add_param("w2", w2)
    lin1, act, lin2 = Linear("w1", d_h), Gelu((batch, d_h)), Linear("w2", d_in)

    xb = T(sess.dev.up(x), x.shape)
    tb = T(sess.dev.up(tgt), tgt.shape)
    dy_buf = sess.dev.buf(max(4096, batch * d_in * 2))

    # torch 侧同参（用 Adam，与 device 的 adamw1 对齐；eps 也按 scale 缩放）
    import torch.nn as nn
    xf = torch.from_numpy(x.astype(np.float32))
    tf = torch.from_numpy(tgt.astype(np.float32))
    tw1 = nn.Parameter(torch.from_numpy(w1.copy()))
    tw2 = nn.Parameter(torch.from_numpy(w2.copy()))
    topt = torch.optim.Adam([tw1, tw2], lr=lr, eps=1e-8)

    losses_gpu, losses_torch = [], []
    for step in range(1, steps + 1):
        sess.zero_grads()
        h = lin1.forward(sess, xb, p1)
        a = act.forward(sess, h)
        y = lin2.forward(sess, a, p2)
        # MSE + 梯度（fp16 下用 scale 防下溢，与 L1 一致：eps 同步 ×scale）
        n = y.numel
        _kernels(sess.dev)["mse_grad_h"](sess.dev.q, (n,), None, y.buf, tb.buf, dy_buf,
                                         np.int64(n), np.float32(1.0 / n), np.float32(scale))
        sess.dev.q.finish()
        ynp = sess.dev.down(y.buf, y.shape)
        tnp = tgt.astype(np.float32)
        losses_gpu.append(float(((ynp.astype(np.float32) - tnp) ** 2).mean()))

        # 反传：dy 需除以 scale 才等于真实梯度（Adam 与 scale 无关，但这里 GEMM 用它）
        d_a = lin2.backward(sess, a, p2, T(dy_buf, y.shape))
        d_h = act.backward(sess, h, d_a[0])
        dx, dw1 = lin1.backward(sess, xb, p1, d_h[0])
        acc_grad(sess, p2.grad, d_a[1].buf, p2.numel)
        acc_grad(sess, p1.grad, dw1.buf, p1.numel)
        sess.dev.q.finish()
        sess.adam_step(lr, step, eps=1e-8 * scale)

        # torch
        topt.zero_grad()
        ty = F.linear(F.gelu(F.linear(xf, tw1)), tw2)
        tloss = F.mse_loss(ty, tf)
        tloss.backward()
        losses_torch.append(float(tloss.detach()))
        topt.step()

    print("  step   device      torch       rel")
    worst = 0.0
    for i, (a, b) in enumerate(zip(losses_gpu, losses_torch), 1):
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        print("  %4d  %10.6f  %10.6f  %8.2e" % (i, a, b, rel))
    print("RESULT " + ("PASS" if worst < 5e-2 else "FAIL"))
    return worst < 5e-2


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
