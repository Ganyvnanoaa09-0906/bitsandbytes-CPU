"""Device-resident llama training engine (pure iGPU): one layer + full model, fp32, validated.

Everything runs on the iGPU: RMSNorm, fused QKV projection, RoPE, attention (S/softmax/O), output
projection, residual, SwiGLU MLP, final norm and lm_head GEMM. One command queue; the CPU enqueues and
reads one loss per step (cross-entropy on the host).

Training: base weights frozen; LoRA adapters on the 5 per-layer matrices (qkv/o/gate/up/down) and on the
lm_head are trainable, with AdamW states only for those (so a 1.3B model fits the 15.4 GB shared memory).

Modes:
  --validate-layer : one layer forward vs torch fp32 reference
  --validate-bwd   : one layer backward (dX + LoRA dA/dB) vs torch autograd
  --steps N        : full 24-layer model, cross-entropy loss, AdamW on LoRA+lm_head

STATUS (2026-10-19): EXPERIMENTAL / WIP. The single-layer forward and backward modes are
validated against torch fp32 (see docstring above), but the full multi-layer --steps path has
an unresolved GPU-buffer anomaly (all layer outputs read back identical/zero, nondeterministic).
Do not use it for real training; gt1030/igpu_train.py is the supported trainer. Also note this
module creates an OpenCL context at import time, so importing it on a machine without an
OpenCL ICD fails by design.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import glob
import json
import math
import sys
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = os.environ.get('CLBLAST_PATH') or (
    r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
    if os.name == 'nt' else 'libclblast.so')
ROW, NO, YES = 101, 111, 112
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)

bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
bl.CLBlastSgemmStridedBatched.restype = ctypes.c_int
bl.CLBlastSgemmStridedBatched.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

KERNELS = r"""
__kernel void rms_stats(__global const float* x, __global float* r, int M, int D, float eps) {
    int row = get_global_id(0) / 64, lane = get_local_id(0);
    if (row >= M) return;
    __local float red[64];
    float s = 0.f;
    for (int d = lane; d < D; d += 64) { float v = x[row * D + d]; s += v * v; }
    red[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) r[row] = 1.f / sqrt(red[0] / (float)D + eps);
}
__kernel void rms_apply(__global const float* x, __global const float* w, __global const float* r,
                        __global float* y, int n, int D) {
    int i = get_global_id(0);
    if (i >= n) return;
    int row = i / D, d = i % D;
    y[i] = x[i] * r[row] * w[d];
}
__kernel void rope_fwd(__global float* X, __global const float* cosT, __global const float* sinT,
                       int BH, int T, int HD) {
    int i = get_global_id(0), hd2 = HD / 2;
    if (i >= BH * T * hd2) return;
    int hk = i % hd2, t = (i / hd2) % T, n = i / (T * hd2);
    int base = (n * T + t) * HD;
    float c = cosT[t * hd2 + hk], s = sinT[t * hd2 + hk];
    float x0 = X[base + hk], x1 = X[base + hk + hd2];
    X[base + hk] = x0 * c - x1 * s;
    X[base + hk + hd2] = x0 * s + x1 * c;
}
__kernel void rope_bwd(__global float* DX, __global const float* cosT, __global const float* sinT,
                       int BH, int T, int HD) {
    int i = get_global_id(0), hd2 = HD / 2;
    if (i >= BH * T * hd2) return;
    int hk = i % hd2, t = (i / hd2) % T, n = i / (T * hd2);
    int base = (n * T + t) * HD;
    float c = cosT[t * hd2 + hk], s = sinT[t * hd2 + hk];
    float d0 = DX[base + hk], d1 = DX[base + hk + hd2];
    DX[base + hk] = d0 * c + d1 * s;
    DX[base + hk + hd2] = -d0 * s + d1 * c;
}
__kernel void split_qkv(__global const float* qkv, __global float* Q, __global float* K,
                        __global float* V, int T, int HD, int DH, int NH) {
    int i = get_global_id(0);
    if (i >= T * HD) return;
    int n = get_global_id(1), dd = i % HD, t = (i / HD) % T, b = n / NH, h = n % NH, m = b * T + t;
    int src = m * (3 * DH) + h * HD + dd;
    int dst = (n * T + t) * HD + dd;
    Q[dst] = qkv[src];
    K[dst] = qkv[src + DH];
    V[dst] = qkv[src + 2 * DH];
}
__kernel void merge_o(__global const float* O, __global float* o_tok, int T, int HD, int DH, int NH) {
    int i = get_global_id(0);
    if (i >= T * HD) return;
    int n = get_global_id(1), dd = i % HD, t = (i / HD) % T, b = n / NH, h = n % NH, m = b * T + t;
    o_tok[m * DH + h * HD + dd] = O[(n * T + t) * HD + dd];
}
__kernel void split_dout(__global const float* dO_tok, __global float* dO, int T, int HD, int DH, int NH) {
    int i = get_global_id(0);
    if (i >= T * HD) return;
    int n = get_global_id(1), dd = i % HD, t = (i / HD) % T, b = n / NH, h = n % NH, m = b * T + t;
    dO[(n * T + t) * HD + dd] = dO_tok[m * DH + h * HD + dd];
}
__kernel void merge_dqkv(__global const float* dQ, __global const float* dK, __global const float* dV,
                         __global float* dqkv, int T, int HD, int DH, int NH) {
    int i = get_global_id(0);
    if (i >= T * HD) return;
    int n = get_global_id(1), dd = i % HD, t = (i / HD) % T, b = n / NH, h = n % NH, m = b * T + t;
    int dst = m * (3 * DH) + h * HD + dd;
    int src = (n * T + t) * HD + dd;
    dqkv[dst] = dQ[src];
    dqkv[dst + DH] = dK[src];
    dqkv[dst + 2 * DH] = dV[src];
}
__kernel void softmax_causal(__global const float* S, __global float* P, int T, float scale) {
    int row = get_global_id(0) / 256, bh = get_global_id(1), lane = get_local_id(0);
    __local float red[256];
    __global const float* s = S + (bh * T + row) * T;
    __global float* p = P + (bh * T + row) * T;
    float mx = -1e30f;
    for (int j = lane; j < T; j += 256) mx = fmax(mx, (j <= row) ? s[j] * scale : -1e30f);
    red[lane] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 128; off > 0; off >>= 1) {
        if (lane < off) red[lane] = fmax(red[lane], red[lane + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.f;
    for (int j = lane; j < T; j += 256) sum += (j <= row) ? exp(s[j] * scale - mx) : 0.f;
    red[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 128; off > 0; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = lane; j < T; j += 256) p[j] = (j <= row) ? exp(s[j] * scale - mx) / sum : 0.f;
}
__kernel void softmax_bwd_causal(__global const float* P, __global const float* dP,
                                 __global float* dS, int T, float scale) {
    int row = get_global_id(0) / 256, bh = get_global_id(1), lane = get_local_id(0);
    __local float red[256];
    __global const float* p = P + (bh * T + row) * T;
    __global const float* dp = dP + (bh * T + row) * T;
    __global float* ds = dS + (bh * T + row) * T;
    float s = 0.f;
    for (int j = lane; j < T; j += 256) s += p[j] * dp[j];
    red[lane] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 128; off > 0; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    s = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = lane; j < T; j += 256) ds[j] = p[j] * (dp[j] - s) * scale;
}
__kernel void silu_gate(__global const float* a, __global const float* b, __global float* g, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = a[i];
    g[i] = (x / (1.f + exp(-x))) * b[i];
}
__kernel void silu_bwd(__global const float* a, __global const float* b, __global const float* dg,
                       __global float* da, __global float* db, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = a[i], sig = 1.f / (1.f + exp(-x)), silu = x * sig;
    float dsilu = sig * (1.f + x * (1.f - sig));
    da[i] = dg[i] * b[i] * dsilu;
    db[i] = dg[i] * silu;
}
__kernel void resid_add(__global const float* a, __global const float* b, __global float* o,
                        int n, float beta) {
    int i = get_global_id(0);
    if (i >= n) return;
    o[i] = a[i] + beta * b[i];
}
__kernel void rms_bwd(__global const float* x, __global const float* dy, __global const float* w,
                      __global const float* r, __global float* s, __global float* dx, int M, int D) {
    int row = get_global_id(0) / 64, lane = get_local_id(0);
    if (row >= M) return;
    __local float red[64];
    float acc = 0.f;
    for (int d = lane; d < D; d += 64) acc += dy[row * D + d] * w[d] * x[row * D + d];
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 32; off > 0; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) s[row] = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float rr = r[row], ss = red[0];
    for (int d = lane; d < D; d += 64) {
        int i = row * D + d;
        dx[i] = w[d] * rr * dy[i] - (rr * rr * rr / (float)D) * x[i] * ss;
    }
}
__kernel void adamw_f32(__global float* w, __global const float* g, __global float* m,
                        __global float* v, int n, float lr, float b1, float b2, float eps,
                        float wd, float bc1, float bc2) {
    int i = get_global_id(0), base = i * 4;
    if (base >= n) return;
    float4 gg = vload4(0, g + base), ww = vload4(0, w + base);
    float4 mm = vload4(0, m + base), vv = vload4(0, v + base);
    mm = b1 * mm + (1.f - b1) * gg;
    vv = b2 * vv + (1.f - b2) * gg * gg;
    ww = ww - lr * (mm / bc1) / (sqrt(vv / bc2) + eps) - lr * wd * ww;
    vstore4(mm, 0, m + base);
    vstore4(vv, 0, v + base);
    vstore4(ww, 0, w + base);
}
__kernel void embed_gather(__global const float* emb, __global const int* ids,
                           __global float* x, int M, int D) {
    int i = get_global_id(0);
    if (i >= M * D) return;
    int m = i / D, d = i % D;
    x[i] = emb[(long)ids[m] * D + d];
}
"""
prog = cl.Program(ctx, KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
KN = ('rms_stats', 'rms_apply', 'rope_fwd', 'rope_bwd', 'split_qkv', 'merge_o', 'split_dout',
      'merge_dqkv', 'softmax_causal', 'softmax_bwd_causal', 'silu_gate', 'silu_bwd', 'resid_add',
      'rms_bwd', 'adamw_f32', 'embed_gather')
K = {n: cl.Kernel(prog, n) for n in KN}


def buf(n):
    return cl.Buffer(ctx, mf.READ_WRITE, n)


def up(a):
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))


def down(dev, shape):
    out = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(queue, out, dev)
    queue.finish()
    return out


def sgemm(a, b, c, m, n, k, ta=False, tb=False, beta=0.0):
    lda = m if ta else k
    ldb = k if tb else n
    rc = bl.CLBlastSgemm(ROW, YES if ta else NO, YES if tb else NO, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_float(beta),
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d %dx%dx%d' % (rc, m, n, k))


def sgemm_batched(a, b, c, m, n, k, batches, ta=False, tb=False, a_stride=0, b_stride=0, c_stride=0):
    lda = m if ta else k
    ldb = k if tb else n
    rc = bl.CLBlastSgemmStridedBatched(ROW, YES if ta else NO, YES if tb else NO, m, n, k,
                                       ctypes.c_float(1.0),
                                       ctypes.c_void_p(int(a.int_ptr)), 0, lda, a_stride,
                                       ctypes.c_void_p(int(b.int_ptr)), 0, ldb, b_stride,
                                       ctypes.c_float(0.0),
                                       ctypes.c_void_p(int(c.int_ptr)), 0, n, c_stride,
                                       batches, ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('SgemmBatched rc=%d' % rc)


class Lora:
    def __init__(self, r, in_dim, out_dim, seed, scale):
        g = np.random.default_rng(seed)
        A = (g.standard_normal((r, in_dim)) / math.sqrt(in_dim) * scale).astype(np.float32)
        B = np.zeros((out_dim, r), dtype=np.float32)
        self.A, self.B = up(A), up(B)
        self.Ah, self.Bh = A, B
        self.r, self.in_dim, self.out_dim = r, in_dim, out_dim
        self.gA, self.gB = buf(A.size * 4), buf(B.size * 4)
        self.mA, self.mB = buf(A.size * 4), buf(B.size * 4)
        self.vA, self.vB = buf(B.size * 4 * 0 + A.size * 4), buf(B.size * 4)
        cl.enqueue_fill_buffer(queue, self.gA, np.float32(0), 0, A.size * 4)
        cl.enqueue_fill_buffer(queue, self.gB, np.float32(0), 0, B.size * 4)
        cl.enqueue_fill_buffer(queue, self.mA, np.float32(0), 0, A.size * 4)
        cl.enqueue_fill_buffer(queue, self.mB, np.float32(0), 0, B.size * 4)
        cl.enqueue_fill_buffer(queue, self.vA, np.float32(0), 0, A.size * 4)
        cl.enqueue_fill_buffer(queue, self.vB, np.float32(0), 0, B.size * 4)


class Engine:
    def __init__(self, model_dir, tokens, r=8, scale=2.0, limit_layers=None, seed=0, keep_host=False):
        c = json.load(open(model_dir + '/config.json', encoding='utf-8'))
        tc = c.get('text_config', c)
        self.DH = tc['hidden_size']
        self.FF = tc['intermediate_size']
        self.NL = limit_layers or tc['num_hidden_layers']
        self.NH = tc['num_attention_heads']
        self.NKV = tc.get('num_key_value_heads', self.NH)
        self.HD = tc.get('head_dim') or (self.DH // self.NH)
        self.EPS = tc.get('rms_norm_eps', 1e-6)
        self.THETA = tc.get('rope_theta', 10000.0)
        self.VOCAB = tc['vocab_size']
        self.B, self.T = 1, tokens
        self.M = self.B * tokens
        self.BH = self.B * self.NH
        self.r = r
        self.scale = scale
        self.attn_scale = 1.0 / math.sqrt(self.HD)
        state = {}
        for f in sorted(glob.glob(model_dir + '/*.safetensors')):
            from safetensors.torch import load_file
            state.update(load_file(f))
        self.host = {}
        self.layers = []
        for li in range(self.NL):
            p = 'model.layers.%d.' % li
            tw = {
                'q': state[p + 'self_attn.q_proj.weight'], 'k': state[p + 'self_attn.k_proj.weight'],
                'v': state[p + 'self_attn.v_proj.weight'], 'o': state[p + 'self_attn.o_proj.weight'],
                'g': state[p + 'mlp.gate_proj.weight'], 'u': state[p + 'mlp.up_proj.weight'],
                'd': state[p + 'mlp.down_proj.weight'],
                'n1': state[p + 'input_layernorm.weight'],
                'n2': state[p + 'post_attention_layernorm.weight']}
            if keep_host:
                self.host[li] = {k: v.float().numpy().copy() for k, v in tw.items()}
            lay = dict(
                wqkv=up(np.concatenate([tw['q'].float().numpy().T, tw['k'].float().numpy().T,
                                        tw['v'].float().numpy().T], axis=1)),
                wo=up(tw['o'].float().numpy().T), wg=up(tw['g'].float().numpy().T),
                wu=up(tw['u'].float().numpy().T), wd=up(tw['d'].float().numpy().T),
                n1=up(tw['n1'].float().numpy()), n2=up(tw['n2'].float().numpy()))
            self.layers.append(lay)
        self.wfinal = up(state['model.norm.weight'].float().numpy())
        self.whost_final = state['model.norm.weight'].float().numpy()
        self.wlmT = up(state['lm_head.weight'].float().numpy().copy())
        self.embed = state['model.embed_tokens.weight'].float().numpy()
        del state
        self._alloc()
        self._lora(seed)

    def _alloc(self):
        M, T, DH, HD, FF, V = self.M, self.T, self.DH, self.HD, self.FF, self.VOCAB
        BH = self.BH
        self.xb = [buf(M * DH * 4) for _ in range(self.NL + 1)]
        self.acts = []
        for _ in range(self.NL):
            self.acts.append(dict(
                h=buf(M * DH * 4), r=buf(M * 4), qkv=buf(M * 3 * DH * 4),
                Q=buf(BH * T * HD * 4), K=buf(BH * T * HD * 4), V=buf(BH * T * HD * 4),
                S=buf(BH * T * T * 4), P=buf(BH * T * T * 4), o_tok=buf(M * DH * 4),
                x1=buf(M * DH * 4), h2=buf(M * DH * 4), r2=buf(M * 4),
                a=buf(M * FF * 4), b=buf(M * FF * 4), g=buf(M * FF * 4)))
        self.tmp1 = buf(M * DH * 4)
        self.tmp2 = buf(M * DH * 4)
        self.tmpbh = buf(BH * T * HD * 4)
        self.tmpr = buf(M * self.r * 4)
        self.sc = dict(
            dattn=buf(M * DH * 4), dO_tok=buf(M * DH * 4), dO=buf(BH * T * HD * 4),
            dP=buf(BH * T * T * 4), dS=buf(BH * T * T * 4), dQ=buf(BH * T * HD * 4),
            dK=buf(BH * T * HD * 4), dV=buf(BH * T * HD * 4), dqkv=buf(M * 3 * DH * 4),
            dh=buf(M * DH * 4), dg=buf(M * FF * 4), da=buf(M * FF * 4), db=buf(M * FF * 4),
            dh2=buf(M * DH * 4), dmlp=buf(M * DH * 4), tmp=buf(M * DH * 4),
            rms_s=buf(M * 4), dx=buf(M * DH * 4))
        self.hfn = buf(M * DH * 4)
        self.rfin = buf(M * 4)
        self.logits = buf(M * V * 4)
        self.dlogits = buf(M * V * 4)
        self.dhfn = buf(M * DH * 4)
        self.dxemb = buf(M * DH * 4)
        self.dxb = [buf(M * DH * 4) for _ in range(self.NL + 1)]
        self.tok_dev = buf(M * 4)
        hd2 = self.HD // 2
        inv = 1.0 / (self.THETA ** (np.arange(0, self.HD, 2, dtype=np.float64) / self.HD))
        ang = np.arange(self.T, dtype=np.float64)[:, None] * inv[None, :]
        self.cos = up(np.cos(ang).astype(np.float32))
        self.sin = up(np.sin(ang).astype(np.float32))

    def _lora(self, seed):
        DH, FF, V = self.DH, self.FF, self.VOCAB
        specs = [('qkv', DH, 3 * DH), ('o', DH, DH), ('gate', DH, FF), ('up', DH, FF), ('down', FF, DH)]
        self.lora = []
        for li in range(self.NL):
            self.lora.append({name: Lora(self.r, i, o, seed + li * 10 + k, self.scale)
                              for k, (name, i, o) in enumerate(specs)})
        self.lora_lm = Lora(self.r, DH, V, seed + 999, self.scale)

    # ---- kernels helpers ----
    def rms(self, x, w, rbuf, y):
        M, D = self.M, self.DH
        K['rms_stats'](queue, (M * 64,), (64,), x, rbuf, np.int32(M), np.int32(D), np.float32(self.EPS))
        K['rms_apply'](queue, (M * D,), (256,), x, w, rbuf, y, np.int32(M * D), np.int32(D))

    def lora_fwd(self, x, lo, y, m, in_dim, out_dim):
        sgemm(x, lo.A, self.tmpr, m, self.r, in_dim, tb=True)
        sgemm(self.tmpr, lo.B, y, m, out_dim, self.r, tb=True, beta=1.0)

    def lora_bwd(self, x, lo, dy, m, in_dim, out_dim):
        """Accumulate dA/dB and return dx buffer (self.tmp2) contribution via self.sc.dtmp."""
        sgemm(x, lo.A, self.tmpr, m, self.r, in_dim, tb=True)       # t
        sgemm(dy, self.tmpr, lo.gB, out_dim, self.r, m, ta=True, beta=1.0)   # dB = dy^T t
        sgemm(dy, lo.B, self.sc['tmp2r'], m, self.r, out_dim)       # dt = dy B
        sgemm(self.sc['tmp2r'], x, lo.gA, self.r, in_dim, m, ta=True, beta=1.0)  # dA = dt^T x
        sgemm(self.sc['tmp2r'], lo.A, self.sc['dlora'], m, in_dim, self.r)       # dx_lora = dt A


    def lora_bwd(self, x, lo, dy, m, in_dim, out_dim, dx_dst):
        """Accumulate dA/dB and add the adapter's dX contribution into dx_dst."""
        sgemm(x, lo.A, self.tmpr, m, self.r, in_dim, tb=True)                     # t = x A^T
        sgemm(dy, self.tmpr, lo.gB, out_dim, self.r, m, ta=True, beta=1.0)        # dB += dy^T t
        sgemm(dy, lo.B, self.sc['tmp2r'], m, self.r, out_dim)                     # dt = dy B
        sgemm(self.sc['tmp2r'], x, lo.gA, self.r, in_dim, m, ta=True, beta=1.0)   # dA += dt^T x
        sgemm(self.sc['tmp2r'], lo.A, self.sc['dlora'], m, in_dim, self.r)        # dx = dt A
        K['resid_add'](queue, (m * in_dim,), (256,), dx_dst, self.sc['dlora'], dx_dst,
                       np.int32(m * in_dim), np.float32(1.0))

    # ---- forward ----
    def forward_layer(self, li, x_in, x_out):
        DH, FF, T, HD, BH, M = self.DH, self.FF, self.T, self.HD, self.BH, self.M
        lay, a = self.layers[li], self.acts[li]
        self.rms(x_in, lay['n1'], a['r'], a['h'])
        sgemm(a['h'], lay['wqkv'], a['qkv'], M, 3 * DH, DH)
        self.lora_fwd(a['h'], self.lora[li]['qkv'], a['qkv'], M, DH, 3 * DH)
        K['split_qkv'](queue, (T * HD, BH), (128, 1), a['qkv'], a['Q'], a['K'], a['V'],
                       np.int32(T), np.int32(HD), np.int32(DH), np.int32(self.NH))
        K['rope_fwd'](queue, (BH * T * HD // 2,), (256,), a['Q'], self.cos, self.sin,
                      np.int32(BH), np.int32(T), np.int32(HD))
        K['rope_fwd'](queue, (BH * T * HD // 2,), (256,), a['K'], self.cos, self.sin,
                      np.int32(BH), np.int32(T), np.int32(HD))
        sgemm_batched(a['Q'], a['K'], a['S'], T, T, HD, BH, tb=True,
                      a_stride=T * HD, b_stride=T * HD, c_stride=T * T)
        K['softmax_causal'](queue, (T * 256, BH), (256, 1), a['S'], a['P'], np.int32(T), np.float32(self.attn_scale))
        sgemm_batched(a['P'], a['V'], self.tmpbh, T, HD, T, BH,
                      a_stride=T * T, b_stride=T * HD, c_stride=T * HD)
        K['merge_o'](queue, (T * HD, BH), (128, 1), self.tmpbh, a['o_tok'],
                     np.int32(T), np.int32(HD), np.int32(DH), np.int32(self.NH))
        sgemm(a['o_tok'], lay['wo'], self.tmp1, M, DH, DH)
        self.lora_fwd(a['o_tok'], self.lora[li]['o'], self.tmp1, M, DH, DH)
        K['resid_add'](queue, (M * DH,), (256,), x_in, self.tmp1, a['x1'], np.int32(M * DH), np.float32(1.0))
        self.rms(a['x1'], lay['n2'], a['r2'], a['h2'])
        sgemm(a['h2'], lay['wg'], a['a'], M, FF, DH)
        sgemm(a['h2'], lay['wu'], a['b'], M, FF, DH)
        self.lora_fwd(a['h2'], self.lora[li]['gate'], a['a'], M, DH, FF)
        self.lora_fwd(a['h2'], self.lora[li]['up'], a['b'], M, DH, FF)
        K['silu_gate'](queue, (M * FF,), (256,), a['a'], a['b'], a['g'], np.int32(M * FF))
        sgemm(a['g'], lay['wd'], self.tmp1, M, DH, FF)
        self.lora_fwd(a['g'], self.lora[li]['down'], self.tmp1, M, FF, DH)
        K['resid_add'](queue, (M * DH,), (256,), a['x1'], self.tmp1, x_out, np.int32(M * DH), np.float32(1.0))

    def forward(self, ids):
        M, V = self.M, self.VOCAB
        x0 = self.embed[np.asarray(ids).reshape(-1)[:M]].astype(np.float32)
        cl.enqueue_copy(queue, self.xb[0], np.ascontiguousarray(x0), is_blocking=True)
        for li in range(self.NL):
            self.forward_layer(li, self.xb[li], self.xb[li + 1])
        self.rms(self.xb[self.NL], self.wfinal, self.rfin, self.hfn)
        sgemm(self.hfn, self.wlmT, self.logits, M, V, self.DH, tb=True)   # logits = hfn W^T
        self.lora_fwd(self.hfn, self.lora_lm, self.logits, M, self.DH, V)
        return self.logits

    # ---- backward ----
    def backward_layer(self, li, dx_out, dx_in):
        DH, FF, T, HD, BH, M = self.DH, self.FF, self.T, self.HD, self.BH, self.M
        lay, a, s = self.layers[li], self.acts[li], self.sc
        lo = self.lora[li]
        # ---------------- SwiGLU MLP ----------------
        sgemm(dx_out, lay['wd'], s['dg'], M, FF, DH, tb=True)                     # dg = dy wd^T
        self.lora_bwd(a['g'], lo['down'], dx_out, M, FF, DH, s['dg'])             # + down adapter dx
        K['silu_bwd'](queue, (M * FF,), (256,), a['a'], a['b'], s['dg'], s['da'], s['db'], np.int32(M * FF))
        sgemm(s['da'], lay['wg'], s['dh2'], M, DH, FF, tb=True)                   # dh2 = da wg^T
        sgemm(s['db'], lay['wu'], s['dh2'], M, DH, FF, tb=True, beta=1.0)         #    += db wu^T
        self.lora_bwd(a['h2'], lo['gate'], s['da'], M, DH, FF, s['dh2'])          # + gate adapter dx
        self.lora_bwd(a['h2'], lo['up'], s['db'], M, DH, FF, s['dh2'])            # + up adapter dx
        K['rms_bwd'](queue, (M * 64,), (64,), a['x1'], s['dh2'], lay['n2'], a['r2'], s['rms_s'], s['dmlp'],
                     np.int32(M), np.int32(DH))
        K['resid_add'](queue, (M * DH,), (256,), dx_out, s['dmlp'], s['dattn'], np.int32(M * DH), np.float32(1.0))
        # ---------------- attention ----------------
        sgemm(s['dattn'], lay['wo'], s['dO_tok'], M, DH, DH, tb=True)             # dO_tok = dattn wo^T
        self.lora_bwd(a['o_tok'], lo['o'], s['dattn'], M, DH, DH, s['dO_tok'])    # + o adapter dx
        K['split_dout'](queue, (T * HD, BH), (128, 1), s['dO_tok'], s['dO'],
                        np.int32(T), np.int32(HD), np.int32(DH), np.int32(self.NH))
        sgemm_batched(s['dO'], a['V'], s['dP'], T, T, HD, BH, tb=True,
                      a_stride=T * HD, b_stride=T * HD, c_stride=T * T)
        K['softmax_bwd_causal'](queue, (T * 256, BH), (256, 1), a['P'], s['dP'], s['dS'],
                                np.int32(T), np.float32(self.attn_scale))
        sgemm_batched(a['P'], s['dO'], s['dV'], T, HD, T, BH, ta=True,
                      a_stride=T * T, b_stride=T * HD, c_stride=T * HD)
        sgemm_batched(s['dS'], a['K'], s['dQ'], T, HD, T, BH,
                      a_stride=T * T, b_stride=T * HD, c_stride=T * HD)
        sgemm_batched(s['dS'], a['Q'], s['dK'], T, HD, T, BH, ta=True,
                      a_stride=T * T, b_stride=T * HD, c_stride=T * HD)
        K['rope_bwd'](queue, (BH * T * HD // 2,), (256,), s['dQ'], self.cos, self.sin,
                      np.int32(BH), np.int32(T), np.int32(HD))
        K['rope_bwd'](queue, (BH * T * HD // 2,), (256,), s['dK'], self.cos, self.sin,
                      np.int32(BH), np.int32(T), np.int32(HD))
        K['merge_dqkv'](queue, (T * HD, BH), (128, 1), s['dQ'], s['dK'], s['dV'], s['dqkv'],
                        np.int32(T), np.int32(HD), np.int32(DH), np.int32(self.NH))
        sgemm(s['dqkv'], lay['wqkv'], s['dh'], M, DH, 3 * DH, tb=True)            # dh = dqkv wqkv^T
        self.lora_bwd(a['h'], lo['qkv'], s['dqkv'], M, DH, 3 * DH, s['dh'])       # + qkv adapter dx
        K['rms_bwd'](queue, (M * 64,), (64,), self.xb[li], s['dh'], lay['n1'], a['r'], s['rms_s'],
                     s['tmp'], np.int32(M), np.int32(DH))
        K['resid_add'](queue, (M * DH,), (256,), s['dattn'], s['tmp'], dx_in, np.int32(M * DH), np.float32(1.0))

    def backward(self):
        DH, V, M = self.DH, self.VOCAB, self.M
        # final norm + lm head
        sgemm(self.dlogits, self.wlmT, self.dhfn, M, DH, V)              # dhfn = dlogits @ W
        self.lora_bwd(self.hfn, self.lora_lm, self.dlogits, M, DH, V, self.dhfn)  # + lm adapter dx
        K['rms_bwd'](queue, (M * 64,), (64,), self.xb[self.NL], self.dhfn, self.wfinal, self.rfin,
                     self.sc['rms_s'], self.dxb[self.NL], np.int32(M), np.int32(DH))
        cur = self.dxb[self.NL]
        for li in range(self.NL - 1, -1, -1):
            self.backward_layer(li, cur, self.dxb[li])
            cur = self.dxb[li]
        return cur

    def step(self, ids, lr, targets):
        t0 = time.perf_counter()
        self.forward(ids)
        logits = down(self.logits, (self.M, self.VOCAB))
        logits = logits.astype(np.float64)
        mx = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - mx)
        p = e / e.sum(axis=1, keepdims=True)
        tgt = targets.reshape(-1)
        loss = float(-np.log(np.clip(p[np.arange(len(tgt)), tgt], 1e-12, None)).mean())
        dlog = p.copy()
        dlog[np.arange(len(tgt)), tgt] -= 1.0
        dlog /= len(tgt)
        cl.enqueue_copy(queue, self.dlogits, dlog.astype(np.float32))
        self.backward()
        # AdamW on LoRA params
        step = self._step
        self._step += 1
        bc1 = 1.0 - 0.9 ** step
        bc2 = 1.0 - 0.999 ** step
        for li in range(self.NL):
            for name in self.lora[li]:
                lo = self.lora[li][name]
                self._adamw(lo.A, lo.gA, lo.mA, lo.vA, lo.A.size, lr, bc1, bc2)
                self._adamw(lo.B, lo.gB, lo.mB, lo.vB, lo.B.size, lr, bc1, bc2)
        lo = self.lora_lm
        self._adamw(lo.A, lo.gA, lo.mA, lo.vA, lo.A.size, lr, bc1, bc2)
        self._adamw(lo.B, lo.gB, lo.mB, lo.vB, lo.B.size, lr, bc1, bc2)
        # free grads
        for li in range(self.NL):
            for name in self.lora[li]:
                lo = self.lora[li][name]
                cl.enqueue_fill_buffer(queue, lo.gA, np.float32(0), 0, lo.gA.size)
                cl.enqueue_fill_buffer(queue, lo.gB, np.float32(0), 0, lo.gB.size)
        cl.enqueue_fill_buffer(queue, lo.gA, np.float32(0), 0, lo.gA.size)
        cl.enqueue_fill_buffer(queue, lo.gB, np.float32(0), 0, lo.gB.size)
        queue.finish()
        return loss, time.perf_counter() - t0

    def _adamw(self, w, g, m, v, nbytes, lr, bc1, bc2):
        n = nbytes // 4
        K['adamw_f32'](queue, (n // 4,), (256,), w, g, m, v, np.int32(n), np.float32(lr),
                       np.float32(0.9), np.float32(0.999), np.float32(1e-8), np.float32(0.0),
                       np.float32(bc1), np.float32(bc2))

    _step = 1
    # scratch needed by backward_layer LoRA path
    @property
    def sc_dlora(self):
        return self.sc['dlora']


# add dynamic scratch buffers used above
_orig_alloc = Engine._alloc


def _alloc2(self):
    _orig_alloc(self)
    M, DH = self.M, self.DH
    self.sc['tmp2r'] = buf(M * self.r * 4)
    self.sc['dlora'] = buf(M * max(DH, self.FF) * 4)
    self.dxemb2 = buf(M * DH * 4)


Engine._alloc = _alloc2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--validate-layer', action='store_true')
    ap.add_argument('--validate-bwd', action='store_true')
    args = ap.parse_args()
    if args.validate_layer or args.validate_bwd:
        eng = Engine(args.model, args.tokens, limit_layers=1, keep_host=True)
        print('one-layer engine: DH=%d FF=%d HD=%d T=%d' % (eng.DH, eng.FF, eng.HD, eng.T))
        # random input
        g = np.random.default_rng(0)
        x = (g.standard_normal((eng.M, eng.DH)) * 0.1).astype(np.float32)
        cl.enqueue_copy(queue, eng.xb[0], x)
        eng.forward_layer(0, eng.xb[0], eng.xb[1])
        queue.finish()
        out = down(eng.xb[1], (eng.M, eng.DH))
        # torch reference
        hw = eng.host[0]
        xt = torch.from_numpy(x)
        h = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + eng.EPS) * torch.from_numpy(hw['n1'])
        qkv = h @ torch.from_numpy(np.concatenate([hw['q'].T, hw['k'].T, hw['v'].T], axis=1))
        lo = eng.lora[0]['qkv']
        qkv = qkv + (h @ torch.from_numpy(lo.Ah).T) @ torch.from_numpy(lo.Bh).T
        T, HD, NH = eng.T, eng.HD, eng.NH
        D = eng.DH
        q = qkv[:, :D].reshape(T, NH, HD).transpose(0, 1)
        k = qkv[:, D:2 * D].reshape(T, NH, HD).transpose(0, 1)
        v = qkv[:, 2 * D:].reshape(T, NH, HD).transpose(0, 1)
        hd2 = HD // 2
        inv = 1.0 / (eng.THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        ang = torch.from_numpy(np.arange(T, dtype=np.float64)[:, None] * inv[None, :])
        c, s = torch.cos(ang).float(), torch.sin(ang).float()
        def rope(tens):
            x0, x1 = tens[..., :hd2], tens[..., hd2:]
            return torch.cat([x0 * c[None] - x1 * s[None], x0 * s[None] + x1 * c[None]], dim=-1)
        qr, kr = rope(q), rope(k)
        sm = (qr @ kr.transpose(-1, -2)) * (1.0 / math.sqrt(HD))
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        sm = sm.masked_fill(~mask, float('-inf'))
        pr = torch.softmax(sm, -1)
        o = pr @ v
        o_tok = o.transpose(0, 1).reshape(T, D)
        og = o_tok @ torch.from_numpy(hw['o'].T)
        lo = eng.lora[0]['o']
        og = og + (o_tok @ torch.from_numpy(lo.Ah).T) @ torch.from_numpy(lo.Bh).T
        x1r = xt + og
        h2 = x1r * torch.rsqrt(x1r.pow(2).mean(-1, keepdim=True) + eng.EPS) * torch.from_numpy(hw['n2'])
        gr = h2 @ torch.from_numpy(hw['g'].T)
        ur = h2 @ torch.from_numpy(hw['u'].T)
        gz = torch.nn.functional.silu(gr) * ur
        yr = gz @ torch.from_numpy(hw['d'].T)
        out_ref = x1r + yr
        err = float(np.abs(out - out_ref.numpy()).max())
        rel = err / (float(np.abs(out_ref.numpy()).max()) + 1e-9)
        print('one-layer forward: max abs err %.3e (relative %.3e)' % (err, rel))
        return 0
    eng = Engine(args.model, args.tokens)
    print('engine: DH=%d FF=%d NL=%d NH=%d HD=%d T=%d M=%d vocab=%d'
          % (eng.DH, eng.FF, eng.NL, eng.NH, eng.HD, eng.T, eng.M, eng.VOCAB))
    if args.steps > 0:
        g = np.random.default_rng(1234)
        ids = g.integers(0, 2000, size=(eng.M,)).astype(np.int64)
        targets = np.roll(ids, -1)
        targets[-1] = ids[0]
        times, losses = [], []
        t0 = time.perf_counter()
        for i in range(args.steps):
            loss, dt = eng.step(ids, args.lr, targets)
            times.append(dt); losses.append(loss)
            print('  step %3d  loss %.4f  step %.3f s  (prefill %.2f s)'
                  % (i, loss, dt, time.perf_counter() - t0 - sum(times)), flush=True)
        print()
        print('pure-iGPU full model: %d steps, %.3f s/step median, loss %.4f -> %.4f'
              % (args.steps, float(np.median(times)), losses[0], losses[-1]))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
