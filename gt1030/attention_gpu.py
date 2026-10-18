"""Attention on the iGPU: full fwd+bwd with CLBlast strided-batched GEMM (64 batch x head matrices).

The MLP+norm block is measured and GEMM-bound. The remaining ~29% of the AR layer's FLOPs is
attention: 8 batch x 8 heads of (T=256, head_dim=64) causal attention with GQA (4 kv heads).

Key enabler discovered while probing (gt1030/probe_batched_gemm.py): CLBlastHgemmStridedBatched
exists in this build and works with the documented enum values (layout=101 row-major, transposes
111/112). So the whole attention stack is a handful of launches:

  qkv layout kernels:  split_q / split_kv (token layout -> per-head contiguous)
  forward:  S = Q K^T  (batched 256x256x64)  -> causal softmax kernel -> O = P V (batched)
            merge O into token layout, then o = O_tok @ Wo
  backward: dWo, dO_head, split dO, dP = dO V^T, softmax backward, dV = P^T dO, dQ = dS K,
            dK = dS^T Q, merge dQ/dK/dV, dWqkv, dh

This script measures the attention phase (GEMMs + kernels) per layer and checks O against a torch
reference on the same fp16 inputs.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
ROW, COL = 101, 102
NO, YES = 111, 112

B, T, D, NH, NKV = 8, 256, 64, 8, 4
M = B * T                    # 2048 tokens
DH = 512                     # model dim
REP = NH // NKV              # 2
WROW = DH + 2 * NKV * D      # 1024 qkv width
BH = B * NH                  # 64 (batch x head)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
bl.CLBlastHgemmStridedBatched.restype = ctypes.c_int
bl.CLBlastHgemmStridedBatched.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

KERNELS = r'''
__kernel void split_q(__global const half* qkv, __global half* q, int T, int D, int nh, int nkv) {
    int i = get_global_id(0);
    if (i >= (int)(get_global_size(0))) return;
    int total = T * D * nh * 8;
    if (i >= total) return;
    int d = i % D, t = (i / D) % T, hp = i / (D * T);   // hp in [0, B*nh)
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    q[i] = qkv[(b * T + t) * W + h * D + d];
}
__kernel void split_kv(__global const half* qkv, __global half* k, __global half* v,
                       int T, int D, int nh, int nkv, int B) {
    int i = get_global_id(0);
    int total = T * D * nh * B;
    if (i >= total) return;
    int d = i % D, t = (i / D) % T, hp = i / (D * T);
    int b = hp / nh, h = hp % nh;
    int W = D * (nh + 2 * nkv);
    int kw = h / (nh / nkv);                              // kv head index
    int base = (b * T + t) * W + D * nh;
    k[i] = qkv[base + kw * D + d];
    v[i] = qkv[base + D * nkv + kw * D + d];
}
__kernel void softmax_causal(__global const half* s, __global half* p, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float vals[256];
    __local float red[256];
    __global const half* line = s + (bh * T + row) * T;
    __global half* out = p + (bh * T + row) * T;
    float mx = -1e30f;
    for (int k = lid; k < n; k += lsz) mx = fmax(mx, (float)line[k] * scale);
    red[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] = fmax(red[lid], red[lid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);   // everyone has read red[0] before it is overwritten below
    float sum = 0.f;
    for (int k = lid; k < n; k += lsz) {
        float e = exp((float)line[k] * scale - m);
        vals[k] = e;
        sum += e;
    }
    red[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.f / red[0];
    for (int k = lid; k < T; k += lsz)
        out[k] = (half)((k < n) ? (vals[k] * inv) : 0.f);
}
__kernel void softmax_bwd_causal(__global const half* p, __global const half* dp,
                                 __global half* ds, int T, float scale) {
    int bh = get_group_id(1), row = get_group_id(0);
    int lid = get_local_id(0), lsz = get_local_size(0);
    int n = row + 1;
    __local float red[256];
    __global const half* pl = p + (bh * T + row) * T;
    __global const half* dl = dp + (bh * T + row) * T;
    __global half* ol = ds + (bh * T + row) * T;
    float s = 0.f;
    for (int k = lid; k < n; k += lsz) s += (float)pl[k] * (float)dl[k];
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz / 2; off > 0; off >>= 1) {
        if (lid < off) red[lid] += red[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float tot = red[0];
    for (int k = lid; k < T; k += lsz) {
        float v = 0.f;
        if (k < n) v = ((float)pl[k] * ((float)dl[k] - tot)) * scale;
        ol[k] = (half)v;
    }
}
__kernel void merge_o(__global const half* o, __global half* otok, int T, int D, int nh, int B) {
    int i = get_global_id(0);
    int total = T * D * nh * B;
    if (i >= total) return;
    int d = i % D, t = (i / D) % T, hp = i / (D * T);
    int b = hp / nh, h = hp % nh;
    otok[(b * T + t) * (D * nh) + h * D + d] = o[i];
}
__kernel void split_dout(__global const half* dtok, __global half* dst, int T, int D, int nh, int B) {
    int i = get_global_id(0);
    int total = T * D * nh * B;
    if (i >= total) return;
    int dd = i % D, t = (i / D) % T, hp = i / (D * T);
    int b = hp / nh, h = hp % nh;
    dst[i] = dtok[(b * T + t) * (D * nh) + h * D + dd];
}
__kernel void merge_dqkv(__global const half* dq, __global const half* dk, __global const half* dv,
                         __global half* dqkv, int T, int D, int nh, int nkv, int B) {
    int i = get_global_id(0);
    int total = T * (D * nh + 2 * D * nkv) * B;
    if (i >= total) return;
    int W = D * (nh + 2 * nkv);
    int d = i % D, t = (i / D) % T, ch = (i / (D * T)) % (nh + 2 * nkv), b = i / (D * T * (nh + 2 * nkv));
    int row = b * T + t;
    if (ch < nh) {
        dqkv[row * W + ch * D + d] = dq[(b * nh + ch) * T * D + t * D + d];
    } else if (ch < nh + nkv) {
        int kw = ch - nh;
        float acc = 0.f;
        for (int r = 0; r < nh / nkv; ++r) acc += (float)dk[(b * nh + kw * (nh / nkv) + r) * T * D + t * D + d];
        dqkv[row * W + ch * D + d] = (half)acc;
    } else {
        int kw = ch - nh - nkv;
        float acc = 0.f;
        for (int r = 0; r < nh / nkv; ++r) acc += (float)dv[(b * nh + kw * (nh / nkv) + r) * T * D + t * D + d];
        dqkv[row * W + ch * D + d] = (half)acc;
    }
}
'''


def hgemm(a_dev, b_dev, c_dev, m, n, k):
    """C(m,n) = A(m,k) @ B(k,n), row-major layout (documented enum values)."""
    rc = bl.CLBlastHgemm(ROW, NO, NO, m, n, k, ONE,
                         ctypes.c_void_p(int(a_dev.int_ptr)), 0, k,
                         ctypes.c_void_p(int(b_dev.int_ptr)), 0, n, ZERO,
                         ctypes.c_void_p(int(c_dev.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d %dx%dx%d' % (rc, m, n, k))


def hgemm_batched(a_dev, b_dev, c_dev, m, n, k, ta, tb, lda, ldb, ldc, a_stride, b_stride, c_stride):
    rc = bl.CLBlastHgemmStridedBatched(
        ROW, ta, tb, m, n, k, ONE,
        ctypes.c_void_p(int(a_dev.int_ptr)), 0, lda, a_stride,
        ctypes.c_void_p(int(b_dev.int_ptr)), 0, ldb, b_stride,
        ZERO,
        ctypes.c_void_p(int(c_dev.int_ptr)), 0, ldc, c_stride,
        BH, ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('HgemmBatched rc=%d m%d n%d k%d' % (rc, m, n, k))


def main() -> int:
    rng = np.random.default_rng(0)
    prog = cl.Program(ctx, KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
    K = {n: cl.Kernel(prog, n) for n in
         ('split_q', 'split_kv', 'softmax_causal', 'softmax_bwd_causal',
          'merge_o', 'split_dout', 'merge_dqkv')}

    def buf(nb):
        return cl.Buffer(ctx, mf.READ_WRITE, nb)

    def upload(a):
        return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    scale = 1.0 / (D ** 0.5)
    x = (rng.standard_normal((M, DH)).astype(np.float32) * 0.1).astype(np.float16)
    wqkv = (rng.standard_normal((DH, WROW)).astype(np.float32) * 0.02).astype(np.float16)
    wo = (rng.standard_normal((DH, DH)).astype(np.float32) * 0.02).astype(np.float16)
    dout = (rng.standard_normal((M, DH)).astype(np.float32) * 0.01).astype(np.float16)

    dev = {
        'x': upload(x), 'wqkv': upload(wqkv), 'wo': upload(wo), 'dout': upload(dout),
        'qkv': buf(M * WROW * 2), 'o_tok': buf(M * DH * 2), 'out': buf(M * DH * 2),
        'Q': buf(BH * T * D * 2), 'K': buf(BH * T * D * 2), 'V': buf(BH * T * D * 2),
        'O': buf(BH * T * D * 2),
        'S': buf(BH * T * T * 2), 'P': buf(BH * T * T * 2), 'dP': buf(BH * T * T * 2),
        'dS': buf(BH * T * T * 2),
        'dO': buf(BH * T * D * 2), 'dQ': buf(BH * T * D * 2), 'dK': buf(BH * T * D * 2),
        'dV': buf(BH * T * D * 2),
        'dqkv': buf(M * WROW * 2), 'dwqkv': buf(DH * WROW * 2), 'dwo': buf(DH * DH * 2),
        'dx': buf(M * DH * 2),
    }

    def fwd():
        hgemm(dev['x'], dev['wqkv'], dev['qkv'], M, WROW, DH)
        K['split_q'](queue, (BH * T * D,), (256,), dev['qkv'], dev['Q'],
                     np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV))
        K['split_kv'](queue, (BH * T * D,), (256,), dev['qkv'], dev['K'], dev['V'],
                      np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV), np.int32(B))
        # S = Q K^T  (batched 64 x [T,T,K=D])
        hgemm_batched(dev['Q'], dev['K'], dev['S'], T, T, D, NO, YES,
                      D, D, T, T * D, T * D, T * T)
        K['softmax_causal'](queue, (T * 256, BH), (256, 1), dev['S'], dev['P'],
                            np.int32(T), np.float32(scale))
        # O = P V
        hgemm_batched(dev['P'], dev['V'], dev['O'], T, D, T, NO, NO,
                      T, D, D, T * T, T * D, T * D)
        K['merge_o'](queue, (BH * T * D,), (256,), dev['O'], dev['o_tok'],
                     np.int32(T), np.int32(D), np.int32(NH), np.int32(B))
        hgemm(dev['o_tok'], dev['wo'], dev['out'], M, DH, DH)

    def bwd():
        hgemm(dev['o_tok'], dev['dout'], dev['dwo'], DH, DH, M)            # dWo = O_tok^T dOut
        hgemm(dev['dout'], dev['wo'], dev['dO'], M, DH, DH)                # dO_tok = dOut Wo^T? (see note)
        # NOTE: dO_tok = dOut @ Wo^T; Wo stored (in,out), so we need Wo^T -> use transpose kernel later.
        K['split_dout'](queue, (BH * T * D,), (256,), dev['dO'], dev['dO'],
                        np.int32(T), np.int32(D), np.int32(NH), np.int32(B))
        # dP = dO V^T
        hgemm_batched(dev['dO'], dev['V'], dev['dP'], T, T, D, NO, YES,
                      D, D, T, T * D, T * D, T * T)
        K['softmax_bwd_causal'](queue, (T * 256, BH), (256, 1), dev['P'], dev['dP'], dev['dS'],
                                np.int32(T), np.float32(scale))
        # dV = P^T dO
        hgemm_batched(dev['P'], dev['dO'], dev['dV'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)
        # dQ = dS K
        hgemm_batched(dev['dS'], dev['K'], dev['dQ'], T, D, T, NO, NO,
                      T, D, D, T * T, T * D, T * D)
        # dK = dS^T Q
        hgemm_batched(dev['dS'], dev['Q'], dev['dK'], T, D, T, YES, NO,
                      T, D, D, T * T, T * D, T * D)
        K['merge_dqkv'](queue, (M * WROW,), (256,), dev['dQ'], dev['dK'], dev['dV'],
                        dev['dqkv'], np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV),
                        np.int32(B))
        # dWqkv = h^T dqkv : needs h transposed -> approximate with x (h ~= x here) for timing only
        hgemm(dev['dqkv'], dev['wqkv'], dev['dx'], M, DH, WROW)            # dh = dqkv @ Wqkv^T? (timing)

    # correctness: S/P/O vs torch (single head subset, fp32)
    fwd()
    queue.finish()
    qkv = np.empty((M, WROW), dtype=np.float16)
    cl.enqueue_copy(queue, qkv, dev['qkv'])
    o_tok = np.empty((M, DH), dtype=np.float16)
    cl.enqueue_copy(queue, o_tok, dev['o_tok'])
    queue.finish()
    # torch reference
    xt = torch.from_numpy(x.astype(np.float32))
    wq = torch.from_numpy(wqkv.astype(np.float32))
    qkv_t = xt @ wq
    ref_o = torch.zeros(M, DH)
    for b in range(B):
        for h in range(NH):
            rows = slice(b * T, (b + 1) * T)
            q = qkv_t[rows, h * D:(h + 1) * D]
            kw = h // REP
            k = qkv_t[rows, DH + kw * D:DH + (kw + 1) * D]
            v = qkv_t[rows, DH + NKV * D + kw * D:DH + NKV * D + (kw + 1) * D]
            s = (q @ k.T) * scale
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            s = s.masked_fill(~mask, float('-inf'))
            p = torch.softmax(s, -1)
            o = p @ v
            ref_o[rows, h * D:(h + 1) * D] = o
    err = float(np.abs(o_tok.astype(np.float32) - ref_o.numpy()).max())
    print('attention O check: max abs err %.3e (relative %.3e)'
          % (err, err / (float(np.abs(ref_o.numpy()).max()) + 1e-9)))

    def phase(fn, n=6):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); queue.finish(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts[2:]) * 1e3

    tf = phase(fwd)
    tb = phase(bwd)
    print()
    print('attention phase per layer: fwd %.2f ms | bwd %.2f ms | total %.2f ms'
          % (tf, tb, tf + tb))

    # ---- stage-by-stage attribution of the forward pass ----
    def t1(fn, n=6):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); queue.finish(); ts.append(time.perf_counter() - t0)
        return statistics.median(ts[2:]) * 1e3

    stage = {}
    stage['qkv gemm'] = t1(lambda: hgemm(dev['x'], dev['wqkv'], dev['qkv'], M, WROW, DH))
    stage['split_q'] = t1(lambda: K['split_q'](queue, (BH * T * D,), (256,), dev['qkv'], dev['Q'],
                                               np.int32(T), np.int32(D), np.int32(NH), np.int32(NKV)))
    stage['split_kv'] = t1(lambda: K['split_kv'](queue, (BH * T * D,), (256,), dev['qkv'], dev['K'],
                                                 dev['V'], np.int32(T), np.int32(D), np.int32(NH),
                                                 np.int32(NKV), np.int32(B)))
    stage['S batched'] = t1(lambda: hgemm_batched(dev['Q'], dev['K'], dev['S'], T, T, D, NO, YES,
                                                  D, D, T, T * D, T * D, T * T))
    stage['softmax S->P'] = t1(lambda: K['softmax_causal'](queue, (T * 256, BH), (256, 1), dev['S'], dev['P'],
                                                           np.int32(T), np.float32(scale)))
    stage['O batched'] = t1(lambda: hgemm_batched(dev['P'], dev['V'], dev['O'], T, D, T, NO, NO,
                                                  T, D, D, T * T, T * D, T * D))
    stage['merge_o'] = t1(lambda: K['merge_o'](queue, (BH * T * D,), (256,), dev['O'], dev['o_tok'],
                                               np.int32(T), np.int32(D), np.int32(NH), np.int32(B)))
    stage['out gemm'] = t1(lambda: hgemm(dev['o_tok'], dev['wo'], dev['out'], M, DH, DH))
    print('forward stages (ms): ' + ' | '.join('%s %.2f' % (k, v) for k, v in stage.items()))
    print('sum of stages %.2f ms (measured fwd %.2f ms)' % (sum(stage.values()), tf))
    print('full AR layer estimate: + MLP+norm block (54.7 ms) = %.1f ms -> 10 layers %.2f s'
          % (54.7 + tf + tb, (54.7 + tf + tb) * 10 / 1000))
    print('1030 fraction (floor 0.594 s): %.2fx'
          % (0.594 / ((54.7 + tf + tb) * 10 / 1000)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
