"""Integrated full-layer training step on the iGPU: attention + MLP, forward + backward + AdamW.

One AR layer (attention -> residual -> MLP -> residual) as a single device-resident pipeline: one
queue, one synchronisation at the end of the whole step. The measured number is a real step, not a
sum of two halves.

Dimensions: M = 2048 tokens/step (8 batches x 256), MD = 512 model dim, HD = 64 head dim,
NH/NKV = 8/4 heads, FF = 1368 SwiGLU hidden.

Weights are stored (in,out) fp16 only; the backward uses transA/transB CLBlast calls instead of
materialising transposed copies. AdamW keeps fp32 master+moments and writes the single fp16 copy.
Attention kernels/softmax come from gt1030/attention_gpu.py, RMSNorm/SiLU kernels from
gt1030/full_gpu_step_v2.py (imported, not copied).

Correctness: layer output vs a torch fp32 reference on the same fp16 inputs (per-part gradients were
validated in attention_gpu.py: dWo/dWqkv/dx and the MLP dW2).
"""
from __future__ import annotations

import ctypes
import statistics as _st
import sys
import time as _time

import numpy as np
import pyopencl as cl
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.attention_gpu as A
import gt1030.full_gpu_step_v2 as V2

M = V2.M                 # 2048 tokens/step
MD = V2.D                # 512 model dim
FF = V2.FF               # 1368
B, T, NH, NKV = A.B, A.T, A.NH, A.NKV
HD = A.D                 # 64 head dim
WROW = A.WROW            # 1024
BH = A.BH
scale = 1.0 / (HD ** 0.5)

EXTRA_KERNELS = r'''
__kernel void add_vec(__global const half* a, __global const half* b, __global half* o, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    o[i] = (half)((float)a[i] + (float)b[i]);
}
__kernel void adamw_single(__global float* master, __global float* mom, __global float* vel,
                           __global const half* grad, __global half* w,
                           int n, float lr, float b1, float b2, float eps, float wd,
                           float bc1, float bc2) {
    int i4 = get_global_id(0);
    int base = i4 * 4;
    if (base >= n) return;
    float4 g = (float4)((float)grad[base], (float)grad[base+1],
                        (float)grad[base+2], (float)grad[base+3]);
    float4 w4 = vload4(0, master + base);
    float4 m4 = vload4(0, mom + base);
    float4 v4 = vload4(0, vel + base);
    m4 = b1*m4 + (1.f-b1)*g;
    v4 = b2*v4 + (1.f-b2)*g*g;
    float4 wh = w4 - lr*(m4/bc1)/(sqrt(v4/bc2)+eps) - lr*wd*w4;
    vstore4(m4, 0, mom + base);
    vstore4(v4, 0, vel + base);
    vstore4(wh, 0, master + base);
    w[base] = (half)wh.x; w[base+1] = (half)wh.y;
    w[base+2] = (half)wh.z; w[base+3] = (half)wh.w;
}
'''


def hgemm_nn(a, b, c, m, n, k, beta=0.0):
    rc = A.bl.CLBlastHgemm(A.ROW, A.NO, A.NO, m, n, k, A.ONE,
                           ctypes.c_void_p(int(a.int_ptr)), 0, k,
                           ctypes.c_void_p(int(b.int_ptr)), 0, n,
                           A.ONE if beta else A.ZERO,
                           ctypes.c_void_p(int(c.int_ptr)), 0, n,
                           ctypes.byref(A.QP), None)
    if rc != 0:
        raise RuntimeError('gemm_nn rc=%d %dx%dx%d' % (rc, m, n, k))


def hgemm_nt(a, b, c, m, n, k, beta=0.0):
    """C = A @ B^T, A (m,k), B (n,k)."""
    rc = A.bl.CLBlastHgemm(A.ROW, A.NO, A.YES, m, n, k, A.ONE,
                           ctypes.c_void_p(int(a.int_ptr)), 0, k,
                           ctypes.c_void_p(int(b.int_ptr)), 0, k,
                           A.ONE if beta else A.ZERO,
                           ctypes.c_void_p(int(c.int_ptr)), 0, n,
                           ctypes.byref(A.QP), None)
    if rc != 0:
        raise RuntimeError('gemm_nt rc=%d %dx%dx%d' % (rc, m, n, k))


def hgemm_tn(a, b, c, m, n, k, beta=0.0):
    """C = A^T @ B, A stored (k,m), B stored (k,n)."""
    rc = A.bl.CLBlastHgemm(A.ROW, A.YES, A.NO, m, n, k, A.ONE,
                           ctypes.c_void_p(int(a.int_ptr)), 0, m,
                           ctypes.c_void_p(int(b.int_ptr)), 0, n,
                           A.ONE if beta else A.ZERO,
                           ctypes.c_void_p(int(c.int_ptr)), 0, n,
                           ctypes.byref(A.QP), None)
    if rc != 0:
        raise RuntimeError('gemm_tn rc=%d %dx%dx%d' % (rc, m, n, k))


def main() -> int:
    import os
    _ov = os.environ.get('AR_OVERRIDE', '')
    if _ov:
        pairs = []
        for item in _ov.split(','):
            k, v = item.split('=')
            pairs.append((k.strip(), int(v)))
        names = (ctypes.c_char_p * len(pairs))(*[k.encode() for k, _ in pairs])
        vals = (ctypes.c_size_t * len(pairs))(*[v for _, v in pairs])
        A.bl.CLBlastOverrideParameters.restype = ctypes.c_int
        A.bl.CLBlastOverrideParameters.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
        rc = A.bl.CLBlastOverrideParameters(ctypes.c_void_p(int(A.ctx.devices[0].int_ptr)),
                                           b'Xgemm', 16, len(pairs), names, vals)
        print('override rc=%d %s' % (rc, _ov), flush=True)

    rng = np.random.default_rng(0)
    prog = cl.Program(A.ctx, A.KERNELS + V2.KERNELS + EXTRA_KERNELS).build(
        options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
    names = ('split_q', 'split_kv', 'softmax_causal', 'softmax_bwd_causal', 'merge_o',
             'split_dout', 'merge_dqkv', 'rms_stats', 'rms_apply', 'rms_bwd_stats',
             'rms_bwd_apply', 'silu_gate', 'silu_bwd', 'add_vec', 'adamw_single')
    K = {n: cl.Kernel(prog, n) for n in names}
    import collections, time as _time
    REC = collections.defaultdict(list)
    def _w(fn, name):
        def call(*args, **kwargs):
            A.queue.finish()
            t0 = _time.perf_counter()
            r = fn(*args, **kwargs)
            A.queue.finish()
            REC[name].append(_time.perf_counter() - t0)
            return r
        return call
    for _n in list(K):
        K[_n] = _w(K[_n], 'K:' + _n)
    for _n in ('hgemm_nn', 'hgemm_nt', 'hgemm_tn'):
        globals()[_n] = _w(globals()[_n], 'G:' + _n)
    A.hgemm_batched = _w(A.hgemm_batched, 'G:batched')


    def buf(nb):
        return cl.Buffer(A.ctx, A.mf.READ_WRITE, nb)

    def upload(a):
        return cl.Buffer(A.ctx, A.mf.READ_WRITE | A.mf.COPY_HOST_PTR, hostbuf=a)

    def mk_weight(shape):
        w = (rng.standard_normal(shape).astype(np.float32) * 0.02).astype(np.float16)
        return dict(host=w, dev=upload(w), master=upload(w.astype(np.float32).copy()),
                    mom=upload(np.zeros(shape, dtype=np.float32)),
                    vel=upload(np.zeros(shape, dtype=np.float32)))

    WQKV = mk_weight((MD, WROW))
    WO = mk_weight((MD, MD))
    W1 = mk_weight((MD, FF))
    W3 = mk_weight((MD, FF))
    W2 = mk_weight((FF, MD))

    x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
    dout = (rng.standard_normal((M, MD)).astype(np.float32) * 0.01).astype(np.float16)
    wnorm = np.ones(MD, dtype=np.float16)

    def hbuf(rows, cols):
        return buf(rows * cols * 2)

    b = {
        'x': upload(x), 'dout': upload(dout), 'wnorm': upload(wnorm),
        'wqkv': WQKV['dev'], 'wo': WO['dev'], 'w1': W1['dev'], 'w3': W3['dev'], 'w2': W2['dev'],
        'h1': hbuf(M, MD), 'qkv': hbuf(M, WROW),
        'Q': hbuf(BH * T, HD), 'Kk': hbuf(BH * T, HD), 'V': hbuf(BH * T, HD),
        'S': hbuf(BH * T, T), 'P': hbuf(BH * T, T), 'O': hbuf(BH * T, HD),
        'o_tok': hbuf(M, MD), 'attn': hbuf(M, MD),
        'x1': hbuf(M, MD), 'h2': hbuf(M, MD),
        'a': hbuf(M, FF), 'bb': hbuf(M, FF), 'g': hbuf(M, FF), 'y': hbuf(M, MD), 'x2': hbuf(M, MD),
        'ss': buf(M * 4), 's': buf(M * 4),
        'dattn': hbuf(M, MD), 'dx1': hbuf(M, MD), 'dO_tok': hbuf(M, MD),
        'dP': hbuf(BH * T, T), 'dS': hbuf(BH * T, T),
        'dO': hbuf(BH * T, HD), 'dQ': hbuf(BH * T, HD), 'dK': hbuf(BH * T, HD), 'dV': hbuf(BH * T, HD),
        'dqkv': hbuf(M, WROW),
        'dh1': hbuf(M, MD), 'dh2': hbuf(M, MD),
        'dy_mlp': hbuf(M, FF), 'da': hbuf(M, FF), 'db': hbuf(M, FF),
        'dWqkv': hbuf(MD, WROW), 'dWo': hbuf(MD, MD),
        'dW1': hbuf(MD, FF), 'dW3': hbuf(MD, FF), 'dW2': hbuf(FF, MD),
        'dx1_mlp': hbuf(M, MD), 'dx_attn': hbuf(M, MD), 'dx': hbuf(M, MD),
    }

    def forward():
        K['rms_stats'](A.queue, (M,), (64,), b['x'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](A.queue, (M * MD,), (256,), b['x'], b['wnorm'], b['ss'], b['h1'],
                       np.int32(M), np.int32(MD))
        hgemm_nn(b['h1'], b['wqkv'], b['qkv'], M, WROW, MD)
        K['split_q'](A.queue, (BH * T * HD,), (256,), b['qkv'], b['Q'],
                     np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
        K['split_kv'](A.queue, (BH * T * HD,), (256,), b['qkv'], b['Kk'], b['V'],
                      np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        A.hgemm_batched(b['Q'], b['Kk'], b['S'], T, T, HD, A.NO, A.YES,
                        HD, HD, T, T * HD, T * HD, T * T)
        K['softmax_causal'](A.queue, (T * 256, BH), (256, 1), b['S'], b['P'],
                            np.int32(T), np.float32(scale))
        A.hgemm_batched(b['P'], b['V'], b['O'], T, HD, T, A.NO, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        K['merge_o'](A.queue, (BH * T * HD,), (256,), b['O'], b['o_tok'],
                     np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
        hgemm_nn(b['o_tok'], b['wo'], b['attn'], M, MD, MD)
        K['add_vec'](A.queue, (M * MD,), (256,), b['x'], b['attn'], b['x1'], np.int32(M * MD))
        K['rms_stats'](A.queue, (M,), (64,), b['x1'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](A.queue, (M * MD,), (256,), b['x1'], b['wnorm'], b['ss'], b['h2'],
                       np.int32(M), np.int32(MD))
        hgemm_nn(b['h2'], b['w1'], b['a'], M, FF, MD)
        hgemm_nn(b['h2'], b['w3'], b['bb'], M, FF, MD)
        K['silu_gate'](A.queue, (M * FF,), (256,), b['a'], b['bb'], b['g'], np.int32(M * FF))
        hgemm_nn(b['g'], b['w2'], b['y'], M, MD, FF)
        K['add_vec'](A.queue, (M * MD,), (256,), b['x1'], b['y'], b['x2'], np.int32(M * MD))

    def backward():
        # MLP branch (gradient of the final residual)
        hgemm_tn(b['g'], b['dout'], b['dW2'], FF, MD, M)                  # dW2 = g^T dOut
        hgemm_nt(b['dout'], b['w2'], b['dy_mlp'], M, FF, MD)              # dG = dOut W2^T
        K['silu_bwd'](A.queue, (M * FF,), (256,), b['a'], b['bb'], b['dy_mlp'],
                      b['da'], b['db'], np.int32(M * FF))
        hgemm_tn(b['h2'], b['da'], b['dW1'], MD, FF, M)                   # dW1 = h2^T da
        hgemm_tn(b['h2'], b['db'], b['dW3'], MD, FF, M)                   # dW3 = h2^T db
        hgemm_nt(b['da'], b['w1'], b['dh2'], M, MD, FF)                   # dh2 = da W1^T
        hgemm_nt(b['db'], b['w3'], b['dh2'], M, MD, FF, beta=1.0)         #      += db W3^T
        K['rms_bwd_stats'](A.queue, (M,), (64,), b['x1'], b['dh2'], b['wnorm'],
                           b['ss'], b['s'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](A.queue, (M * MD,), (256,), b['x1'], b['dh2'], b['wnorm'],
                           b['ss'], b['s'], b['dx1_mlp'], np.int32(M), np.int32(MD))
        K['add_vec'](A.queue, (M * MD,), (256,), b['dout'], b['dx1_mlp'], b['dattn'],
                     np.int32(M * MD))
        # attention branch
        hgemm_tn(b['o_tok'], b['dattn'], b['dWo'], MD, MD, M)             # dWo = O^T dAttn
        hgemm_nt(b['dattn'], b['wo'], b['dO_tok'], M, MD, MD)             # dO_tok = dAttn Wo^T
        K['split_dout'](A.queue, (BH * T * HD,), (256,), b['dO_tok'], b['dO'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
        A.hgemm_batched(b['dO'], b['V'], b['dP'], T, T, HD, A.NO, A.YES,
                        HD, HD, T, T * HD, T * HD, T * T)
        K['softmax_bwd_causal'](A.queue, (T * 256, BH), (256, 1), b['P'], b['dP'], b['dS'],
                                np.int32(T), np.float32(scale))
        A.hgemm_batched(b['P'], b['dO'], b['dV'], T, HD, T, A.YES, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        A.hgemm_batched(b['dS'], b['Kk'], b['dQ'], T, HD, T, A.NO, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        A.hgemm_batched(b['dS'], b['Q'], b['dK'], T, HD, T, A.YES, A.NO,
                        T, HD, HD, T * T, T * HD, T * HD)
        K['merge_dqkv'](A.queue, (M * WROW,), (256,), b['dQ'], b['dK'], b['dV'], b['dqkv'],
                        np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
        hgemm_tn(b['h1'], b['dqkv'], b['dWqkv'], MD, WROW, M)             # dWqkv = h1^T dqkv
        hgemm_nt(b['dqkv'], b['wqkv'], b['dh1'], M, MD, WROW)             # dh1 = dqkv Wqkv^T
        K['rms_bwd_stats'](A.queue, (M,), (64,), b['x'], b['dh1'], b['wnorm'],
                           b['ss'], b['s'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](A.queue, (M * MD,), (256,), b['x'], b['dh1'], b['wnorm'],
                           b['ss'], b['s'], b['dx_attn'], np.int32(M), np.int32(MD))
        K['add_vec'](A.queue, (M * MD,), (256,), b['dattn'], b['dx_attn'], b['dx'],
                     np.int32(M * MD))

    def opt():
        for grad, wdict, dims in ((b['dWqkv'], WQKV, (MD, WROW)), (b['dWo'], WO, (MD, MD)),
                                  (b['dW1'], W1, (MD, FF)), (b['dW3'], W3, (MD, FF)),
                                  (b['dW2'], W2, (FF, MD))):
            n = dims[0] * dims[1]
            K['adamw_single'](A.queue, (n // 4,), (256,), wdict['master'], wdict['mom'],
                              wdict['vel'], grad, wdict['dev'], np.int32(n),
                              np.float32(1e-4), np.float32(0.9), np.float32(0.999),
                              np.float32(1e-8), np.float32(0.0), np.float32(1.0), np.float32(1.0))

    # ---- correctness: layer output vs torch fp32 on the same fp16 inputs ----
    forward()
    A.queue.finish()
    out = np.empty((M, MD), dtype=np.float16)
    cl.enqueue_copy(A.queue, out, b['x2'])
    A.queue.finish()
    xt = torch.from_numpy(x.astype(np.float32))
    wq = torch.from_numpy(WQKV['host'].astype(np.float32))
    wot = torch.from_numpy(WO['host'].astype(np.float32))
    w1t = torch.from_numpy(W1['host'].astype(np.float32))
    w3t = torch.from_numpy(W3['host'].astype(np.float32))
    w2t = torch.from_numpy(W2['host'].astype(np.float32))
    wn = torch.from_numpy(wnorm.astype(np.float32))
    h1r = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6) * wn
    qkvr = h1r @ wq
    orow = torch.zeros(M, MD)
    for bi in range(B):
        for hh in range(NH):
            rows = slice(bi * T, (bi + 1) * T)
            q = qkvr[rows, hh * HD:(hh + 1) * HD]
            kw = hh // 2
            k = qkvr[rows, MD + kw * HD:MD + (kw + 1) * HD]
            v = qkvr[rows, MD + NKV * HD + kw * HD:MD + NKV * HD + (kw + 1) * HD]
            s = (q @ k.T) * scale
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            s = s.masked_fill(~mask, float('-inf'))
            orow[rows, hh * HD:(hh + 1) * HD] = torch.softmax(s, -1) @ v
    x1r = xt + orow @ wot
    h2r = x1r * torch.rsqrt(x1r.pow(2).mean(-1, keepdim=True) + 1e-6) * wn
    x2r = x1r + (torch.nn.functional.silu(h2r @ w1t) * (h2r @ w3t)) @ w2t
    err = float(np.abs(out.astype(np.float32) - x2r.numpy()).max())
    print('integrated layer forward: max abs err %.3e (relative %.3e)'
          % (err, err / (float(np.abs(x2r.numpy()).max()) + 1e-9)))
    # ---- gradient check: one layer backward vs torch autograd ----
    def torch_layer(x_np, wqkv_np, wo_np, w1_np, w3_np, w2_np, wn_np, dout_np):
        xt = torch.from_numpy(x_np.astype(np.float32)).requires_grad_(True)
        wq = torch.from_numpy(wqkv_np.astype(np.float32)).requires_grad_(True)
        wot = torch.from_numpy(wo_np.astype(np.float32)).requires_grad_(True)
        w1t = torch.from_numpy(w1_np.astype(np.float32)).requires_grad_(True)
        w3t = torch.from_numpy(w3_np.astype(np.float32)).requires_grad_(True)
        w2t = torch.from_numpy(w2_np.astype(np.float32)).requires_grad_(True)
        wn = torch.from_numpy(wn_np.astype(np.float32))
        h1r = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6) * wn
        qkvr = h1r @ wq
        orow = torch.zeros(M, MD)
        for bi in range(B):
            for hh in range(NH):
                rows = slice(bi * T, (bi + 1) * T)
                q = qkvr[rows, hh * HD:(hh + 1) * HD]
                kw = hh // 2
                k = qkvr[rows, MD + kw * HD:MD + (kw + 1) * HD]
                v = qkvr[rows, MD + NKV * HD + kw * HD:MD + NKV * HD + (kw + 1) * HD]
                s = (q @ k.T) * scale
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
                s = s.masked_fill(~mask, float('-inf'))
                orow[rows, hh * HD:(hh + 1) * HD] = torch.softmax(s, -1) @ v
        x1r = xt + orow @ wot
        h2r = x1r * torch.rsqrt(x1r.pow(2).mean(-1, keepdim=True) + 1e-6) * wn
        x2r = x1r + (torch.nn.functional.silu(h2r @ w1t) * (h2r @ w3t)) @ w2t
        (x2r * torch.from_numpy(dout_np.astype(np.float32))).sum().backward()
        return dict(dx=xt.grad, dWqkv=wq.grad, dWo=wot.grad, dW1=w1t.grad,
                    dW3=w3t.grad, dW2=w2t.grad)

    def download(buf_, shape):
        out_ = np.empty(shape, dtype=np.float16)
        cl.enqueue_copy(A.queue, out_, buf_)
        A.queue.finish()
        return out_.astype(np.float32)

    ref_g = torch_layer(x, WQKV['host'], WO['host'], W1['host'], W3['host'],
                        W2['host'], wnorm, dout)
    # run a fresh backward for the check (do not let the check's forward alter weights)
    forward(); backward(); A.queue.finish()
    gpu_g = {'dx': download(b['dx'], (M, MD)),
             'dWqkv': download(b['dWqkv'], (MD, WROW)),
             'dWo': download(b['dWo'], (MD, MD)),
             'dW1': download(b['dW1'], (MD, FF)),
             'dW3': download(b['dW3'], (MD, FF)),
             'dW2': download(b['dW2'], (FF, MD))}
    print('gradient check (max abs / rel vs torch fp32):')
    for k, v in gpu_g.items():
        r = ref_g[k].detach().numpy()
        denom = float(np.abs(r).max()) + 1e-12
        print('  %-6s abs %.3e  rel %.3e' % (k, np.abs(v - r).max(), np.abs(v - r).max() / denom))

    for _ in range(3):
        forward(); backward(); A.queue.finish()
    import statistics as _st
    print('PHASE_TABLE (flush-then-time, one layer):', flush=True)
    for _k, _v in sorted(REC.items(), key=lambda kv: -_st.median(kv[1][2:] or kv[1])):
        _vv = _v[2:] if len(_v) > 2 else _v
        print('  %-16s median %7.2f ms  n=%d' % (_k, _st.median(_vv) * 1e3, len(_v)), flush=True)

    def phase(fn, n=6):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            A.queue.finish()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts[2:]) * 1e3

    tf = phase(forward)
    tb = phase(backward)
    to = phase(opt)
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        forward()
        backward()
        opt()
        A.queue.finish()
        times.append(time.perf_counter() - t0)
    print('raw combined times (ms): %s' % [round(v * 1e3, 2) for v in times])
    med_ms = statistics.median(times[2:]) * 1e3
    print()
    print('integrated layer step: %.2f ms  (separately: fwd %.2f | bwd %.2f | AdamW %.2f)'
          % (med_ms, tf, tb, to))
    print('10 layers: %.2f s -> 1030 fraction (floor 0.594 s): %.2fx'
          % (med_ms * 10 / 1000, 0.594 / (med_ms * 10 / 1000)))

    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if n_steps > 0:
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        for _ in range(n_steps):
            forward()
            backward()
            opt()
            A.queue.finish()
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        print()
        print('pure-iGPU loop: %d steps in %.2f s (%.2f ms/step); CPU busy %.1f%% of wall'
              % (n_steps, wall, wall / n_steps * 1e3, 100.0 * cpu / wall))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
