"""10-layer full-GPU AR stack: actual assembly, not 10x extrapolation.

Each Layer owns independent weights, fp32 AdamW state and activation buffers. The stack is
chained forward (x2 of layer i is x of layer i+1) and chained backward (dx of layer i+1 is dout
of layer i). No host round-trip inside the step; only final download for the timing harness.

This file deliberately stops before embedding/lm_head/CE loss: its purpose is to measure the
assembly/scheduling overhead that the single-layer x10 estimate hides.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_layer as L  # noqa: E402

M, MD, FF, WROW = L.M, L.MD, L.FF, L.WROW
B, T, NH, NKV, HD, BH = L.B, L.T, L.NH, L.NKV, L.HD, L.BH
scale = L.scale
A = L.A

prog = cl.Program(A.ctx, A.KERNELS + L.V2.KERNELS + L.EXTRA_KERNELS).build(
    options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
NAMES = ('split_q', 'split_kv', 'softmax_causal', 'softmax_bwd_causal', 'merge_o',
         'split_dout', 'merge_dqkv', 'rms_stats', 'rms_apply', 'rms_bwd_stats',
         'rms_bwd_apply', 'silu_gate', 'silu_bwd', 'add_vec', 'adamw_single')
K = {n: cl.Kernel(prog, n) for n in NAMES}


def buf(nb):
    return cl.Buffer(A.ctx, A.mf.READ_WRITE, nb)


def upload(a):
    # Explicit copy instead of COPY_HOST_PTR: allocating many COPY_HOST_PTR buffers
    # was measured to corrupt previously computed buffers on this driver.
    b = cl.Buffer(A.ctx, A.mf.READ_WRITE, a.nbytes)
    cl.enqueue_copy(A.queue, b, a)
    return b


class Layer:
    def __init__(self, rng, n):
        self.n = n
        self.WQKV = self._w(rng, (MD, WROW))
        self.WO = self._w(rng, (MD, MD))
        self.W1 = self._w(rng, (MD, FF))
        self.W3 = self._w(rng, (MD, FF))
        self.W2 = self._w(rng, (FF, MD))
        self.b = {
            'x': None, 'dout': None, 'wnorm': upload(np.ones(MD, dtype=np.float16)),
            'wqkv': self.WQKV['dev'], 'wo': self.WO['dev'],
            'w1': self.W1['dev'], 'w3': self.W3['dev'], 'w2': self.W2['dev'],
            'h1': buf(M * MD * 2), 'qkv': buf(M * WROW * 2),
            'Q': buf(BH * T * HD * 2), 'Kk': buf(BH * T * HD * 2), 'V': buf(BH * T * HD * 2),
            'S': buf(BH * T * T * 2), 'P': buf(BH * T * T * 2), 'O': buf(BH * T * HD * 2),
            'o_tok': buf(M * MD * 2), 'attn': buf(M * MD * 2),
            'x1': buf(M * MD * 2), 'h2': buf(M * MD * 2),
            'a': buf(M * FF * 2), 'bb': buf(M * FF * 2), 'g': buf(M * FF * 2),
            'y': buf(M * MD * 2), 'x2': buf(M * MD * 2),
            'ss': buf(M * 4), 's': buf(M * 4),
            'dattn': buf(M * MD * 2), 'dx1': buf(M * MD * 2), 'dO_tok': buf(M * MD * 2),
            'dP': buf(BH * T * T * 2), 'dS': buf(BH * T * T * 2),
            'dO': buf(BH * T * HD * 2), 'dQ': buf(BH * T * HD * 2),
            'dK': buf(BH * T * HD * 2), 'dV': buf(BH * T * HD * 2),
            'dqkv': buf(M * WROW * 2), 'dh1': buf(M * MD * 2), 'dh2': buf(M * MD * 2),
            'dy_mlp': buf(M * FF * 2), 'da': buf(M * FF * 2), 'db': buf(M * FF * 2),
            'dWqkv': buf(MD * WROW * 2), 'dWo': buf(MD * MD * 2),
            'dW1': buf(MD * FF * 2), 'dW3': buf(MD * FF * 2), 'dW2': buf(FF * MD * 2),
            'dx1_mlp': buf(M * MD * 2), 'dx_attn': buf(M * MD * 2), 'dx': buf(M * MD * 2),
        }

    def _w(self, rng, shape):
        w = (rng.standard_normal(shape).astype(np.float32) * 0.02).astype(np.float16)
        return dict(host=w, dev=upload(w), master=upload(w.astype(np.float32).copy()),
                    mom=upload(np.zeros(shape, dtype=np.float32)),
                    vel=upload(np.zeros(shape, dtype=np.float32)))

    def forward(self, x_in):
        b = self.b
        b['x'] = x_in
        K['rms_stats'](A.queue, (M,), (64,), b['x'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](A.queue, (M * MD,), (256,), b['x'], b['wnorm'], b['ss'], b['h1'],
                       np.int32(M), np.int32(MD))
        L.hgemm_nn(b['h1'], b['wqkv'], b['qkv'], M, WROW, MD)
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
        L.hgemm_nn(b['o_tok'], b['wo'], b['attn'], M, MD, MD)
        K['add_vec'](A.queue, (M * MD,), (256,), b['x'], b['attn'], b['x1'], np.int32(M * MD))
        K['rms_stats'](A.queue, (M,), (64,), b['x1'], b['ss'], np.int32(M), np.int32(MD))
        K['rms_apply'](A.queue, (M * MD,), (256,), b['x1'], b['wnorm'], b['ss'], b['h2'],
                       np.int32(M), np.int32(MD))
        L.hgemm_nn(b['h2'], b['w1'], b['a'], M, FF, MD)
        L.hgemm_nn(b['h2'], b['w3'], b['bb'], M, FF, MD)
        K['silu_gate'](A.queue, (M * FF,), (256,), b['a'], b['bb'], b['g'], np.int32(M * FF))
        L.hgemm_nn(b['g'], b['w2'], b['y'], M, MD, FF)
        K['add_vec'](A.queue, (M * MD,), (256,), b['x1'], b['y'], b['x2'], np.int32(M * MD))
        return b['x2']

    def backward(self, dout):
        b = self.b
        b['dout'] = dout
        L.hgemm_tn(b['g'], b['dout'], b['dW2'], FF, MD, M)
        L.hgemm_nt(b['dout'], b['w2'], b['dy_mlp'], M, FF, MD)
        K['silu_bwd'](A.queue, (M * FF,), (256,), b['a'], b['bb'], b['dy_mlp'],
                      b['da'], b['db'], np.int32(M * FF))
        L.hgemm_tn(b['h2'], b['da'], b['dW1'], MD, FF, M)
        L.hgemm_tn(b['h2'], b['db'], b['dW3'], MD, FF, M)
        L.hgemm_nt(b['da'], b['w1'], b['dh2'], M, MD, FF)
        L.hgemm_nt(b['db'], b['w3'], b['dh2'], M, MD, FF, beta=1.0)
        K['rms_bwd_stats'](A.queue, (M,), (64,), b['x1'], b['dh2'], b['wnorm'],
                           b['ss'], b['s'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](A.queue, (M * MD,), (256,), b['x1'], b['dh2'], b['wnorm'],
                           b['ss'], b['s'], b['dx1_mlp'], np.int32(M), np.int32(MD))
        K['add_vec'](A.queue, (M * MD,), (256,), b['dout'], b['dx1_mlp'], b['dattn'],
                     np.int32(M * MD))
        L.hgemm_tn(b['o_tok'], b['dattn'], b['dWo'], MD, MD, M)
        L.hgemm_nt(b['dattn'], b['wo'], b['dO_tok'], M, MD, MD)
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
        L.hgemm_tn(b['h1'], b['dqkv'], b['dWqkv'], MD, WROW, M)
        L.hgemm_nt(b['dqkv'], b['wqkv'], b['dh1'], M, MD, WROW)
        K['rms_bwd_stats'](A.queue, (M,), (64,), b['x'], b['dh1'], b['wnorm'],
                           b['ss'], b['s'], np.int32(M), np.int32(MD))
        K['rms_bwd_apply'](A.queue, (M * MD,), (256,), b['x'], b['dh1'], b['wnorm'],
                           b['ss'], b['s'], b['dx_attn'], np.int32(M), np.int32(MD))
        K['add_vec'](A.queue, (M * MD,), (256,), b['dattn'], b['dx_attn'], b['dx'],
                     np.int32(M * MD))
        return b['dx']

    def opt(self):
        for grad, wdict, dims in ((self.b['dWqkv'], self.WQKV, (MD, WROW)),
                                  (self.b['dWo'], self.WO, (MD, MD)),
                                  (self.b['dW1'], self.W1, (MD, FF)),
                                  (self.b['dW3'], self.W3, (MD, FF)),
                                  (self.b['dW2'], self.W2, (FF, MD))):
            n = dims[0] * dims[1]
            K['adamw_single'](A.queue, (n // 4,), (256,), wdict['master'], wdict['mom'],
                              wdict['vel'], grad, wdict['dev'], np.int32(n),
                              np.float32(1e-4), np.float32(0.9), np.float32(0.999),
                              np.float32(1e-8), np.float32(0.0), np.float32(1.0), np.float32(1.0))


def main() -> int:
    rng = np.random.default_rng(0)
    x = upload((rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16))
    dout = upload((rng.standard_normal((M, MD)).astype(np.float32) * 0.01).astype(np.float16))
    n_layers = int(os.environ.get('N_LAYERS', '10'))
    layers = [Layer(rng, i) for i in range(n_layers)]
    A.queue.finish()

    sync_each = os.environ.get('SYNC_EACH') == '1'

    def step():
        cur = x
        for lay in layers:
            cur = lay.forward(cur)
            if sync_each:
                A.queue.finish()
        grad = dout
        for lay in reversed(layers):
            grad = lay.backward(grad)
            if sync_each:
                A.queue.finish()
        for lay in layers:
            lay.opt()
            if sync_each:
                A.queue.finish()
        A.queue.finish()

    step()  # warmup
    ts = []
    for _ in range(6):
        t0 = time.perf_counter(); step(); ts.append(time.perf_counter() - t0)
    print('%d-layer actual assembly step: %.2f ms (median of %d, 1 warmup)'
          % (n_layers, statistics.median(ts[1:]) * 1e3, len(ts) - 1))
    print('raw ms: %s' % [round(v * 1e3, 1) for v in ts])
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
