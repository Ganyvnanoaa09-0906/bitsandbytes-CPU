"""Per-shape Hgemm benchmark for AR 30M training GEMMs, with CLBlast override params.

Shapes are the actual M=2048 layer GEMMs (fwd qkv/out/w1/w3/w2, bwd dW TN, bwd dX NT).
Usage: python bench_ar_gemm_sweep.py --params "..."  (empty = CLBlast default)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112
M, MD, FF, WROW = 2048, 512, 1368, 1024
ONE, ZERO = 0x3C00, 0x0000

SHAPES = [
    ('fwd_qkv', M, WROW, MD),
    ('fwd_out', M, MD, MD),
    ('fwd_w1', M, FF, MD),
    ('fwd_w3', M, FF, MD),
    ('fwd_w2', M, MD, FF),
    ('bwd_dW2', FF, MD, M),
    ('bwd_dW1', MD, FF, M),
    ('bwd_dW3', MD, FF, M),
    ('bwd_dWo', MD, MD, M),
    ('bwd_dWqkv', MD, WROW, M),
    ('bwd_dy_mlp', M, FF, MD),
    ('bwd_dh2', M, MD, FF),
    ('bwd_dO_tok', M, MD, MD),
    ('bwd_dh1', M, MD, WROW),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', default='')
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    bl = ctypes.CDLL(DLL)
    bl.CLBlastHgemm.restype = ctypes.c_int
    bl.CLBlastHgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ushort,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ushort,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    bl.CLBlastOverrideParameters.restype = ctypes.c_int
    bl.CLBlastOverrideParameters.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]

    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    qp = ctypes.c_void_p(int(q.int_ptr))
    if args.params:
        pairs = [(k.strip(), int(v)) for k, v in (x.split('=') for x in args.params.split(','))]
        names = (ctypes.c_char_p * len(pairs))(*[k.encode() for k, _ in pairs])
        vals = (ctypes.c_size_t * len(pairs))(*[v for _, v in pairs])
        rc = bl.CLBlastOverrideParameters(ctypes.c_void_p(int(ctx.devices[0].int_ptr)),
                                         b'Xgemm', 16, len(pairs), names, vals)
        print('override rc=%d' % rc, flush=True)

    out = {}
    for name, m, n, k in SHAPES:
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((m, k)).astype(np.float32) * 0.02).astype(np.float16)
        b = (rng.standard_normal((k, n)).astype(np.float32) * 0.02).astype(np.float16)
        ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        cb = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 2)
        c = np.empty((m, n), dtype=np.float16)

        def call():
            return bl.CLBlastHgemm(ROW, NO, NO, m, n, k, ONE,
                                   ctypes.c_void_p(int(ab.int_ptr)), 0, k,
                                   ctypes.c_void_p(int(bb.int_ptr)), 0, n, ZERO,
                                   ctypes.c_void_p(int(cb.int_ptr)), 0, n,
                                   ctypes.byref(qp), None)

        call(); q.finish()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter(); rc = call(); q.finish(); samples.append(time.perf_counter() - t0)
        sec = statistics.median(samples)
        cl.enqueue_copy(q, c, cb).wait()
        ref = a.astype(np.float64) @ b.astype(np.float64)
        err = float(np.max(np.abs(c.astype(np.float64) - ref)))
        gf = 2.0 * m * n * k / sec / 1e9
        out[name] = dict(M=m, N=n, K=k, gflops=round(gf, 1), ms=round(sec * 1e3, 3), err=err, rc=rc)
        print('%-12s %6d %6d %6d  %8.1f GF  %7.3f ms  err=%.3f' % (name, m, n, k, gf, sec * 1e3, err), flush=True)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'params': args.params, 'results': out}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
