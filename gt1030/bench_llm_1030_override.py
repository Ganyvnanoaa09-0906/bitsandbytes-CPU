"""Acceptance harness for CLBlast CLBlastOverrideParameters tuning results.

Loads the same clblast.dll, overrides Xgemm (half) tuning parameters for the current
device, then benchmarks the six Qwen3-8B / 1.7B shapes with Hgemm and writes JSON.

Args:
  --params "GEMMK=0,KREG=1,KWG=32,KWI=2,MDIMA=16,MDIMC=16,MWG=64,NDIMB=8,NDIMC=8,NWG=64,SA=1,SB=1,STRM=0,STRN=0,VWM=4,VWN=4"
  --json   optional output path (default research/bench_results/igpu_vs_1030_qwen3_8b_override.json)
  --iters  number of timed iterations (default 5)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
REF_THEORY = 1127.0
REF_PRACTICAL = 631.0

SHAPES = {
    'Qwen3-8B qkv/o': (512, 4096, 4096),
    'Qwen3-8B mlp up': (512, 12288, 4096),
    'Qwen3-8B mlp down': (512, 4096, 12288),
    'Qwen3-1.7B qkv/o': (512, 2048, 2048),
    'Qwen3-1.7B mlp up': (512, 6144, 2048),
    'Qwen3-1.7B mlp down': (512, 2048, 6144),
}
TARGET = {'Qwen3-8B qkv/o', 'Qwen3-8B mlp up', 'Qwen3-8B mlp down'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--json', default=r'D:\work\research\bench_results\igpu_vs_1030_qwen3_8b_override.json')
    ap.add_argument('--iters', type=int, default=5)
    args = ap.parse_args()

    pairs = []
    for item in args.params.split(','):
        k, v = item.split('=')
        pairs.append((k.strip(), int(v)))
    names = (ctypes.c_char_p * len(pairs))(*[k.encode() for k, _ in pairs])
    values = (ctypes.c_size_t * len(pairs))(*[v for _, v in pairs])

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
    dev = ctx.devices[0]
    qp = ctypes.c_void_p(int(q.int_ptr))
    devp = ctypes.c_void_p(int(dev.int_ptr))

    rc = bl.CLBlastOverrideParameters(devp, b'Xgemm', 16, len(pairs), names, values)
    print('device %s  override rc=%d  %s' % (dev.name, rc, args.params), flush=True)
    if rc != 0:
        raise SystemExit('override failed rc=%d' % rc)

    results = {}
    for name, (m, n, k) in SHAPES.items():
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((m, k)).astype(np.float32) * 0.02).astype(np.float16)
        b = (rng.standard_normal((k, n)).astype(np.float32) * 0.02).astype(np.float16)
        af = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        bf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        cf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, m * n * 2)
        c = np.empty((m, n), dtype=np.float16)

        def call():
            return bl.CLBlastHgemm(101, 111, 111, m, n, k, ONE,
                            ctypes.c_void_p(int(af.int_ptr)), 0, k,
                            ctypes.c_void_p(int(bf.int_ptr)), 0, n, ZERO,
                            ctypes.c_void_p(int(cf.int_ptr)), 0, n,
                            ctypes.byref(qp), None)

        rc = call(); q.finish()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter(); call(); q.finish(); samples.append(time.perf_counter() - t0)
        sec = float(np.median(samples))
        cl.enqueue_copy(q, c, cf).wait()
        ref = a.astype(np.float64) @ b.astype(np.float64)
        err = float(np.max(np.abs(c.astype(np.float64) - ref)))
        gflops = 2.0 * m * n * k / sec / 1e9
        results[name] = dict(M=m, N=n, K=k, gflops=round(gflops, 1), ms=round(sec * 1e3, 3),
                             ratio_theory=round(gflops / REF_THEORY, 4),
                             ratio_practical=round(gflops / REF_PRACTICAL, 4),
                             max_err=err, rc=rc)
        print('%-20s %6d %6d %6d  %8.1f GFLOPS  %.3f x theory  %.3f x pract  err=%.3f rc=%d'
              % (name, m, n, k, gflops, gflops / REF_THEORY, gflops / REF_PRACTICAL, err, rc),
              flush=True)

    targets = [results[n]['ratio_theory'] for n in TARGET]
    out = dict(reference_theory=REF_THEORY, reference_practical=REF_PRACTICAL,
               override_params=args.params,
               min_target_ratio=round(min(targets), 4), results=results)
    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('min target ratio %.3f -> %s' % (min(targets), 'PASS' if min(targets) >= 1.0 else 'FAIL'))
    print('saved:', args.json)


if __name__ == '__main__':
    main()
