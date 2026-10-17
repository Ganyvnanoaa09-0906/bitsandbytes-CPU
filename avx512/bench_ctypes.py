#!/usr/bin/env python3
"""AVX2 vs AVX-512 cross-check and benchmark for the fork's CPU kernels.

No torch, no numpy -- ctypes only, so any Linux box with python3 can run it. Two shared
libraries are expected (built by avx512/verify.sh):

    libbnb_avx2.so     -mavx2 -mfma -mf16c, all AVX-512 features explicitly disabled
    libbnb_avx512.so   same base plus -mavx512f/-bw/-dq/-vl(-bf16)

Why two libraries instead of one: the AVX-512 build is today a server-only build --
compiling the whole translation unit with -mavx512f makes GCC auto-vectorise ordinary
functions (optimizer update, dequantize, fused rmsnorm) with zmm registers, so the binary
dies with SIGILL on an AVX2-only CPU (measured: 1861 zmm lines, exit 132 on a Ryzen 4500U).
This harness therefore loads each build explicitly and only calls the AVX-512 one when the
CPU really has AVX-512. On the same inputs it reports correctness (AVX-512 vs AVX2) and
median wall time per kernel, then the speedup.
"""
from __future__ import annotations

import argparse
import ctypes as C
import statistics
import sys
import time

LL = C.c_longlong
FP = C.POINTER(C.c_float)
U8 = C.POINTER(C.c_ubyte)
INT = C.c_int


def cpuinfo_flags() -> str:
    try:
        with open('/proc/cpuinfo', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('flags'):
                    return line.split(':', 1)[1]
    except OSError:
        pass
    return ''


def has_avx512(flags: str) -> bool:
    return all(f in flags for f in ('avx512f', 'avx512bw', 'avx512dq', 'avx512vl'))


def load(path: str):
    lib = C.CDLL(path)
    lib.cquantize_blockwise_cpu_fp32.argtypes = [FP, FP, FP, U8, LL, LL]
    lib.cdequantize_blockwise_cpu_fp32.argtypes = [FP, U8, FP, FP, LL, LL]
    lib.cgemm_8bit_inference_cpu_fp32.argtypes = [FP, U8, FP, FP, LL, LL, LL, LL, LL, LL, LL]
    lib.cgemv_4bit_inference_cpu_fp32.argtypes = [FP, U8, FP, FP, LL, LL, LL, LL, LL, LL, LL, INT]
    lib.coptimizer_update_8bit_blockwise_cpu.argtypes = [
        INT, C.c_void_p, C.c_void_p, U8, U8,
        C.c_float, C.c_float, C.c_float, C.c_float, C.c_float, INT, C.c_float,
        FP, FP, FP, FP, C.c_float, C.c_float, INT, LL, INT,
    ]
    for name in ('cquantize_blockwise_cpu_fp32', 'cdequantize_blockwise_cpu_fp32',
                 'cgemm_8bit_inference_cpu_fp32', 'cgemv_4bit_inference_cpu_fp32',
                 'coptimizer_update_8bit_blockwise_cpu'):
        getattr(lib, name).restype = None
    return lib


CODE = (C.c_float * 256)(*[2.0 * i / 255.0 - 1.0 for i in range(256)])


def floats(n, fill):
    a = (C.c_float * n)()
    for i in range(n):
        a[i] = fill(i)
    return a


def timed(fn, reps: int) -> float:
    for _ in range(max(1, reps // 5)):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def compare(a, b, n: int, stride: int = 1):
    """max abs diff / max rel diff over a sampled subset (pure python, keep it cheap)."""
    worst_abs = 0.0
    worst_rel = 0.0
    for i in range(0, min(n, len(a), len(b)), stride):
        d = abs(float(a[i]) - float(b[i]))
        if d > worst_abs:
            worst_abs = d
        m = max(abs(float(a[i])), abs(float(b[i])))
        r = d / (m + 1e-9)
        if r > worst_rel:
            worst_rel = r
    return worst_abs, worst_rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--avx2', required=True)
    ap.add_argument('--avx512', default='')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    flags = cpuinfo_flags()
    print('avx* flags on this CPU: %s'
          % (' '.join(f for f in flags.split() if f.startswith('avx')) or '(none)'))
    cpu_avx512 = has_avx512(flags)
    print('CPU executes AVX-512  : %s' % ('YES' if cpu_avx512 else 'NO'))

    lib2 = load(args.avx2)
    lib5 = load(args.avx512) if (args.avx512 and cpu_avx512) else None
    if args.avx512 and not cpu_avx512:
        print('skip AVX-512 library  : CPU cannot run AVX-512, calling it would SIGILL')
    if lib5 is not None:
        caps = []
        for sym in ('has_avx512f_cpu', 'has_avx512bf16_cpu'):
            fn = getattr(lib5, sym, None)
            if fn is not None:
                fn.restype = C.c_bool
                caps.append('%s=%s' % (sym, bool(fn())))
        print('AVX-512 lib says      : %s' % (', '.join(caps) or '(no capability exports)'))

    results = []
    checks = []

    def record(name, ta, tb=None):
        results.append((name, ta, tb))
    q = 20 if args.quick else 22
    N = 1 << q

    # ------------------------------------------------------------- quantize 8-bit
    def make_quant(lib):
        A = floats(N, lambda i: (i % 97) / 97.0 - 0.5)
        absmax = floats(N // 256 + 2, lambda i: 0.0)
        out = (C.c_ubyte * N)()
        return (lambda: lib.cquantize_blockwise_cpu_fp32(CODE, A, absmax, out, 256, N)), out

    f2, out2 = make_quant(lib2)
    t2 = timed(f2, 8)
    if lib5:
        f5, out5 = make_quant(lib5)
        t5 = timed(f5, 8)
        record('quantize8_fp32', t2, t5)
        checks.append(('quantize8 output bytes identical',
                       bytes(out2) == bytes(out5)))
    else:
        record('quantize8_fp32', t2)

    # ----------------------------------------------------------- dequantize 8-bit
    def make_dequant(lib):
        A = floats(N, lambda i: (i % 89) / 89.0 - 0.5)
        absmax = floats(N // 256 + 2, lambda i: 0.0)
        q8 = (C.c_ubyte * N)()
        lib.cquantize_blockwise_cpu_fp32(CODE, A, absmax, q8, 256, N)
        out = floats(N, lambda i: 0.0)
        return (lambda: lib.cdequantize_blockwise_cpu_fp32(CODE, q8, absmax, out, 256, N)), out

    f2, out2 = make_dequant(lib2)
    t2 = timed(f2, 8)
    if lib5:
        f5, out5 = make_dequant(lib5)
        t5 = timed(f5, 8)
        record('dequantize8_fp32', t2, t5)
        d, r = compare(out2, out5, 65536, 4)
        checks.append(('dequantize8 max_rel=%.2e (max_abs=%.2e)' % (r, d), r < 1e-6))
    else:
        record('dequantize8_fp32', t2)

    # ---------------------------------------------------------------- gemm_8bit
    M, Nn, K, bs = (2, 1024, 1024, 64) if not args.quick else (1, 256, 256, 64)
    A = floats(M * K, lambda i: (i % 17) / 17.0 - 0.5)
    Bf = floats(Nn * K, lambda i: (i % 13) / 13.0 - 0.4)
    Bq = (C.c_ubyte * (Nn * K))()
    ab = floats(Nn * (K // bs), lambda i: 0.0)
    for row in range(Nn):
        lib2.cquantize_blockwise_cpu_fp32(
            CODE, C.cast(C.byref(Bf, row * K * 4), FP),
            C.cast(C.byref(ab, row * (K // bs) * 4), FP),
            C.cast(C.byref(Bq, row * K), U8), bs, K)
    out_g = floats(M * Nn, lambda i: 0.0)

    def make_gemm(lib):
        return (lambda: lib.cgemm_8bit_inference_cpu_fp32(A, Bq, ab, out_g, M, Nn, K, K, K, Nn, bs)), out_g

    f2, out2 = make_gemm(lib2)
    t2 = timed(f2, 8)
    ref_g = list(out_g)
    if lib5:
        f5, out5 = make_gemm(lib5)
        t5 = timed(f5, 8)
        record('gemm_8bit_inference', t2, t5)
        d, r = compare(ref_g, out5, M * Nn)
        checks.append(('gemm_8bit max_abs=%.2e max_rel=%.2e' % (d, r), r < 5e-3))
    else:
        record('gemm_8bit_inference', t2)

    # ------------------------------------- gemm_8bit, compute-bound (M=64, not GEMV-like)
    M2, N2, K2, bs2 = 64, 1024, 1024, 64
    A2 = floats(M2 * K2, lambda i: (i % 17) / 17.0 - 0.5)
    Bf2 = floats(N2 * K2, lambda i: (i % 13) / 13.0 - 0.4)
    Bq2 = (C.c_ubyte * (N2 * K2))()
    ab2 = floats(N2 * (K2 // bs2), lambda i: 0.0)
    for row in range(N2):
        lib2.cquantize_blockwise_cpu_fp32(
            CODE, C.cast(C.byref(Bf2, row * K2 * 4), FP),
            C.cast(C.byref(ab2, row * (K2 // bs2) * 4), FP),
            C.cast(C.byref(Bq2, row * K2), U8), bs2, K2)
    out_g2 = floats(M2 * N2, lambda i: 0.0)

    def make_big_gemm(lib):
        return (lambda: lib.cgemm_8bit_inference_cpu_fp32(
            A2, Bq2, ab2, out_g2, M2, N2, K2, K2, K2, N2, bs2)), out_g2

    f2, out2 = make_big_gemm(lib2)
    t2 = timed(f2, 5)
    ref_g2 = list(out_g2)
    if lib5:
        f5, out5 = make_big_gemm(lib5)
        t5 = timed(f5, 5)
        record('gemm_8bit_M64_compute', t2, t5)
        d, r = compare(ref_g2, out5, M2 * N2)
        checks.append(('gemm_8bit_M64 max_abs=%.2e max_rel=%.2e' % (d, r), r < 5e-3))
    else:
        record('gemm_8bit_M64_compute', t2)

    # ------------------------------------------------------------------ gemv_4bit
    Mv, Nv, Kv, bsv = (1, 2048, 2048, 64) if not args.quick else (1, 256, 256, 64)
    Av = floats(Mv * Kv, lambda i: (i % 23) / 8.0 - 1.4)
    Wv = (C.c_ubyte * (Nv * (Kv // 2)))()
    for i in range(Nv * (Kv // 2)):
        Wv[i] = (i * 37 + 11) & 0xFF
    amv = floats(Nv * (Kv // bsv), lambda i: 0.5 + 0.5 * (i % 3))
    out_v = floats(Mv * Nv, lambda i: 0.0)

    def make_gemv(lib):
        return (lambda: lib.cgemv_4bit_inference_cpu_fp32(
            Av, Wv, amv, out_v, Mv, Nv, Kv, Kv, Kv // 2, Nv, bsv, 2)), out_v

    f2, out2 = make_gemv(lib2)
    t2 = timed(f2, 8)
    ref_v = list(out_v)
    if lib5:
        f5, out5 = make_gemv(lib5)
        t5 = timed(f5, 8)
        record('gemv_4bit_nf4', t2, t5)
        d, r = compare(ref_v, out5, Mv * Nv)
        checks.append(('gemv_4bit max_abs=%.2e max_rel=%.2e' % (d, r), r < 5e-3))
    else:
        record('gemv_4bit_nf4', t2)

    # --------------------------------------------------- optimizer adamw 8-bit
    n_opt = 1 << (20 if args.quick else 22)

    def make_opt(lib):
        g = floats(n_opt, lambda i: (i % 61) / 61.0 - 0.5)
        p = floats(n_opt, lambda i: 0.01 * (i % 7))
        s1 = (C.c_ubyte * n_opt)()
        s2 = (C.c_ubyte * n_opt)()
        a1 = floats(n_opt // 256 + 2, lambda i: 0.0)
        a2 = floats(n_opt // 256 + 2, lambda i: 0.0)
        return (lambda: lib.coptimizer_update_8bit_blockwise_cpu(
            0, C.cast(g, C.c_void_p), C.cast(p, C.c_void_p), s1, s2,
            0.9, 0.999, 0.0, 0.0, 1e-8, 1, 1e-3,
            CODE, CODE, a1, a2, 0.0, 1.0, 0, n_opt, 0)), p

    f2, out2 = make_opt(lib2)
    t2 = timed(f2, 8)
    ref_p = list(out2)
    if lib5:
        f5, out5 = make_opt(lib5)
        t5 = timed(f5, 8)
        record('adamw8bit_update_1step', t2, t5)
        d, r = compare(ref_p, out5, 65536, 3)
        checks.append(('adamw8bit params max_rel=%.2e (max_abs=%.2e)' % (r, d), r < 5e-3))
    else:
        record('adamw8bit_update_1step', t2)

    # -------------------------------------------------------------------- report
    print()
    print('=== correctness (AVX-512 build vs AVX2 build, identical inputs) ===')
    if checks:
        for name, ok in checks:
            print('  [%s] %s' % ('PASS' if ok else 'FAIL', name))
    else:
        print('  (skipped: no AVX-512 CPU or no AVX-512 library)')

    print()
    print('=== timing (ms, median; speedup = avx2 / avx512) ===')
    print('  %-26s %10s %10s %9s' % ('kernel', 'avx2', 'avx512', 'speedup'))
    for name, ta, tb in results:
        if tb is None:
            print('  %-26s %10.3f %10s %9s' % (name, ta, '-', '-'))
        else:
            print('  %-26s %10.3f %10.3f %8.2fx'
                  % (name, ta, tb, ta / tb if tb else float('nan')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
