"""Does the 4-bit optimizer's 2.63x speed advantage survive at the scale where it is supposed to matter?

bench_4bit_opt_fixed.py measured 2.63x at 1M parameters. The docs then say, correctly, that the
optimizer is only worth using at 300M+ -- because at 30M it saves 203 MB out of 15.4 GB. But 1M
parameters is 1 MB of state per array, which lives in L2/L3; 300M is 300-600 MB of state, which is
DRAM. Those are different regimes, and the 2.63x number was never measured outside the first one.

So: sweep N over four orders of magnitude with the same kernels, the same interleaved measurement, and
the same estimator, and report the ratio against N. Also report effective bandwidth from the traffic
model, because if both kernels converge to the same GB/s then the effect is a cache artifact and the
honest statement is "memory win, not a speed win, at the scale where it matters".

Traffic per step (read-modify-write counted as 2x):
    grad read 4N + params read 4N + params written 4N
    8-bit: m and v are N bytes each        -> 4N
    4-bit: m and v are packed, N/2 each    -> 2N
    scales: 2 arrays * 4 bytes / BLOCK elements, read + write
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import ctypes
import gc
import os
import statistics as st
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # IDLE: stdout is a StdOutputFile without reconfigure

# The DLL must be loaded BEFORE anything that might import bitsandbytes: on Windows LoadLibrary
# dedups by module basename, and measure_guard's import chain can pin an older libbitsandbytes_cpu.
DLL = r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll'
lib = ctypes.CDLL(DLL)
F = ctypes.c_float
U8P = ctypes.POINTER(ctypes.c_ubyte)
FP = ctypes.POINTER(F)

f8 = lib.coptimizer_update_8bit_blockwise_cpu
f8.restype = None
f8.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, U8P, U8P,
               F, F, F, F, F, ctypes.c_int, F, FP, FP, FP, FP,
               F, F, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
f4 = lib.coptimizer_update_4bit_blockwise_cpu
f4.restype = None
f4.argtypes = list(f8.argtypes)

BLOCK = 256
LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8
OPT_ADAM = 0
ROUNDS = 9
REPS = 5

# 1M reproduces 7.80's number; 384M is a "300M+ class" anchor.
SIZES = [
    (1 << 20, '1M'),
    (1 << 22, '4M'),
    (1 << 24, '16M'),
    (1 << 26, '64M'),
    (1 << 28, '256M'),
    (3 << 27, '384M'),
]


def qmap(levels):
    q = np.zeros(256, np.float32)
    for i in range(256):
        q[i] = (i & (levels - 1)) * (2.0 / (levels - 1)) - 1.0
    return q


def pf(a):
    return a.ctypes.data_as(FP)


def pu(a):
    return a.ctypes.data_as(U8P)


def traffic_bytes(N):
    """Read + write bytes per step, per the state layout (see module docstring)."""
    nb = (N + BLOCK - 1) // BLOCK
    common = 4 * N + 4 * N + 4 * N          # grad read, params read, params write
    scales = 2 * (2 * 4 * nb)               # two absmax arrays, read + written
    t8 = common + 4 * N + scales            # m + v, one byte each
    t4 = common + 2 * N + scales            # m + v, half a byte each
    return t8, t4


def timeit(fn, step, reps):
    fn(step)                                # warm-up: fault the pages first
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(step)
    return (time.perf_counter() - t0) / reps


def bench(N):
    nb = (N + BLOCK - 1) // BLOCK
    rng = np.random.default_rng(0)
    g = rng.normal(0, 1e-3, N).astype(np.float32)
    p = rng.normal(0, 0.02, N).astype(np.float32)
    qm8, qm4 = qmap(256), qmap(16)

    s8a = np.zeros(N, np.uint8)
    s8b = np.zeros(N, np.uint8)
    a8a = np.zeros(nb, np.float32)
    a8b = np.zeros(nb, np.float32)
    s4a = np.zeros((N + 1) // 2, np.uint8)
    s4b = np.zeros((N + 1) // 2, np.uint8)
    a4a = np.zeros(nb, np.float32)
    a4b = np.zeros(nb, np.float32)

    def run8(step):
        f8(OPT_ADAM, g.ctypes.data, p.ctypes.data, pu(s8a), pu(s8b),
           F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
           pf(qm8), pf(qm8), pf(a8a), pf(a8b), F(0.0), F(1.0), 0, N, 0)

    def run4(step):
        f4(OPT_ADAM, g.ctypes.data, p.ctypes.data, pu(s4a), pu(s4b),
           F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
           pf(qm4), pf(qm4), pf(a4a), pf(a4b), F(0.0), F(1.0), 0, N, 0)

    # Interleaved: each round measures the 8-bit arm then the 4-bit arm, so any slow drift in the
    # machine (frequency, background process) hits both arms symmetrically.
    t8, t4 = [], []
    for it in range(ROUNDS):
        t8.append(timeit(run8, it + 1, REPS))
        t4.append(timeit(run4, it + 1, REPS))

    med8, med4 = st.median(t8), st.median(t4)
    ratio = med8 / med4
    if len(t8) > 2:
        rat = [a / b for a, b in zip(t8, t4)]
        m = st.median(rat)
        se = st.pstdev(rat) / np.sqrt(len(rat))
        lo, hi = m - 1.96 * se, m + 1.96 * se
    else:
        m, lo, hi = ratio, float('nan'), float('nan')

    tr8, tr4 = traffic_bytes(N)
    gbs8 = tr8 / med8 / 1e9
    gbs4 = tr4 / med4 / 1e9

    out = dict(N=N, med8=med8, med4=med4, ratio=ratio, med_ratio=m, lo=lo, hi=hi,
               gbs8=gbs8, gbs4=gbs4, bytes8=tr8, bytes4=tr4,
               cv8=st.pstdev(t8) / st.mean(t8), cv4=st.pstdev(t4) / st.mean(t4))

    del g, p, s8a, s8b, a8a, a8b, s4a, s4b, a4a, a4b
    gc.collect()
    return out


def main():
    print()
    print('  === 4-bit vs 8-bit optimizer kernel, swept over parameter count ===')
    print('  python %s · %d rounds x %d reps, interleaved · %d logical cores'
          % (os.path.basename(__file__), ROUNDS, REPS, os.cpu_count()))
    print()
    print('  %-6s %11s %11s %10s %14s %9s %9s %9s'
          % ('N', '8-bit ms', '4-bit ms', 'paired', '95% CI of paired', '8b GB/s', '4b GB/s',
             'med8/med4'))
    print('  ' + '-' * 96)
    rows = []
    for N, name in SIZES:
        r = bench(N)
        rows.append((name, r))
        print('  %-6s %11.3f %11.3f %8.2fx %6.2f~%-7.2f %9.1f %9.1f %9.2f'
              % (name, r['med8'] * 1e3, r['med4'] * 1e3, r['med_ratio'],
                 r['lo'], r['hi'], r['gbs8'], r['gbs4'], r['ratio']))
        sys.stdout.flush()

    print()
    print('  === reading ===')
    first = rows[0]
    last = rows[-1]
    print('  at 1M    : ratio %.2fx (CI %.2f~%.2f) · %.1f vs %.1f GB/s'
          % (first[1]['med_ratio'], first[1]['lo'], first[1]['hi'],
             first[1]['gbs8'], first[1]['gbs4']))
    print('  at %s: ratio %.2fx (CI %.2f~%.2f) · %.1f vs %.1f GB/s'
          % (last[0], last[1]['med_ratio'], last[1]['lo'], last[1]['hi'],
             last[1]['gbs8'], last[1]['gbs4']))
    print('  traffic ratio (8-bit / 4-bit, from the state layout): %.3fx' % (last[1]['bytes8'] / last[1]['bytes4']))
    if last[1]['hi'] < 1.0:
        print('  => 4-bit is SLOWER at the large end: the 2.63x does not scale.')
    elif last[1]['lo'] > 1.0:
        print('  => 4-bit is still faster at the large end, but the margin moved from %.2fx to %.2fx.'
              % (first[1]['med_ratio'], last[1]['med_ratio']))
    else:
        print('  => at the large end the two kernels are indistinguishable (CI includes 1).')


if __name__ == '__main__':
    main()
