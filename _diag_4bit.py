"""Find the FIRST step where the C kernel and the numpy reference diverge.

Guessing at formulas wasted time; measure instead. One step, print the leading
values from both sides, and diff them. The first divergence localises the cause:
  * diverge on step 1        -> a formula/scale/packing mismatch
  * agree on step 1, drift later -> the state quantisation or absmax rule differs
"""
import ctypes
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
lib = ctypes.CDLL(r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll')
fn = lib.coptimizer_update_4bit_blockwise_cpu
fn.restype = None
fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
               ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte),
               ctypes.c_float] * 0  # placeholder replaced below

F = ctypes.c_float
fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
               ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte),
               F, F, F, F, F, ctypes.c_int, F,
               ctypes.POINTER(F), ctypes.POINTER(F), ctypes.POINTER(F), ctypes.POINTER(F),
               F, F, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]

BLOCK = 256
N = 512                     # 2 blocks, small enough to print
LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8
OPT_ADAM = 0


def pf(a):
    return a.ctypes.data_as(ctypes.POINTER(F))


def make_qmap():
    q = np.zeros(256, np.float32)
    for i in range(256):
        q[i] = (i & 0x0F) * (2.0 / 15.0) - 1.0
    return q


def c_step(p, st1, st2, am1, am2, qmap, g, step):
    fn(OPT_ADAM, g.ctypes.data, p.ctypes.data,
       st1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
       st2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
       F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
       pf(qmap), pf(qmap), pf(am1), pf(am2),
       F(0.0), F(1.0), 0, N, 0)


def q4(x):
    return np.clip(np.rint((x + 1.0) * 7.5).astype(np.int64), 0, 15)


def dq4(c):
    return c * (2.0 / 15.0) - 1.0


def ref_step(p, st1, st2, am1, am2, g, step, fm, fv):
    c1 = 1.0 - B1 ** step
    c2 = np.sqrt(1.0 - B2 ** step)
    step_size = -LR * c2 / c1
    out = p.copy()
    nb = (N + BLOCK - 1) // BLOCK
    for b in range(nb):
        lo, hi = b * BLOCK, min(N, b * BLOCK + BLOCK)
        for i in range(lo, hi):
            bi = i >> 1
            m = (dq4(st1[bi] & 0x0F if i & 1 else st1[bi] >> 4) * am1[b]) if am1[b] > 0 else 0.0
            v = (dq4(st2[bi] & 0x0F if i & 1 else st2[bi] >> 4) * am2[b]) if am2[b] > 0 else 0.0
            gr = float(g[i])
            m = B1 * m + (1 - B1) * gr
            v = B2 * v + (1 - B2) * gr * gr
            out[i] = p[i] + step_size * m / (np.sqrt(v) + c2 * EPS)
            fm[i], fv[i] = m, v
        am1[b] = np.abs(fm[lo:hi]).max()
        am2[b] = np.abs(fv[lo:hi]).max()
        for i in range(lo, hi):
            bi = i >> 1
            cm = int(q4(fm[i] / am1[b])) if am1[b] > 0 else 8
            cv = int(q4(fv[i] / am2[b])) if am2[b] > 0 else 8
            if i & 1:
                st1[bi] = (st1[bi] & 0xF0) | (cm & 0x0F)
                st2[bi] = (st2[bi] & 0xF0) | (cv & 0x0F)
            else:
                st1[bi] = (st1[bi] & 0x0F) | ((cm & 0x0F) << 4)
                st2[bi] = (st2[bi] & 0x0F) | ((cv & 0x0F) << 4)
    return out


rng = np.random.default_rng(1)
p0 = rng.normal(0, 0.02, N).astype(np.float32)
qmap = make_qmap()
nb = (N + BLOCK - 1) // BLOCK

pc = p0.copy(); sc1 = np.zeros((N + 1) // 2, np.uint8); sc2 = np.zeros_like(sc1)
ac1 = np.zeros(nb, np.float32); ac2 = np.zeros(nb, np.float32)
pr = p0.copy().astype(np.float64); sr1 = np.zeros_like(sc1); sr2 = np.zeros_like(sc1)
ar1 = np.zeros(nb, np.float32); ar2 = np.zeros(nb, np.float32)
fm = np.zeros(N); fv = np.zeros(N)

for step in range(1, 4):
    g = rng.normal(0, 1e-3, N).astype(np.float32)
    c_step(pc, sc1, sc2, ac1, ac2, qmap, g, step)
    pr = ref_step(pr, sr1, sr2, ar1, ar2, g, step, fm, fv)
    d = np.abs(pc.astype(np.float64) - pr)
    print('step %d  最大偏差 %.3e' % (step, d.max()))
    print('    C   p[:6] = %s' % np.array2string(pc[:6], precision=8))
    print('    ref p[:6] = %s' % np.array2string(pr[:6], precision=8))
    print('    C   am1 = %s' % np.array2string(ac1, precision=8))
    print('    ref am1 = %s' % np.array2string(ar1, precision=8))
    print('    C   s1[:4] = %s' % sc1[:4])
    print('    ref s1[:4] = %s' % sr1[:4])
    if d.max() > 1e-9:
        i = int(np.argmax(d))
        print('  ★ 第一处分歧在 i=%d: C=%.8e ref=%.8e' % (i, pc[i], pr[i]))
        print('    g[%d]=%.6e   块内 absmax C=%.6e ref=%.6e'
              % (i, g[i], ac1[i // BLOCK], ar1[i // BLOCK]))
        break
