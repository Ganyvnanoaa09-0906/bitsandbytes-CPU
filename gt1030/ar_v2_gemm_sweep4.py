# -*- coding: utf-8 -*-
"""第四轮：KREG 16/32，并把反向用的 tn/nt 形状也测一遍（确认同参数不回退）。"""
from __future__ import annotations
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A
import gt1030.full_gpu_layer as L

M = 2048
PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def cfg(**kw):
    d = dict(GEMMK=0, KREG=16, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
             NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    d.update(kw)
    return d


bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(4)


def mk(shape):
    return cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                     hostbuf=(rng.standard_normal(shape) * 0.05).astype(np.float16))


def bench(fn, n=10):
    fn(); A.queue.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    A.queue.finish()
    return (time.time() - t0) / n


# 前向 nn 形状
nn = {}
for nm, n, k in (('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)):
    nn[nm] = (mk((M, k)), mk((k, n)), cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2), (M, n, k))
# 反向 tn: C(m,n) = A^T B，A 存 (k,m)，B 存 (k,n)
tn = {
    'dW3(FF,MD)': (mk((M, 1368)), mk((M, 512)), cl.Buffer(A.ctx, mf.READ_WRITE, 1368 * 512 * 2),
                   (1368, 512, M)),
    'dW12(MD,2FF)': (mk((M, 512)), mk((M, 2736)),
                     cl.Buffer(A.ctx, mf.READ_WRITE, 512 * 2736 * 2), (512, 2736, M)),
    'dWo(MD,MD)': (mk((M, 512)), mk((M, 512)),
                   cl.Buffer(A.ctx, mf.READ_WRITE, 512 * 512 * 2), (512, 512, M)),
}
# 反向 nt: C(m,n) = A B^T，A (m,k)，B (n,k)
nt = {
    'dg(M,FF)': (mk((M, 512)), mk((1368, 512)), cl.Buffer(A.ctx, mf.READ_WRITE, M * 1368 * 2),
                 (M, 1368, 512)),
    'dh2(M,MD)': (mk((M, 2736)), mk((512, 2736)), cl.Buffer(A.ctx, mf.READ_WRITE, M * 512 * 2),
                  (M, 512, 2736)),
}
for cname, params in (('kreg16', cfg()), ('kreg32', cfg(KREG=32))):
    names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
    vals = (ctypes.c_size_t * len(PN))(*[int(params.get(k, 0)) for k in PN])
    rc = bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)
    print('=== %s (rc=%d) ===' % (cname, rc))
    tot = 0.0
    for nm, n, k in (('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)):
        a, b, c, (m_, n_, k_) = nn[nm]
        try:
            dt = bench(lambda: L.hgemm_nn(a, b, c, m_, n_, k_))
            print('  nn %-6s %7.2f ms  %6.1f GFLOPS' % (nm, dt * 1000,
                                                        2.0 * m_ * n_ * k_ / dt / 1e9))
            tot += dt * 1000
        except Exception as e:
            print('  nn %-6s FAIL %s' % (nm, str(e)[:40]))
    for nm, (a, b, c, (m_, n_, k_)) in tn.items():
        try:
            dt = bench(lambda: L.hgemm_tn(a, b, c, m_, n_, k_))
            print('  tn %-12s %7.2f ms  %6.1f GFLOPS' % (nm, dt * 1000,
                                                         2.0 * m_ * n_ * k_ / dt / 1e9))
            tot += dt * 1000
        except Exception as e:
            print('  tn %-12s FAIL %s' % (nm, str(e)[:40]))
    for nm, (a, b, c, (m_, n_, k_)) in nt.items():
        try:
            dt = bench(lambda: L.hgemm_nt(a, b, c, m_, n_, k_))
            print('  nt %-12s %7.2f ms  %6.1f GFLOPS' % (nm, dt * 1000,
                                                         2.0 * m_ * n_ * k_ / dt / 1e9))
            tot += dt * 1000
        except Exception as e:
            print('  nt %-12s FAIL %s' % (nm, str(e)[:40]))
    print('  total %.2f ms/层' % tot)
