# -*- coding: utf-8 -*-
"""第二轮：围绕 KREG=2 精扫 + 测 batched attention GEMM 是否也受影响。"""
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
T, HD, BH = 1024, 64, 16
SHAPES = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]
PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def base(**kw):
    d = dict(GEMMK=0, KREG=2, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
             NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    d.update(kw)
    return d


CFGS = [
    ('kreg2(base)', base()),
    ('kreg2_kwi1', base(KWI=1)),
    ('kreg2_kwi4', base(KWI=4)),
    ('kreg2_mdima32', base(MDIMA=32)),
    ('kreg2_kwi1_mdima32', base(KWI=1, MDIMA=32)),
    ('kreg4', base(KREG=4)),
    ('kreg2_kwg32', base(KWG=32)),
    ('kreg2_vwm2', base(VWM=2)),
    ('kreg2_mwg256', base(MWG=256)),
    ('kreg2_ndim16_mwg256', base(MWG=256, NWG=256)),
]

bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(1)
bufs = {}
for nm, n, k in SHAPES:
    bufs[nm] = (cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=(rng.standard_normal((M, k)) * 0.05).astype(np.float16)),
                cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=(rng.standard_normal((k, n)) * 0.05).astype(np.float16)),
                cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2), (M, n, k))
# attention buffers
Q = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
              hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
Kb = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
               hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
S = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * T * 2)
O = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * HD * 2)


def bench(fn, n=10):
    fn(); A.queue.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    A.queue.finish()
    return (time.time() - t0) / n


print('%-22s %8s %8s %8s %8s %10s %10s' % ('config', 'Wqkv', 'Wo', 'W12', 'W3',
                                            'bGEMM_S', 'bGEMM_O'))
for cname, params in CFGS:
    names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
    vals = (ctypes.c_size_t * len(PN))(*[int(params.get(k, 0)) for k in PN])
    rc = bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)
    if rc != 0:
        print('%-22s override rc=%d' % (cname, rc)); continue
    row = []
    ok = True
    for nm, n, k in SHAPES:
        ab, bb, cb, (m_, n_, k_) = bufs[nm]
        try:
            dt = bench(lambda: L.hgemm_nn(ab, bb, cb, m_, n_, k_))
        except Exception as e:
            ok = False
            print('%-22s %s FAIL %s' % (cname, nm, str(e)[:50]))
            break
        row.append(2.0 * m_ * n_ * k_ / dt / 1e9)
    if not ok:
        continue
    try:
        dt = bench(lambda: A.hgemm_batched(Q, Kb, S, T, T, HD, A.NO, A.YES,
                                           HD, HD, T, T * HD, T * HD, T * T))
        g_s = 2.0 * BH * T * T * HD / dt / 1e9
        dt = bench(lambda: A.hgemm_batched(S, Kb, O, T, HD, T, A.NO, A.NO,
                                           T, HD, HD, T * T, T * HD, T * HD))
        g_o = 2.0 * BH * T * HD * T / dt / 1e9
    except Exception as e:
        g_s = g_o = float('nan')
    print('%-22s %8.1f %8.1f %8.1f %8.1f %10.1f %10.1f'
          % (cname, row[0], row[1], row[2], row[3], g_s, g_o))
