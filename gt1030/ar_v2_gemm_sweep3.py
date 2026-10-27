# -*- coding: utf-8 -*-
"""第三轮：KREG=4 附近再扫一轮（KREG 8/16、MWG/NWG、KWI），找 GEMM 的最优点。"""
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
SHAPES = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]
PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def base(**kw):
    d = dict(GEMMK=0, KREG=4, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
             NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    d.update(kw)
    return d


CFGS = [
    ('kreg4', base()),
    ('kreg8', base(KREG=8)),
    ('kreg16', base(KREG=16)),
    ('kreg4_kwi1', base(KWI=1)),
    ('kreg4_mwg256', base(MWG=256, NWG=256)),
    ('kreg4_mwg128_nwg64', base(NWG=64)),
    ('kreg4_mdima32', base(MDIMA=32)),
    ('kreg8_kwi1', base(KREG=8, KWI=1)),
    ('kreg8_mwg256', base(KREG=8, MWG=256, NWG=256)),
    ('kreg4_vwm2', base(VWM=2)),
]
bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(3)
bufs = {}
for nm, n, k in SHAPES:
    bufs[nm] = (cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=(rng.standard_normal((M, k)) * 0.05).astype(np.float16)),
                cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=(rng.standard_normal((k, n)) * 0.05).astype(np.float16)),
                cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2), (M, n, k))


def bench(fn, n=10):
    fn(); A.queue.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    A.queue.finish()
    return (time.time() - t0) / n


print('%-22s %8s %8s %8s %8s %9s' % ('config', 'Wqkv', 'Wo', 'W12', 'W3', 'sum_ms'))
for cname, params in CFGS:
    names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
    vals = (ctypes.c_size_t * len(PN))(*[int(params.get(k, 0)) for k in PN])
    rc = bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)
    if rc != 0:
        print('%-22s rc=%d' % (cname, rc)); continue
    row, tms, ok = [], 0.0, True
    for nm, n, k in SHAPES:
        ab, bb, cb, (m_, n_, k_) = bufs[nm]
        try:
            dt = bench(lambda: L.hgemm_nn(ab, bb, cb, m_, n_, k_))
        except Exception as e:
            ok = False
            print('%-22s %s FAIL %s' % (cname, nm, str(e)[:40]))
            break
        row.append(2.0 * m_ * n_ * k_ / dt / 1e9)
        tms += dt * 1000
    if ok:
        print('%-22s %8.1f %8.1f %8.1f %8.1f %9.2f' % (cname, row[0], row[1], row[2], row[3], tms))
