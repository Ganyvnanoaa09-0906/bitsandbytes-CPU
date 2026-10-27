# -*- coding: utf-8 -*-
"""给 AR-V2 的 4 个 GEMM 形状扫 CLBlast 覆盖参数（row-major, fp16, M=2048）。

形状（都属于 C[m,n] = A[m,k] @ B[k,n]，也就是我们的 hgemm_nn）：
    Wqkv 2048 x 1024 x 512
    Wo   2048 x  512 x 512
    W12  2048 x 2736 x 512
    W3   2048 x  512 x 1368
候选配置来自 research/bench_results/clblast_best_configs_1030.json（Qwen 形状上验过的那几组）。
"""
from __future__ import annotations
import ctypes
import itertools
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A

M = 2048
SHAPES = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]

PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]


def base(**kw):
    d = dict(GEMMK=0, KREG=1, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
             NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    d.update(kw)
    return d


CFGS = [
    ('default', None),
    ('B(qkv/o)', base()),
    ('D(kwi1)', base(KWI=1)),
    ('I(mdima32)', base(MDIMA=32)),
    ('kwi1_mdima32', base(KWI=1, MDIMA=32)),
    ('vwm2', base(VWM=2)),
    ('kwg32', base(KWG=32)),
    ('mwg64', base(MWG=64)),
    ('nwg64', base(NWG=64)),
    ('kwi4', base(KWI=4)),
    ('kreg2', base(KREG=2)),
    ('strm0', base(STRM=0, STRN=0)),
]

bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(0)
bufs = {}
for nm, n, k in SHAPES:
    a = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
    bufs[nm] = (cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a),
                cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b),
                cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2), (M, n, k))
print('%-12s %s' % ('config', ' '.join('%s:GFLOPS' % s[0] for s in SHAPES)))
best = {}
for cname, params in CFGS:
    if params is None:
        names = vals = None
    else:
        names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
        vals = (ctypes.c_size_t * len(PN))(*[int(params.get(k, 0)) for k in PN])
    if params is not None:
        rc = bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)
        if rc != 0:
            print('%-12s override rc=%d' % (cname, rc)); continue
    row = []
    for nm, n, k in SHAPES:
        ab, bb, cb, (m_, n_, k_) = bufs[nm]
        import gt1030.full_gpu_layer as L

        def run():
            L.hgemm_nn(ab, bb, cb, m_, n_, k_)

        try:
            run(); A.queue.finish()
        except Exception as e:
            print('%-12s %s FAILED %s' % (cname, nm, str(e)[:60])); row.append(0.0); continue
        t0 = time.time()
        for _ in range(10):
            run()
        A.queue.finish()
        dt = (time.time() - t0) / 10
        gf = 2.0 * n_ * m_ * k_ / dt / 1e9
        row.append(gf)
        if nm not in best or gf > best[nm][0]:
            best[nm] = (gf, cname)
    print('%-12s %s' % (cname, ' '.join('%8.1f' % x for x in row)))
print()
for nm, (gf, cn) in best.items():
    print('best %-5s %.1f GFLOPS  <- %s' % (nm, gf, cn))
