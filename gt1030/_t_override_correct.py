# -*- coding: utf-8 -*-
"""探针 42：KREG 覆盖参数下的 GEMM 正确性（不能只看速度！loss 变成 ln(1024) 说明算错了）。"""
import ctypes
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A
import gt1030.full_gpu_layer as L

PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]
bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(7)


def setc(kreg):
    vals_d = dict(GEMMK=0, KREG=kreg, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
                  NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
    names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
    vals = (ctypes.c_size_t * len(PN))(*[int(vals_d[k]) for k in PN])
    return bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)


M = 2048
shapes = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]
for kreg in (1, 4, 8, 16, 32):
    rc = setc(kreg)
    print('--- KREG=%d (rc=%d) ---' % (kreg, rc))
    for nm, n, k in shapes:
        a = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
        ab = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
        bb = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
        cb = cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2)
        try:
            L.hgemm_nn(ab, bb, cb, M, n, k)
            A.queue.finish()
        except Exception as e:
            print('  %-5s FAIL %s' % (nm, str(e)[:50])); continue
        out = np.empty((M, n), np.float16)
        cl.enqueue_copy(A.queue, out, cb)
        A.queue.finish()
        ref = a.astype(np.float32) @ b.astype(np.float32)
        rel = float(np.abs(out.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-12))
        print('  %-5s rel=%.3e %s' % (nm, rel, 'OK' if rel < 2e-2 else '<<< WRONG'))
