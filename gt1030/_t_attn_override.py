# -*- coding: utf-8 -*-
"""探针 41：Xgemm 覆盖参数到底影不影响 batched GEMM？影响的话能不能单独控制？"""
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A
import gt1030.full_gpu_layer as L

T, HD, BH, M = 1024, 64, 16, 2048
PN = ["GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG",
      "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"]
CFG = dict(GEMMK=0, KREG=4, KWG=16, KWI=2, MDIMA=16, MDIMC=16, MWG=128,
           NDIMB=16, NDIMC=16, NWG=128, SA=1, SB=1, STRM=1, STRN=1, VWM=4, VWN=2)
bl = A.bl
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
DEV = ctypes.c_void_p(int(A.ctx.devices[0].int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(2)
Q = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
              hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
Kb = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
               hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
S = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * T * 2)
Oh = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * HD * 2)
W1 = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
               hostbuf=(rng.standard_normal((M, 512)) * 0.05).astype(np.float16))
W2 = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
               hostbuf=(rng.standard_normal((512, 1024)) * 0.05).astype(np.float16))
W3 = cl.Buffer(A.ctx, mf.READ_WRITE, M * 1024 * 2)


def bench(fn, n=10):
    fn(); A.queue.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    A.queue.finish()
    return (time.time() - t0) / n


def attn_s():
    A.hgemm_batched(Q, Kb, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)


def attn_o():
    A.hgemm_batched(S, Kb, Oh, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)


def gemm():
    L.hgemm_nn(W1, W2, W3, M, 1024, 512)


def report(tag):
    for nm, fn in (('bGEMM_S', attn_s), ('bGEMM_O', attn_o), ('GEMM_Wqkv', gemm)):
        try:
            dt = bench(fn)
            print('  %-10s %7.2f ms  %7.1f GFLOPS' % (nm, dt * 1000, {
                'bGEMM_S': 2 * BH * T * T * HD, 'bGEMM_O': 2 * BH * T * HD * T,
                'GEMM_Wqkv': 2 * M * 1024 * 512}[nm] / dt / 1e9))
        except Exception as e:
            print('  %-10s FAIL %s' % (nm, str(e)[:70]))


print('=== 无覆盖 ===')
report('base')
names = (ctypes.c_char_p * len(PN))(*[x.encode() for x in PN])
vals = (ctypes.c_size_t * len(PN))(*[int(CFG.get(k, 0)) for k in PN])
rc = bl.CLBlastOverrideParameters(DEV, b'Xgemm', 16, len(PN), names, vals)
print('=== Xgemm 覆盖 KREG=4 (rc=%d) ===' % rc)
report('kreg4')
# 再试试只覆盖 batched 用的 kernel 名
for kname in (b'XgemmBatched', b'XgemmDirect'):
    rc = bl.CLBlastOverrideParameters(DEV, kname, 16, len(PN), names, vals)
    print('=== 覆盖 %s rc=%d ===' % (kname.decode(), rc))
    report(kname.decode())
