"""override_clblast_params.py -- can tuning parameters make non-cubic shapes launch?

Established so far:

  * CLBlast is correct -- it returns B@A for this call convention, verified against
    a distinct-entry diagonal and confirmed at 128..2048 on non-symmetric operands
    (against B@A the difference is 3e-4..2.9, i.e. accumulation; against A@B it is
    1e3..1e4).
  * Every non-cubic shape fails to launch with -1011/-1015/-1016, independent of
    size: 6x4x5 fails as surely as 256x256x128.
  * Thirty-two combinations of layout, transposition and leading dimension were
    swept and none changed that -- so the call side is not the variable.
  * What was never varied is the tuning database entry.

That last point is what this tests. pyopencl reports the device name as "gfx902",
which matches neither "AMD Radeon(TM) Graphics" nor "AMD Radeon(TM) RX Vega 10
Graphics" in the database's gfx902 section, so the lookup falls through to
kDeviceNameDefault: { 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4 } for names
{ GEMMK, KREG, KWG, KWI, MDIMA, MDIMC, MWG, NDIMB, NDIMC, NWG, SA, SB, STRM, STRN,
VWM, VWN }.

CLBlast exposes CLBlastOverrideParameters for exactly this (C signature, so callable
from ctypes): it takes the device, kernel name, precision, a parameter count and
name/value arrays, and recompiles on the next call. CLBlastPrecisionSingle = 32.

Candidates are ordered by how likely they are to matter:
  1. GEMMK=1, the "indirect" kernel, which exists for shapes the standard kernel
     cannot handle -- and the database specifies 0.
  2. The two named entries in the gfx902 section, which are Vega-family APUs and
     therefore closer to this part than the generic default.
  3. Shrinking the tile, in case a resource limit is being crossed.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
bl.CLBlastOverrideParameters.restype = ctypes.c_int
bl.CLBlastOverrideParameters.argtypes = [
    ctypes.c_void_p,                      # device
    ctypes.c_char_p,                      # kernel name
    ctypes.c_int,                         # precision
    ctypes.c_size_t,                      # num parameters
    ctypes.POINTER(ctypes.c_char_p),      # names
    ctypes.POINTER(ctypes.c_size_t),      # values
]
import pyopencl as cl  # noqa: E402

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
ROW, NO = 1, 111

NAMES = [b'GEMMK', b'KREG', b'KWG', b'KWI', b'MDIMA', b'MDIMC', b'MWG',
         b'NDIMB', b'NDIMC', b'NWG', b'SA', b'SB', b'STRM', b'STRN', b'VWM', b'VWN']


def override(values, kernel=b'Xgemm', prec=32):
    na = (ctypes.c_char_p * len(NAMES))(*NAMES)
    va = (ctypes.c_size_t * len(values))(*[int(v) for v in values])
    rc = bl.CLBlastOverrideParameters(ctypes.c_void_p(int(dev.int_ptr)), kernel,
                                      prec, len(NAMES), na, va)
    return rc


DEFAULT = (0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4)
CANDIDATES = [
    ('数据库默认（gfx902 → kDeviceNameDefault）', DEFAULT),
    ('GEMMK=1 indirect', (1, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4)),
    ('"AMD Radeon(TM) Graphics" 条目', (0, 1, 16, 2, 32, 32, 128, 16, 8, 128, 1, 1, 1, 1, 2, 4)),
    ('"RX Vega 10" 条目', (0, 1, 32, 2, 8, 8, 128, 16, 32, 128, 1, 1, 1, 1, 4, 2)),
    ('小 tile MWG=NWG=32, VWM=VWN=2', (0, 1, 16, 2, 8, 8, 32, 8, 8, 32, 0, 0, 0, 0, 2, 2)),
    ('SA=SB=1 本地暂存', (0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4)),
    ('KREG=2', (0, 2, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4)),
    ('KWI=1', (0, 1, 32, 1, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 4, 4)),
]

SHAPES = [(128, 128, 128), (256, 256, 128), (256, 512, 256), (512, 1024, 512)]


def try_shape(m, n, k):
    A = (np.arange(m * k, dtype=np.float32).reshape(m, k) % 61)
    B = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 53)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    cl.enqueue_copy(queue, bc, np.zeros((m, n), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(ROW, NO, NO, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, k,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    queue.finish()
    out = None
    if rc == 0:
        out = np.zeros((m, n), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
    del ba, bb, bc
    return rc, out


print('设备名: %s | %d CU' % (dev.name, dev.max_compute_units))
print('形状列表（前三个是非立方）: %s' % SHAPES)
print()
for label, vals in CANDIDATES:
    orc = override(vals)
    res = []
    for (m, n, k) in SHAPES:
        rc, out = try_shape(m, n, k)
        if rc != 0:
            res.append('%dx%dx%d ✗%d' % (m, n, k, rc))
        else:
            res.append('%dx%dx%d ✓' % (m, n, k))
    print('  %-42s override=%d  %s' % (label[:42], orc, '  '.join(res)))
