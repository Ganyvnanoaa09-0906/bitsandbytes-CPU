"""muon_ns_microbench.py — how expensive is Muon's Newton-Schulz on the iGPU vs the CPU?

The real 30.50M AR model puts 60 matrices through Muon (four families):
    (512,512) x20   (256,512) x20   (2736,512) x10   (512,1368) x10
(NS runs on min(out,in) x max(out,in), transposing when out > in).

Per matrix NS5: X <- X/(||X||+eps); 5x [ A = X X^T; B = bA + cAA; X <- aX + B X ]
Each iteration is 3 GEMMs + 1 scale + 1 copy; this script times exactly that on:
  * the iGPU: fp32 CLBlast Sgemm + one small pyopencl kernel, one set of scratch buffers per
    unique shape (what a device-resident optimizer would do),
  * the CPU: torch fp32, 6 threads.
It also reports the conservative variant (queue.finish per matrix) vs one sync at the end.
Random, normalised data: this is a feasibility probe, not an implementation of Muon.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112
A_, B_, C_ = 3.4445, -4.7750, 2.0315
NS_STEPS = 5

# (out, in), count -- the actual Muon matrices of the 30.50M image-AR model
FAMILIES = [((512, 512), 20), ((256, 512), 20), ((2736, 512), 10), ((512, 1368), 10)]
SHAPES = [(o, i) for (o, i), c in FAMILIES for _ in range(c)]

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
dev = ctx.devices[0]
print('device: %s (%d CU, global %.1f GB)' % (dev.name, dev.max_compute_units,
                                              dev.global_mem_size / 1e9))
print('NS5 problems: %d (four families: %s)' % (len(SHAPES), FAMILIES))

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

SCALE_SRC = r'''
__kernel void scale(__global float* x, const float a) {
    const size_t i = get_global_id(0);
    x[i] *= a;
}
'''
prog = cl.Program(ctx, SCALE_SRC).build()
k_scale = cl.Kernel(prog, 'scale')


def sgemm(a, b, c, m, n, k, tb=False, alpha=1.0, beta=0.0):
    lda = k
    ldb = k if tb else n
    rc = bl.CLBlastSgemm(ROW, NO, YES if tb else NO, m, n, k, ctypes.c_float(alpha),
                         ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_float(beta),
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d %dx%dx%d' % (rc, m, n, k))


class Scratch:
    """Scratch buffers for one NS shape (m <= n); X is reset from a normalised template."""

    def __init__(self, m, n):
        self.m, self.n = m, n
        x = np.random.randn(m, n).astype(np.float32) * 0.02
        x /= np.linalg.norm(x)
        self.X = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
        self.template = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
        self.X2 = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
        self.A = cl.Buffer(ctx, mf.READ_WRITE, m * m * 4)
        self.Bm = cl.Buffer(ctx, mf.READ_WRITE, m * m * 4)

    def reset(self):
        cl.enqueue_copy(queue, self.X, self.template)

    def ns5(self, steps=NS_STEPS):
        m, n = self.m, self.n
        for _ in range(steps):
            sgemm(self.X, self.X, self.A, m, m, n, tb=True)          # A = X X^T
            cl.enqueue_copy(queue, self.Bm, self.A)
            k_scale(queue, (m * m,), None, self.Bm, np.float32(B_))  # Bm = b A
            sgemm(self.A, self.A, self.Bm, m, m, m, alpha=C_, beta=1.0)  # Bm += c A A
            cl.enqueue_copy(queue, self.X2, self.X)
            k_scale(queue, (m * n,), None, self.X2, np.float32(A_))  # X2 = a X
            sgemm(self.Bm, self.X, self.X2, m, n, m, alpha=1.0, beta=1.0)  # X2 = B X + X2
            self.X, self.X2 = self.X2, self.X


scratch = {}
for (o, i) in SHAPES:
    key = (min(o, i), max(o, i))
    if key not in scratch:
        scratch[key] = Scratch(*key)

# warm up every shape (one NS step) and one full NS5
for sc in scratch.values():
    sc.reset()
    sc.ns5(steps=1)
queue.finish()
for sc in scratch.values():
    sc.reset()
    sc.ns5()
queue.finish()
print('warm-up done')


def flops(out, inp):
    m, n = min(out, inp), max(out, inp)
    return NS_STEPS * (4.0 * m * m * n + 2.0 * m * m * m)


total_flops = sum(flops(o, i) for (o, i) in SHAPES) / 1e9
print('total work: %.1f GFLOP' % total_flops)

# (a) conservative: sync once per matrix
times = []
for (o, i) in SHAPES:
    sc = scratch[(min(o, i), max(o, i))]
    t0 = time.perf_counter()
    sc.reset()
    sc.ns5()
    queue.finish()
    times.append((time.perf_counter() - t0) * 1000)
print('\n(a) sync per matrix: median %.2f ms, total %.3f s' %
      (statistics.median(times), sum(times) / 1000))

# (b) one sync at the end (device-resident optimizer step)
t0 = time.perf_counter()
for (o, i) in SHAPES:
    sc = scratch[(min(o, i), max(o, i))]
    sc.reset()
    sc.ns5()
t_issue = time.perf_counter() - t0
queue.finish()
t_total = time.perf_counter() - t0
print('(b) one sync at end: issue %.3f s, wall %.3f s' % (t_issue, t_total))

# CPU reference
torch.set_num_threads(6)
mats = []
for (o, i), c in FAMILIES:
    m, n = min(o, i), max(o, i)
    for _ in range(c):
        x = torch.randn(m, n)
        x = x / (x.norm() + 1e-7)
        mats.append(x)


def ns5_torch(x):
    for _ in range(NS_STEPS):
        a = x @ x.T
        bm = B_ * a + C_ * (a @ a)
        x = A_ * x + bm @ x
    return x


t0 = time.perf_counter()
for x in mats:
    ns5_torch(x)
cpu_total = time.perf_counter() - t0
print('\n(c) CPU torch 6 threads: %.3f s (%.1f GFLOPS)' % (cpu_total, total_flops / cpu_total))
print('    iGPU fp32 one-sync: %.3f s (%.1f GFLOPS), iGPU/CPU %.2fx' %
      (t_total, total_flops / t_total, cpu_total / t_total))

# report 7.31: Muon step 6.42 s of which NS ~2.42 s; AdamW step 3.95 s
print('\nprojected Muon step with iGPU NS: %.2f s -> ~%.2f s (CPU Muon 6.42 s, '
      'AdamW baseline 3.95 s)' % (2.42, 6.42 - 2.42 + t_total))
