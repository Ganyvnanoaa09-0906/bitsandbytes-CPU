"""Pure device AdamW microbench: 33M fp32 params, no preceding GEMM/backward.

Measures the kernel itself (queue.finish after each call), so it is not contaminated by
waiting for earlier backward work. This is the clean baseline P1 should be judged against.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import pyopencl as cl

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_layer as L  # noqa: E402

A = L.A
N = 33_000_000
LOCAL = 256
GLOBAL = ((N + LOCAL - 1) // LOCAL) * LOCAL

KERNEL = r"""
__kernel void adamw_off(__global float* master, __global float* mom, __global float* vel,
                        __global const float* grad, int off, int n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2) {
    int i = get_global_id(0);
    if (i >= n) return;
    int j = off + i;
    float g = grad[i];
    float m = b1 * mom[j] + (1.f - b1) * g;
    float v = b2 * vel[j] + (1.f - b2) * g * g;
    mom[j] = m; vel[j] = v;
    master[j] -= lr * (m / bc1) / (sqrt(v / bc2) + eps) + lr * wd * master[j];
}
"""

prog = cl.Program(A.ctx, KERNEL).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
K = cl.Kernel(prog, 'adamw_off')


def buf(n):
    return cl.Buffer(A.ctx, A.mf.READ_WRITE, n * 4)


master, mom, vel, grad = buf(N), buf(N), buf(N), buf(N)
zero = np.zeros(N, dtype=np.float32)
ones = np.ones(N, dtype=np.float32)
A.cl.enqueue_copy(A.queue, master, ones)
A.cl.enqueue_copy(A.queue, mom, zero)
A.cl.enqueue_copy(A.queue, vel, zero)
A.cl.enqueue_copy(A.queue, grad, zero)
A.queue.finish()


def call():
    K(A.queue, (GLOBAL,), (LOCAL,), master, mom, vel, grad,
      np.int32(0), np.int32(N), np.float32(1e-4), np.float32(0.9),
      np.float32(0.999), np.float32(1e-8), np.float32(0.0),
      np.float32(0.1), np.float32(0.001))


call(); A.queue.finish()
ts = []
for _ in range(6):
    t0 = time.perf_counter(); call(); A.queue.finish(); ts.append(time.perf_counter() - t0)
print('device %s CU=%d' % (A.ctx.devices[0].name, A.ctx.devices[0].max_compute_units))
print('params=%d  pure AdamW: %.2f ms (median %.2f ms)' %
      (N, statistics.median(ts[1:]) * 1e3, statistics.median(ts) * 1e3))
print('raw ms:', [round(v * 1e3, 2) for v in ts])
print('effective traffic: ~%.1f GB/s (read master/mom/vel/grad + write master/mom/vel)' %
      ((N * 4 * 7) / statistics.median(ts[1:]) / 1e9))
