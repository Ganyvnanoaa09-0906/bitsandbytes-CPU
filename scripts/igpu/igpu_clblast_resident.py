"""Device-resident training step with CLBlast, no host round-trips.

History: the first version (igpu_cpu_vs_resident.py) used a hand-written OpenCL GEMM and measured
CPU 6.78 ms vs device 12.94 ms -- the device lost. 7.109 then made CLBlast usable on non-cubic shapes,
and a second version of this benchmark still only won 1.22x on a 2048-wide layer, because the step fell
apart into host glue: download y (47 ms), compute the loss in torch (16 ms), transpose gy in numpy
(32 ms), upload it (4 ms). None of that is the engine.

This version keeps the whole step on the device: y stays in its buffer, a small kernel computes
d = y - target, writes the output gradient AND its transpose in one pass, and reduces sum(d^2) with one
workgroup. The host then only sees a 4-byte loss and the next batch.

Correctness is checked first, as before: same batches, same initial weights, both arms must land on the
same final weights (relative error < 1e-4) before any speed number is read.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import ctypes
import os
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

torch.set_num_threads(6)
torch.set_grad_enabled(False)

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
LAYOUT_COL, TRANSPOSE_NO = 0, 111

_ctx = cl.create_some_context(interactive=False)
_q = cl.CommandQueue(_ctx)
_mf = cl.mem_flags
_qp = ctypes.c_void_p(int(_q.int_ptr))

KER = r'''
__kernel void lossgrad(const int B, const int N,
                       const __global float* restrict y,
                       const __global float* restrict tgt,
                       __global float* restrict gy,
                       __global float* restrict gyT,
                       const float scale) {
    int i = get_global_id(0);
    if (i >= B * N) return;
    float d = y[i] - tgt[i];
    float g = 2.f * d * scale;
    gy[i] = g;
    int b = i / N, n = i - b * N;
    gyT[n * B + b] = g;          /* transposed copy in the same pass: dW needs gy^T */
}

__kernel void reduce2(const int n, __global const float* restrict g,
                      __global float* restrict out) {
    __local float s[256];
    int t = get_local_id(0);
    float acc = 0.f;
    for (int i = t; i < n; i += 256) acc += g[i] * g[i];
    s[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) out[0] = s[0];
}

__kernel void adam2(const int N, const int K,
                    __global float* restrict p, __global float* restrict pT,
                    __global float* restrict m, __global float* restrict v,
                    const __global float* restrict g,
                    const float lr, const float b1, const float b2,
                    const float c1, const float c2, const float eps) {
    int i = get_global_id(0);
    if (i >= N * K) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    int n = i / K, k = i - n * K;
    p[i] -= lr * (mi / c1) / (sqrt(vi / c2) + eps);
    pT[k * N + n] = p[i];
}
'''
_prog = cl.Program(_ctx, KER).build()
k_lg = cl.Kernel(_prog, 'lossgrad')
k_red = cl.Kernel(_prog, 'reduce2')
k_adam = cl.Kernel(_prog, 'adam2')

STAT = {'to_dev': 0, 'to_host': 0}


def buf(n):
    return cl.Buffer(_ctx, _mf.READ_WRITE, max(int(n) * 4, 4))


def up(b, arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    cl.enqueue_copy(_q, b, a).wait()
    STAT['to_dev'] += a.nbytes


def down(b, shape):
    out = np.empty(int(np.prod(shape)), np.float32)
    cl.enqueue_copy(_q, out, b).wait()
    STAT['to_host'] += out.nbytes
    return out.reshape(shape)


def gemm(m, n, k, A, B, C):
    rc = bl.CLBlastSgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO,
        n, m, k, ctypes.c_float(1.0),
        ctypes.c_void_p(int(B.int_ptr)), 0, n,
        ctypes.c_void_p(int(A.int_ptr)), 0, k,
        ctypes.c_float(0.0),
        ctypes.c_void_p(int(C.int_ptr)), 0, n,
        ctypes.byref(_qp), None)
    if rc != 0:
        raise RuntimeError('CLBlast rc=%d on %s' % (rc, (m, n, k)))
    _q.finish()


def one_layer(N, K, B, steps=12, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    torch.manual_seed(0)
    W_true = torch.randn(N, K) * 0.1
    Xt = torch.randn(4096, K)
    Yt = Xt @ W_true.t()
    W0 = torch.randn(N, K) * 0.05
    batches = [torch.randint(0, Xt.shape[0], (B,)) for _ in range(steps)]

    print()
    print('=' * 96)
    print('  layer %dx%d · batch %d · %d params (%.2f MB)'
          % (N, K, B, N * K, N * K * 4 / 1e6))

    # ---------------- CPU arm ----------------
    pc, mc, vc = W0.clone(), torch.zeros(N, K), torch.zeros(N, K)
    ct, losses_c = [], []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = batches[step - 1]
        xb, yb = Xt[idx], Yt[idx]
        y = xb @ pc.t()
        g = 2.0 * (y - yb) / (B * N)
        dW = g.t() @ xb
        mc = b1 * mc + (1 - b1) * dW
        vc = b2 * vc + (1 - b2) * dW * dW
        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
        pc = pc - lr * (mc / c1) / ((vc / c2).sqrt() + eps)
        ct.append(time.perf_counter() - t0)
        losses_c.append(float(((y - yb) ** 2).mean()))
    cpu_ms = st.median(ct) * 1e3

    # ---------------- device arm ----------------
    w, wT = buf(N * K), buf(K * N)
    mt, vt, dWb = buf(N * K), buf(N * K), buf(N * K)
    xa, ya, ta = buf(B * K), buf(B * N), buf(B * N)
    gya, gyaT = buf(B * N), buf(N * B)
    lossb = buf(1)
    up(w, W0.numpy())
    up(wT, np.ascontiguousarray(W0.numpy().T))
    up(mt, np.zeros((N, K)))
    up(vt, np.zeros((N, K)))
    _q.finish()

    # warm iGPU clocks
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 15:
        gemm(N, K, B, gyaT, xa, dWb)

    STAT['to_dev'] = STAT['to_host'] = 0
    dt, losses_d = [], []
    ph = {k: [] for k in ('up_x', 'up_tgt', 'gemm_fwd', 'lossgrad', 'reduce', 'gemm_dW', 'adam')}

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = batches[step - 1]
        xb, yb = Xt[idx], Yt[idx]
        t = time.perf_counter(); up(xa, xb.numpy()); ph['up_x'].append(time.perf_counter() - t)
        t = time.perf_counter(); up(ta, yb.numpy()); ph['up_tgt'].append(time.perf_counter() - t)
        t = time.perf_counter(); gemm(B, N, K, xa, wT, ya); ph['gemm_fwd'].append(time.perf_counter() - t)
        t = time.perf_counter()
        k_lg(_q, ((B * N + 255) // 256 * 256,), None,
             np.int32(B), np.int32(N), ya, ta, gya, gyaT, np.float32(1.0 / (B * N)))
        ph['lossgrad'].append(time.perf_counter() - t)
        k_red(_q, (256,), (256,), np.int32(B * N), gya, lossb)
        _q.finish()
        t = time.perf_counter()
        loss = down(lossb, (1,))[0] * (B * N) / 4.0   # sum(gy^2) * (B*N)/4 = mean((y-t)^2)
        ph['reduce'].append(time.perf_counter() - t)
        losses_d.append(float(loss))
        t = time.perf_counter(); gemm(N, K, B, gyaT, xa, dWb); ph['gemm_dW'].append(time.perf_counter() - t)
        c1, c2 = 1 - b1 ** step, 1 - b2 ** step
        t = time.perf_counter()
        k_adam(_q, ((N * K + 255) // 256 * 256,), None,
               np.int32(N), np.int32(K), w, wT, mt, vt, dWb,
               np.float32(lr), np.float32(b1), np.float32(b2),
               np.float32(c1), np.float32(c2), np.float32(eps))
        _q.finish()
        ph['adam'].append(time.perf_counter() - t)
        dt.append(time.perf_counter() - t0)
    dev_ms = st.median(dt) * 1e3

    wd = down(w, (N, K))
    rel = float(np.abs(wd - pc.numpy()).max() / np.abs(pc.numpy()).max())
    tr = (STAT['to_dev'] + STAT['to_host']) / steps / 1e6

    print('  CPU    %8.2f ms/步' % cpu_ms)
    print('  设备   %8.2f ms/步   (CLBlast + 常驻 Adam + 设备侧 loss/grad)' % dev_ms)
    print('  ⇒ %s' % ('★ 设备快 %.2f×' % (cpu_ms / dev_ms) if dev_ms < cpu_ms
                      else 'CPU 快 %.2f×' % (dev_ms / cpu_ms)))
    print('  逐位对拍: 末步 loss  CPU %.8f  设备 %.8f；最终权重相对误差 %.2e %s'
          % (losses_c[-1], losses_d[-1], rel, '✓' if rel < 1e-4 else '✗'))
    print('  主机流量 %.1f MB/步（只有 x 与 target 两个 O(B) 块）' % tr)
    print('  分项(中位 ms): ' + ' · '.join('%s %.1f' % (k, st.median(v) * 1e3)
                                          for k, v in ph.items() if v))
    return cpu_ms, dev_ms


print('device: %s | %d CU' % (_ctx.devices[0].name, _ctx.devices[0].max_compute_units))
res = []
for N, K, B in ((2048, 2048, 2048), (1024, 1024, 2048), (512, 512, 1024)):
    res.append((N, K, B) + one_layer(N, K, B))

print()
print('=' * 96)
print('  %-16s %10s %10s %10s  %s' % ('layer', 'CPU ms', '设备 ms', '设备/CPU', 'GT 1030 推算'))
# GT 1030: 2 GEMMs of 2*B*N*K FLOPs at 631 GFLOPS (1127 peak * 0.56), plus the elementwise pass at
# 48 GB/s (read y, read target, write gy, write gyT), plus a negligible Adam. Arithmetic only.
for N, K, B, c, d in res:
    gemm_s = 2 * (2.0 * N * K * B) / 631e9
    elem_s = (4.0 * B * N * 4) / 48e9
    print('  %-16s %10.2f %10.2f %9.2fx  %.1f ms'
          % ('%dx%d b=%d' % (N, K, B), c, d, c / d, (gemm_s + elem_s) * 1e3))
