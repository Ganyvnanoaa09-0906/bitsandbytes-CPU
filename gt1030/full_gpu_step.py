"""full-GPU step skeleton: RMSNorm + SwiGLU MLP block, forward + backward + AdamW, all on the device.

Report 7.118.11 closed the hybrid executor: 153 CPU<->GPU round trips cost 0.3-0.6 s of pure plumbing
per step on a 0.594 s reference, so the only shape that can approach it keeps activations on the iGPU.
This script builds that skeleton for the block that dominates the AR model's FLOPs (~71% of a layer):
RMSNorm -> W1/W3 -> SiLU gate -> W2 -> +x, the full backward, then AdamW with fp32 masters and fp16
copies in both layouts. Everything lives in device buffers; the only traffic is the initial upload
and the downloads used for correctness checks.

GEMMs: CLBlast Hgemm, device buffers only, verified column-major conventions.
Kernels: RMSNorm fwd/bwd (row stats + apply), SiLU gate fwd/bwd, residual add, AdamW.
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
ONE, ZERO = 0x3C00, 0x0000
LAYOUT_COL, NO, TRANS = 0, 111, 112

M, D, FF = 2048, 512, 1368

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))

KERNELS = r'''
#define EPS 1e-6f
__kernel void rms_stats(__global const half* x, __global float* ss, int rows, int d) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float s = 0.f;
    for (int i = 0; i < d; ++i) { float v = (float)x[r*d + i]; s += v*v; }
    ss[r] = s / (float)d;
}
__kernel void rms_apply(__global const half* x, __global const half* w, __global const float* ss,
                        __global half* y, int rows, int d) {
    int i = get_global_id(0);
    if (i >= rows*d) return;
    float inv = rsqrt((float)ss[i / d] + EPS);
    y[i] = (half)((float)x[i] * inv * (float)w[i % d]);
}
__kernel void silu_gate(__global const half* a, __global const half* b, __global half* g, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)a[i];
    g[i] = (half)(x / (1.f + exp(-x)) * (float)b[i]);
}
__kernel void silu_bwd(__global const half* a, __global const half* b, __global const half* dy,
                       __global half* da, __global half* db, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float x = (float)a[i];
    float sg = 1.f / (1.f + exp(-x));
    float d = (float)dy[i];
    da[i] = (half)(d * (sg + x * sg * (1.f - sg)) * (float)b[i]);
    db[i] = (half)(d * x * sg);
}
__kernel void resid_add(__global const half* y, __global const half* x, __global half* out, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    out[i] = (half)((float)y[i] + (float)x[i]);
}
__kernel void rms_bwd_stats(__global const half* x, __global const half* dh, __global const half* w,
                            __global float* ss, __global float* s, int rows, int d) {
    int r = get_global_id(0);
    if (r >= rows) return;
    float a = 0.f, b = 0.f;
    for (int i = 0; i < d; ++i) {
        float xv = (float)x[r*d + i];
        a += xv * xv;
        b += (float)dh[r*d + i] * (float)w[i] * xv;
    }
    ss[r] = a / (float)d;
    s[r] = b;
}
__kernel void rms_bwd_apply(__global const half* x, __global const half* dh, __global const half* w,
                            __global const float* ss, __global const float* s,
                            __global half* dx, int rows, int d) {
    int i = get_global_id(0);
    if (i >= rows*d) return;
    int r = i / d;
    float sig = sqrt((float)ss[r] + EPS);
    float inv = 1.f / sig;
    float dxv = ((float)dh[i] * (float)w[i % d]) * inv;
    dxv -= (float)x[i] * (float)s[r] / ((float)d * sig * sig * sig);
    dx[i] = (half)dxv;
}
__kernel void transpose(__global const half* src, __global half* dst, int rows, int cols) {
    int i = get_global_id(0);
    if (i >= rows*cols) return;
    int r = i / cols, c = i % cols;
    dst[c*rows + r] = src[i];
}
__kernel void adamw(__global float* master, __global float* mom, __global float* vel,
                    __global const half* grad, __global half* w_fwd, __global half* w_bwd,
                    int din, int dout, float lr, float b1, float b2, float eps, float wd,
                    float bc1, float bc2) {
    int idx = get_global_id(0);
    if (idx >= din*dout) return;
    int i = idx / dout, j = idx % dout;
    float g = (float)grad[idx];
    float w = master[idx];
    float mm = b1*mom[idx] + (1.f-b1)*g;
    float vv = b2*vel[idx] + (1.f-b2)*g*g;
    mom[idx] = mm; vel[idx] = vv;
    float wh = w - lr*(mm/bc1)/(sqrt(vv/bc2)+eps) - lr*wd*w;
    master[idx] = wh;
    w_fwd[idx] = (half)wh;
    w_bwd[j*din + i] = (half)wh;
}
'''


def hgemm_dev(a_dev, b_dev, c_dev, m, n, k, beta=0.0):
    """C(m,n) = A(m,k) @ B(k,n) + beta*C, device buffers, verified column-major convention."""
    rc = bl.CLBlastHgemm(
        LAYOUT_COL, NO, NO, n, m, k, ONE,
        ctypes.c_void_p(int(b_dev.int_ptr)), 0, n,
        ctypes.c_void_p(int(a_dev.int_ptr)), 0, k,
        ONE if beta else ZERO,
        ctypes.c_void_p(int(c_dev.int_ptr)), 0, n,
        ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d (%dx%dx%d)' % (rc, m, n, k))


def hgemm_dw_dev(a_dev, b_dev, c_dev, m, n, k):
    """C(k,n) = A(m,k)^T @ B(m,n) -- implemented elsewhere via device transpose + gemm_nn."""
    raise NotImplementedError


def main() -> int:
    rng = np.random.default_rng(0)
    prog = cl.Program(ctx, KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])

    K = {name: cl.Kernel(prog, name) for name in
         ('rms_stats', 'rms_apply', 'silu_gate', 'silu_bwd', 'resid_add',
          'rms_bwd_stats', 'rms_bwd_apply', 'transpose', 'adamw')}

    def buf(nbytes):
        return cl.Buffer(ctx, mf.READ_WRITE, nbytes)

    def upload(arr):
        return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)

    def download(b, shape):
        out = np.empty(shape, dtype=np.float16)
        cl.enqueue_copy(queue, out, b)
        queue.finish()
        return out

    x16 = (rng.standard_normal((M, D)).astype(np.float32) * 0.1).astype(np.float16)
    dout16 = (rng.standard_normal((M, D)).astype(np.float32) * 0.01).astype(np.float16)
    w1 = (rng.standard_normal((D, FF)).astype(np.float32) * 0.02).astype(np.float16)   # (in,out)
    w3 = (rng.standard_normal((D, FF)).astype(np.float32) * 0.02).astype(np.float16)
    w2 = (rng.standard_normal((FF, D)).astype(np.float32) * 0.02).astype(np.float16)
    wnorm = np.ones(D, dtype=np.float16)

    dev = {
        'x': upload(x16), 'dout': upload(dout16), 'wnorm': upload(wnorm),
        'w1f': upload(w1), 'w1b': upload(np.ascontiguousarray(w1.T)),
        'w2f': upload(w2), 'w2b': upload(np.ascontiguousarray(w2.T)),
        'w3f': upload(w3), 'w3b': upload(np.ascontiguousarray(w3.T)),
        'h': buf(M * D * 2), 'a': buf(M * FF * 2), 'b': buf(M * FF * 2), 'g': buf(M * FF * 2),
        'y': buf(M * D * 2), 'out': buf(M * D * 2),
        'dy': buf(M * FF * 2), 'da': buf(M * FF * 2), 'db': buf(M * FF * 2),
        'dh': buf(M * D * 2), 'dx': buf(M * D * 2),
        'dw1': buf(D * FF * 2), 'dw2': buf(FF * D * 2), 'dw3': buf(D * FF * 2),
        'hT': buf(D * M * 2), 'gT': buf(FF * M * 2),
        'ss': buf(M * 4), 's': buf(M * 4),
    }
    # fp32 masters + AdamW state, initialised from the fp16 weights
    for tag, wmat, din, dout in (('1', w1, D, FF), ('3', w3, D, FF), ('2', w2, FF, D)):
        master = np.ascontiguousarray(wmat.astype(np.float32))
        zeros = np.zeros_like(master)
        dev['m' + tag] = upload(master)      # fp32 master
        dev['mo' + tag] = upload(zeros)      # Adam first moment
        dev['v' + tag] = upload(zeros)       # Adam second moment

    def fwd():
        K['rms_stats'](queue, (M,), (64,), dev['x'], dev['ss'], np.int32(M), np.int32(D))
        K['rms_apply'](queue, (M * D,), (256,), dev['x'], dev['wnorm'], dev['ss'], dev['h'],
                       np.int32(M), np.int32(D))
        hgemm_dev(dev['h'], dev['w1f'], dev['a'], M, FF, D)
        hgemm_dev(dev['h'], dev['w3f'], dev['b'], M, FF, D)
        K['silu_gate'](queue, (M * FF,), (256,), dev['a'], dev['b'], dev['g'], np.int32(M * FF))
        hgemm_dev(dev['g'], dev['w2f'], dev['y'], M, D, FF)
        K['resid_add'](queue, (M * D,), (256,), dev['y'], dev['x'], dev['out'], np.int32(M * D))

    def bwd():
        hgemm_dev(dev['dout'], dev['w2b'], dev['dy'], M, FF, D)              # dY = dout @ W2
        K['transpose'](queue, (M * FF,), (256,), dev['g'], dev['gT'],
                       np.int32(M), np.int32(FF))
        hgemm_dev(dev['gT'], dev['dout'], dev['dw2'], FF, D, M)              # dW2 = g^T @ dout
        K['silu_bwd'](queue, (M * FF,), (256,), dev['a'], dev['b'], dev['dy'],
                      dev['da'], dev['db'], np.int32(M * FF))
        K['transpose'](queue, (M * D,), (256,), dev['h'], dev['hT'],
                       np.int32(M), np.int32(D))
        hgemm_dev(dev['hT'], dev['da'], dev['dw1'], D, FF, M)                # dW1 = h^T @ da
        hgemm_dev(dev['hT'], dev['db'], dev['dw3'], D, FF, M)                # dW3 = h^T @ db
        hgemm_dev(dev['da'], dev['w1b'], dev['dh'], M, D, FF)                # dh = da @ W1
        hgemm_dev(dev['db'], dev['w3b'], dev['dh'], M, D, FF, beta=1.0)      # dh += db @ W3
        K['rms_bwd_stats'](queue, (M,), (64,), dev['x'], dev['dh'], dev['wnorm'],
                           dev['ss'], dev['s'], np.int32(M), np.int32(D))
        K['rms_bwd_apply'](queue, (M * D,), (256,), dev['x'], dev['dh'], dev['wnorm'],
                           dev['ss'], dev['s'], dev['dx'], np.int32(M), np.int32(D))

    def adamw(tag, din, dout):
        K['adamw'](queue, (din * dout,), (256,), dev['m' + tag], dev['mo' + tag], dev['v' + tag],
                   dev['dw' + tag], dev['w' + tag + 'f'], dev['w' + tag + 'b'],
                   np.int32(din), np.int32(dout), np.float32(1e-4), np.float32(0.9),
                   np.float32(0.999), np.float32(1e-8), np.float32(0.0),
                   np.float32(1.0), np.float32(1.0))

    # ---- correctness: forward vs torch fp32 on the same weights ----
    fwd()
    queue.finish()
    out = download(dev['out'], (M, D)).astype(np.float32)
    xt = torch.from_numpy(x16.astype(np.float32))
    # w1/w3 are stored (in,out) = (D,FF), w2 is (FF,D); the GPU computes h @ w1 etc.
    w1r = torch.from_numpy(w1.astype(np.float32))
    w3r = torch.from_numpy(w3.astype(np.float32))
    w2r = torch.from_numpy(w2.astype(np.float32))
    h = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6)
    a = h @ w1r
    b = h @ w3r
    g = a * torch.sigmoid(a) * b
    y = g @ w2r + xt
    err = float(np.abs(out - y.numpy()).max())
    print('forward check: max abs err %.3e (relative %.3e)'
          % (err, err / float(np.abs(y.numpy()).max())))

    # ---- correctness: dW2 vs torch ----
    bwd()
    queue.finish()
    dw2 = download(dev['dw2'], (FF, D)).astype(np.float32)
    dout = torch.from_numpy(dout16.astype(np.float32))
    g32 = (a * torch.sigmoid(a) * b).detach()
    dw2_ref = (g32.T @ dout).numpy()
    err2 = float(np.abs(dw2 - dw2_ref).max())
    rel2 = err2 / (float(np.abs(dw2_ref).max()) + 1e-9)
    print('dW2 check: max abs err %.3e (relative %.3e)' % (err2, rel2))

    # ---- timing: fwd + bwd + AdamW, device resident, one finish ----
    # 11 GEMMs per block: 3 forward (W1,W3,W2), 2 dY/dW2, 2 dW1/dW3, 2 dX terms (dh)
    gemm_flops = (2 * M * FF * D * 2 + 2 * M * D * FF          # forward
                  + 2 * M * FF * D + 2 * FF * D * M            # dY, dW2
                  + 2 * D * FF * M * 2                         # dW1, dW3
                  + 2 * M * D * FF * 2)                        # dh from da, db
    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        fwd()
        bwd()
        adamw('1', D, FF)
        adamw('3', D, FF)
        adamw('2', FF, D)
        queue.finish()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times[2:])
    print()
    print('MLP+norm block step (fwd + bwd + AdamW, device-resident): %.2f ms'
          % (med * 1e3))
    print('GEMM FLOPs/block ~ %.1f GFLOP -> effective %.0f GFLOPS'
          % (gemm_flops / 1e9, gemm_flops / med / 1e9))
    print('10-block MLP+norm extrapolation: %.3f s (this block is ~71%% of a layer)'
          % (med * 10 / 0.71))

    # phase breakdown
    def phase(fn, n=6):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            queue.finish()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts[2:]) * 1e3

    tf = phase(fwd)
    tb = phase(bwd)
    to = phase(lambda: (adamw('1', D, FF), adamw('3', D, FF), adamw('2', FF, D)))
    print('phase breakdown: fwd %.2f ms | bwd %.2f ms | AdamW %.2f ms' % (tf, tb, to))
    print('10-block estimate: fwd %.0f ms + bwd %.0f ms + opt %.0f ms = %.0f ms'
          % (tf * 10, tb * 10, to * 10, (tf + tb + to) * 10))
    return 0


if __name__ == '__main__':
    sys.exit(main())
