"""probe_opencl_conv_resident.py — device-resident im2col+GEMM conv microbenchmark.

Per-op hybrid conv loses to CPU oneDNN because of im2col/transfer overhead. This probe
keeps weight/columns/output buffers on the iGPU and runs an OpenCL im2col kernel +
CLBlast SGEMM without host round-trips, to see whether a device-resident conv engine
could beat the CPU at representative SD1.5 shapes.
"""
from __future__ import annotations

import ctypes
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO = 101, 111
mf = cl.mem_flags

IM2COL = r"""
__kernel void im2col_nchw(
    const int C, const int H, const int W,
    const int KH, const int KW, const int pad, const int stride,
    const int Hout, const int Wout,
    __global const float* x, __global float* cols)
{
    int idx = get_global_id(0);
    int L = Hout * Wout;
    int K = C * KH * KW;
    if (idx >= K * L) return;
    int k = idx / L;              // (c, kh, kw)
    int l = idx % L;              // (oh, ow)
    int c = k / (KH * KW);
    int rem = k % (KH * KW);
    int kh = rem / KW;
    int kw = rem % KW;
    int oh = l / Wout;
    int ow = l % Wout;
    int ih = oh * stride - pad + kh;
    int iw = ow * stride - pad + kw;
    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = x[(c * H + ih) * W + iw];
    cols[idx] = v;
}
"""


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    bl = ctypes.CDLL(DLL)
    bl.CLBlastSgemm.restype = ctypes.c_int
    bl.CLBlastSgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    prog = cl.Program(ctx, IM2COL).build(options=['-cl-mad-enable'])
    torch.set_num_threads(6)

    for cin, h, w in [(320, 64, 64), (640, 32, 32), (1280, 16, 16),
                      (320, 32, 32), (640, 16, 16), (1280, 8, 8)]:
        cout = cin
        kh = kw = 3
        pad, stride = 1, 1
        hout = wout = h
        K = cin * kh * kw
        L = hout * wout
        x = np.random.randn(1, cin, h, w).astype(np.float32)
        wt = np.random.randn(cout, K).astype(np.float32)
        x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
        w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wt)
        cols_buf = cl.Buffer(ctx, mf.READ_WRITE, K * L * 4)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, cout * L * 4)
        qp = ctypes.c_void_p(int(queue.int_ptr))

        def run_once():
            prog.im2col_nchw(queue, (K * L,), None,
                             np.int32(cin), np.int32(h), np.int32(w),
                             np.int32(kh), np.int32(kw), np.int32(pad), np.int32(stride),
                             np.int32(hout), np.int32(wout), x_buf, cols_buf)
            rc = bl.CLBlastSgemm(
                ROW, NO, NO, cout, L, K, 1.0,
                ctypes.c_void_p(int(w_buf.int_ptr)), 0, K,
                ctypes.c_void_p(int(cols_buf.int_ptr)), 0, L,
                0.0,
                ctypes.c_void_p(int(out_buf.int_ptr)), 0, L,
                ctypes.byref(qp), None)
            assert rc == 0, rc

        run_once()
        queue.finish()  # warmup/compile
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            run_once()
        queue.finish()
        gpu_s = (time.perf_counter() - t0) / iters

        conv = torch.nn.Conv2d(cin, cout, 3, padding=1).eval()
        xt = torch.from_numpy(x)
        with torch.no_grad():
            for _ in range(2):
                conv(xt)
            t0 = time.perf_counter()
            for _ in range(iters):
                y = conv(xt)
            cpu_s = (time.perf_counter() - t0) / iters
        gflops = 2 * cout * cin * kh * kw * h * w / 1e9
        print('shape=(%d,%d,%d) | CPU %.1f ms (%.0f GF) | resident GPU %.1f ms (%.0f GF) '
              '| GPU/CPU %.2fx' % (cin, h, w, cpu_s * 1e3, gflops / cpu_s,
                                   gpu_s * 1e3, gflops / gpu_s, gpu_s / cpu_s))
        for b in (x_buf, w_buf, cols_buf, out_buf):
            b.release()


if __name__ == '__main__':
    main()
