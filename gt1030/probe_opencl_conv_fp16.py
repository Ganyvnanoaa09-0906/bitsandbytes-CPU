"""probe_opencl_conv_fp16.py — device-resident fp16 im2col + Hgemm conv on the iGPU.

The fp32 resident probe reached only CPU parity (oneDNN ~220-350 GFLOPS vs CLBlast/PCIe
overheads). The iGPU's fp16 peak is ~1.29 TFLOPS, so this probe checks whether an
fp16 im2col + CLBlast Hgemm path can actually beat the CPU conv at SD1.5 shapes.
"""
from __future__ import annotations

import ctypes
import time

import numpy as np
import pyopencl as cl
import torch

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO = 101, 111
HALF_ONE, HALF_ZERO = 0x3C00, 0x0000
mf = cl.mem_flags

IM2COL_H = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void im2col_nchw_h(
    const int C, const int H, const int W,
    const int KH, const int KW, const int pad, const int stride,
    const int Hout, const int Wout,
    __global const half* x, __global half* cols)
{
    int idx = get_global_id(0);
    int L = Hout * Wout;
    int K = C * KH * KW;
    if (idx >= K * L) return;
    int k = idx / L;
    int l = idx % L;
    int c = k / (KH * KW);
    int rem = k % (KH * KW);
    int kh = rem / KW;
    int kw = rem % KW;
    int oh = l / Wout;
    int ow = l % Wout;
    int ih = oh * stride - pad + kh;
    int iw = ow * stride - pad + kw;
    half v = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? x[(c * H + ih) * W + iw] : (half)0.0f;
    cols[idx] = v;
}
"""


def main():
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    bl = ctypes.CDLL(DLL)
    bl.CLBlastHgemm.restype = ctypes.c_int
    bl.CLBlastHgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    prog = cl.Program(ctx, IM2COL_H).build(options=['-cl-mad-enable'])
    kern = cl.Kernel(prog, 'im2col_nchw_h')
    torch.set_num_threads(6)

    for cin, h, w in [(320, 64, 64), (640, 32, 32), (1280, 16, 16), (320, 32, 32)]:
        cout = cin
        K, L = cin * 9, h * w
        x = np.random.randn(1, cin, h, w).astype(np.float32)
        wt = np.random.randn(cout, K).astype(np.float32)
        xh = x.astype(np.float16)
        wh = wt.astype(np.float16)
        x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xh)
        w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wh)
        cols_buf = cl.Buffer(ctx, mf.READ_WRITE, K * L * 2)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, cout * L * 2)
        qp = ctypes.c_void_p(int(queue.int_ptr))

        def run_once():
            kern(queue, (K * L,), None, np.int32(cin), np.int32(h), np.int32(w),
                 np.int32(3), np.int32(3), np.int32(1), np.int32(1),
                 np.int32(h), np.int32(w), x_buf, cols_buf)
            rc = bl.CLBlastHgemm(
                ROW, NO, NO, cout, L, K, HALF_ONE,
                ctypes.c_void_p(int(w_buf.int_ptr)), 0, K,
                ctypes.c_void_p(int(cols_buf.int_ptr)), 0, L,
                HALF_ZERO,
                ctypes.c_void_p(int(out_buf.int_ptr)), 0, L,
                ctypes.byref(qp), None)
            assert rc == 0, rc

        run_once()
        queue.finish()
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
        glof = 2 * cout * cin * 9 * h * w / 1e9
        print('shape=(%d,%d,%d) | CPU %.1f ms (%.0f GF) | fp16 resident GPU %.1f ms '
              '(%.0f GF) | GPU/CPU %.2fx'
              % (cin, h, w, cpu_s * 1e3, glof / cpu_s, gpu_s * 1e3, glof / gpu_s, gpu_s / cpu_s))
        for b in (x_buf, w_buf, cols_buf, out_buf):
            b.release()


if __name__ == '__main__':
    main()
