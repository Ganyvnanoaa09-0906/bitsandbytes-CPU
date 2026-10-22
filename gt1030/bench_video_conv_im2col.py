"""Conv2d forward prototype: iGPU im2col + CLBlast Sgemm, vs torch CPU fp32.

Top AnimateDiff conv shapes: C=1280, k=3, p=1, stride=1, N=B*T=16 frames,
spatial 16x16 (mid) or 32x32 (up blocks). Weight layout: torch (Cout,Cin,k,k).
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.hybrid_gpu as G

IM2COL = r'''
__kernel void im2col_f32(__global const float* src, __global float* dst,
                         int N, int C, int H, int W, int K, int S, int P, int Ho, int Wo) {
    int idx = get_global_id(0);
    int rows = N * Ho * Wo;
    int cols = C * K * K;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    int n = r / (Ho * Wo);
    int ohw = r % (Ho * Wo);
    int oh = ohw / Wo, ow = ohw % Wo;
    int ch = c / (K * K);
    int kk = c % (K * K);
    int kh = kk / K, kw = kk % K;
    int ih = oh * S - P + kh, iw = ow * S - P + kw;
    float v = 0.f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = src[((n * C + ch) * H + ih) * W + iw];
    dst[idx] = v;
}
'''


def main():
    G.init()
    ctx, q = G._ctx, G._q
    mf = cl.mem_flags
    prog = cl.Program(ctx, IM2COL).build(options=['-cl-mad-enable'])
    kern = cl.Kernel(prog, 'im2col_f32')
    bl = G._bl
    qp = ctypes.c_void_p(int(q.int_ptr))

    for (H,W,C,Cout) in ((16,16,640,640),(8,8,1280,1280),(16,16,960,320)):
        N,K,S,P = 16,3,1,1
        Ho = (H + 2*P - K)//S + 1; Wo = (W + 2*P - K)//S + 1
        x = np.random.randn(N,C,H,W).astype(np.float32)
        wt = np.random.randn(Cout,C,K,K).astype(np.float32)
        wmat = np.ascontiguousarray(wt.reshape(Cout, C*K*K).T)  # (Cin*k*k, Cout)
        rows = N*Ho*Wo; cols = C*K*K
        col = np.empty((rows, cols), dtype=np.float32)
        out = np.empty((rows, Cout), dtype=np.float32)

        xb = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=x)
        wb = cl.Buffer(ctx, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wmat)
        cb = cl.Buffer(ctx, mf.WRITE_ONLY, rows*cols*4)
        ob = cl.Buffer(ctx, mf.WRITE_ONLY, rows*Cout*4)

        def run():
            kern(q, (rows*cols,), (256,), xb, cb, np.int32(N),np.int32(C),np.int32(H),np.int32(W),
                 np.int32(K),np.int32(S),np.int32(P),np.int32(Ho),np.int32(Wo))
            rc = bl.CLBlastSgemm(101,111,111, rows, Cout, cols, 1.0,
                                 ctypes.c_void_p(int(cb.int_ptr)),0,cols,
                                 ctypes.c_void_p(int(wb.int_ptr)),0,Cout,0.0,
                                 ctypes.c_void_p(int(ob.int_ptr)),0,Cout,
                                 ctypes.byref(qp),None)
            if rc: raise RuntimeError(rc)
        run(); q.finish()
        cl.enqueue_copy(q, out, ob).wait()
        # CPU reference
        xt = torch.from_numpy(x); wtt = torch.from_numpy(wt)
        for _ in range(2): ref = F.conv2d(xt,wtt,padding=P)
        ts=[]
        for _ in range(3):
            t=time.perf_counter(); ref=F.conv2d(xt,wtt,padding=P); ts.append(time.perf_counter()-t)
        cpu=min(ts)*1e3
        ts=[]
        for _ in range(3):
            t=time.perf_counter(); run(); q.finish(); ts.append(time.perf_counter()-t)
        gpu=min(ts)*1e3
        got = out.reshape(N,Ho,Wo,Cout).transpose(0,3,1,2)
        err=float(np.max(np.abs(got-ref.numpy())))
        gf=2*N*Cout*Ho*Wo*C*K*K/1e9
        print('H=%d C=%d out%d: CPU %.2fms  iGPU %.2fms  speedup %.2fx  GF %.1f  err %.2e'
              % (H,C,Cout,cpu,gpu,cpu/gpu,gf,err), flush=True)


if __name__ == '__main__':
    main()
