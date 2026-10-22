"""Direct tiled conv2d forward (k=3,s=1,p=1,groups=1), fp32, vs torch CPU.

Tile: one batch n, output spatial 8x8, BOC=8 output channels per workgroup.
Local: input patch 10x10 x CB=16 channels + weights BOC x CB x 9.
Each thread = one output pixel, accumulates BOC channels.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.hybrid_gpu as G

SRC = r'''
#define TS 8
#define BOC 8
#define CB 16
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void conv2d_direct(__global const float* restrict src,
                   __global const float* restrict wgt,
                   __global float* restrict dst,
                   const int N, const int C, const int H, const int W,
                   const int Cout, const int tiles_x, const int tiles_y) {
    __local float in_t[TS + 2][TS + 2][CB];
    __local float w_t[BOC][CB][9];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * TS + tx;
    const int tile = get_group_id(0);
    const int ocb = get_group_id(1) * BOC;
    const int n = get_group_id(2);
    const int ty0 = tile / tiles_x, tx0 = tile % tiles_x;
    const int oh0 = ty0 * TS, ow0 = tx0 * TS;

    float acc[BOC];
    #pragma unroll
    for (int o = 0; o < BOC; ++o) acc[o] = 0.f;

    for (int c0 = 0; c0 < C; c0 += CB) {
        // load input patch (10x10 x CB)
        for (int idx = tid; idx < (TS + 2) * (TS + 2) * CB; idx += TS * TS) {
            int c = idx % CB;
            int p = idx / CB;
            int py = p / (TS + 2), px = p % (TS + 2);
            int ih = oh0 + py - 1, iw = ow0 + px - 1;
            float v = 0.f;
            if (c0 + c < C && ih >= 0 && ih < H && iw >= 0 && iw < W)
                v = src[((n * C + c0 + c) * H + ih) * W + iw];
            in_t[py][px][c] = v;
        }
        // load weights BOC x CB x 9
        for (int idx = tid; idx < BOC * CB * 9; idx += TS * TS) {
            int k = idx % 9; int q = idx / 9;
            int c = q % CB; int o = q / CB;
            int oc = ocb + o;
            float v = 0.f;
            if (oc < Cout && c0 + c < C)
                v = wgt[((oc * C + c0 + c) * 3 + (k / 3)) * 3 + (k % 3)];
            w_t[o][c][k] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int oh = oh0 + ty, ow = ow0 + tx;
        if (oh < H && ow < W) {
            #pragma unroll
            for (int c = 0; c < CB; ++c) {
                if (c0 + c >= C) break;
                #pragma unroll
                for (int ky = 0; ky < 3; ++ky)
                    #pragma unroll
                    for (int kx = 0; kx < 3; ++kx) {
                        float v = in_t[ty + ky][tx + kx][c];
                        #pragma unroll
                        for (int o = 0; o < BOC; ++o)
                            acc[o] += v * w_t[o][c][ky * 3 + kx];
                    }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int oh = oh0 + ty, ow = ow0 + tx;
    if (oh < H && ow < W) {
        #pragma unroll
        for (int o = 0; o < BOC; ++o) {
            int oc = ocb + o;
            if (oc < Cout) dst[((n * Cout + oc) * H + oh) * W + ow] = acc[o];
        }
    }
}
'''


def bench(N, C, H, W, Cout):
    G.init(); ctx, q = G._ctx, G._q
    mf = cl.mem_flags
    prog = cl.Program(ctx, SRC).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
    kern = cl.Kernel(prog, 'conv2d_direct')
    x = np.random.randn(N, C, H, W).astype(np.float32)
    w = np.random.randn(Cout, C, 3, 3).astype(np.float32)
    out = np.empty((N, Cout, H, W), dtype=np.float32)
    xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    wb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    ob = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
    tiles_x = (W + 7) // 8; tiles_y = (H + 7) // 8

    def run():
        kern(q, (tiles_x * tiles_y * 8, ((Cout + 7) // 8) * 8, N), (8, 8, 1),
             xb, wb, ob, np.int32(N), np.int32(C), np.int32(H), np.int32(W),
             np.int32(Cout), np.int32(tiles_x), np.int32(tiles_y))
    run(); q.finish()
    cl.enqueue_copy(q, out, ob).wait()
    xt = torch.from_numpy(x); wt = torch.from_numpy(w)
    for _ in range(2): ref = F.conv2d(xt, wt, padding=1)
    ts = []
    for _ in range(3):
        t = time.perf_counter(); ref = F.conv2d(xt, wt, padding=1); ts.append(time.perf_counter() - t)
    cpu = min(ts) * 1e3
    ts = []
    for _ in range(3):
        t = time.perf_counter(); run(); q.finish(); ts.append(time.perf_counter() - t)
    gpu = min(ts) * 1e3
    err = float(np.max(np.abs(out - ref.numpy())))
    gf = 2 * N * Cout * H * W * C * 9 / 1e9
    print('N%d C%d H%d W%d out%d: CPU %.2fms iGPU %.2fms speedup %.2fx GF %.1f err %.2e'
          % (N, C, H, W, Cout, cpu, gpu, cpu / gpu, gf, err), flush=True)


if __name__ == '__main__':
    for args in [(16, 640, 16, 16, 640), (16, 1280, 8, 8, 1280), (16, 960, 16, 16, 320)]:
        bench(*args)
