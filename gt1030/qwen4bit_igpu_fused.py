"""qwen4bit_igpu_fused.py — fused NF4 dequant+GEMM hybrid linears for Qwen-Image.

Each Linear4bit is replaced by GPU4BitLinearFused:
  * packed NF4 bytes stay resident (3.4 GB total),
  * the per-64-block absmax is reconstructed inside the GEMM kernel from the raw uint8
    absmax codes + the nested blockwise state2 (small persistent buffers), so there is
    NO 3 MB/layer absmax upload and NO 28.5 GB/forward fp32 weight materialization,
  * the GEMM kernel is the 16x16 tile / one-output-per-thread/fp32-accumulate variant
    that reached 110 GF on gfx902 (M=64: 58.6 ms vs CPU 140 ms),
  * fp32 output goes back to the CPU model; attention/RoPE/norm stay on the CPU.

Intended for small M (<=128). For larger M the scheduler should route to the CPU/fp32 path.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyopencl as cl
import torch

from bitsandbytes.nn import Linear4bit

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

KERNEL = r"""
#define T 16
__kernel void fused_nf4_gemm(__global const float* A,
                             __global const uchar* Bp,
                             __global const uchar* abs_codes,
                             __global const float* s2_absmax,
                             __global const float* s2_code,
                             const float offset,
                             __global const float* nf4,
                             __global float* C,
                             const int M, const int N, const int K)
{
    __local float As[T][T + 1];
    __local float Bs[T][T + 1];
    __local float lcode[16];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * T + ty;
    int col = get_group_id(1) * T + tx;
    if (tx == 0 && ty < 16) lcode[ty] = nf4[ty];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M) ? A[row * K + k0 + tx] : 0.0f;
        if (col < N) {
            int idx = col * K + (k0 + ty);
            uchar b = Bp[idx >> 1];
            int nib = (idx & 1) ? (b & 0xF) : (b >> 4);
            int blk = idx >> 6;              // blocksize 64
            float a = s2_absmax[blk >> 8] * s2_code[abs_codes[blk]] + offset;
            Bs[ty][tx] = a * lcode[nib];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < T; ++kk)
            acc += As[ty][kk] * Bs[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"""


class _RT:
    inst = None

    @classmethod
    def get(cls):
        if cls.inst is None:
            cls.inst = cls()
        return cls.inst

    def __init__(self):
        self.rt = OpenCLFP16.get()
        self.prog = cl.Program(self.rt.ctx, KERNEL).build(options=['-cl-mad-enable'])
        self.kern = cl.Kernel(self.prog, 'fused_nf4_gemm')
        self.xbuf = None
        self.xcap = 0
        self.obuf = None
        self.ocap = 0

    def upload(self, arr):
        arr = np.ascontiguousarray(arr)
        b = cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(4, arr.nbytes))
        cl.enqueue_copy(self.rt.queue, b, arr, is_blocking=True)
        self.rt.queue.finish()
        return b

    def f32(self, numel):
        return cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(4, numel * 4))


def _set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class GPU4BitLinearFused(torch.nn.Module):
    def __init__(self, lin: Linear4bit):
        super().__init__()
        self.r = _RT.get()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        qs = lin.weight.quant_state
        self.quant_state = qs
        self.out_f = self.out_features
        self.in_f = self.in_features
        self.packed = self.r.upload(lin.weight.data.reshape(-1).numpy())
        self.abs_codes = self.r.upload(qs.absmax.numpy().reshape(-1).astype(np.uint8))
        self.s2_abs = self.r.upload(qs.state2.absmax.numpy().astype(np.float32))
        self.s2_code = self.r.upload(qs.state2.code.numpy().astype(np.float32))
        self.offset = np.float32(qs.offset.item())
        self.code = self.r.upload(qs.code.numpy().astype(np.float32))
        # shared scratch (filled lazily on the runtime)

    def forward(self, x: torch.Tensor):
        r = self.r
        org = x.shape
        rows = int(np.prod(org[:-1])) if len(org) > 1 else 1
        K, N = self.in_f, self.out_f
        if r.xcap < rows * K:
            if r.xbuf is not None:
                r.xbuf.release()
            r.xbuf = r.f32(rows * K)
            r.xcap = rows * K
        if r.ocap < rows * N:
            if r.obuf is not None:
                r.obuf.release()
            r.obuf = r.f32(rows * N)
            r.ocap = rows * N
        x_np = np.ascontiguousarray(x.detach().reshape(rows, K).numpy().astype(np.float32))
        cl.enqueue_copy(r.rt.queue, r.xbuf, x_np, is_blocking=True)
        gsz = ((rows + 15) // 16 * 16, (N + 15) // 16 * 16)
        r.kern(r.rt.queue, gsz, (16, 16), r.xbuf, self.packed, self.abs_codes,
               self.s2_abs, self.s2_code, self.offset, self.code, r.obuf,
               np.int32(rows), np.int32(N), np.int32(K))
        out = np.empty((rows, N), dtype=np.float32)
        cl.enqueue_copy(r.rt.queue, out, r.obuf, is_blocking=True)
        r.rt.queue.finish()
        return torch.from_numpy(out).reshape(*org[:-1], N)


def to_igpu_fused(model, verbose=True):
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Linear4bit):
            _set_submodule(model, name, GPU4BitLinearFused(mod))
            n += 1
    if verbose:
        print('wrapped %d linears for fused iGPU' % n, flush=True)
    return n
