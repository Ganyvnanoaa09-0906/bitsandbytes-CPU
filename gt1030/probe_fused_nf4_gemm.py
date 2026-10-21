"""probe_fused_nf4_gemm.py — fused NF4 dequant + GEMM (no fp32 weight materialization).

The correct fp32 hybrid spends ~28.5 GB of shared DDR4 writes per forward materializing
all 7.12B weights as fp32, which is why it cannot beat the CPU. This prototype dequantizes
NF4 inside the GEMM tile loop instead: packed weights + absmax are read, products are
accumulated in fp32, and only C is written.

Kernel: 32x32 output tile, 16x16 threads, each thread accumulates 2x2 outputs, K-tile 32.
A and B tiles are loaded with coalesced global accesses; B is dequantized on the fly.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyopencl as cl
import torch
from safetensors import safe_open

from bitsandbytes.nn import Linear4bit, Params4bit

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.3.img_mlp.gate_layer.weight'

KERNEL = r"""
#define T 16
__kernel void fused_nf4_gemm(__global const float* A,
                             __global const uchar* Bp,
                             __global const float* absmax,
                             __global const float* code,
                             __global float* C,
                             const int M, const int N, const int K,
                             const int blocksize)
{
    __local float As[T][T + 1];
    __local float Bs[T][T + 1];
    __local float lcode[16];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(0) * T + ty;
    int col = get_group_id(1) * T + tx;
    if (tx == 0 && ty < 16) lcode[ty] = code[ty];
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M) ? A[row * K + k0 + tx] : 0.0f;
        if (col < N) {
            int idx = col * K + (k0 + ty);
            uchar b = Bp[idx >> 1];
            int nib = (idx & 1) ? (b & 0xF) : (b >> 4);
            Bs[ty][tx] = absmax[idx >> 6] * lcode[nib];
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


def main():
    name = NAME
    idx = json.load(open(P + r'\diffusion_pytorch_model.safetensors.index.json'))
    shard = idx['weight_map'][name]
    qs = {}
    with safe_open(P + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name).clone()
        for suf in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                    'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suf
            if k in keys:
                qs[suf] = f.get_tensor(k).clone()
    meta = json.load(open(P + r'\quant_meta.json'))['layers'][name]
    out_f, in_f = meta['shape']
    lin = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float32,
                     compress_statistics=True, quant_type='nf4')
    lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu',
                                              module=lin)
    qstate = lin.weight.quant_state
    import bitsandbytes.functional as BF
    absmax = BF.dequantize_blockwise(qstate.absmax, qstate.state2).float() + qstate.offset
    ref_w = BF.dequantize_4bit(lin.weight.data, qstate).float()

    rt = OpenCLFP16.get()
    prog = cl.Program(rt.ctx, KERNEL).build(options=['-cl-mad-enable'])
    kern = cl.Kernel(prog, 'fused_nf4_gemm')
    mf = cl.mem_flags

    def buf(a):
        a = np.ascontiguousarray(a)
        b = cl.Buffer(rt.ctx, mf.READ_WRITE, max(4, a.nbytes))
        cl.enqueue_copy(rt.queue, b, a, is_blocking=True)
        rt.queue.finish()
        return b

    pb = buf(lin.weight.data.reshape(-1).numpy())
    ab = buf(absmax.numpy().astype(np.float32))
    cb = buf(qstate.code.numpy().astype(np.float32))

    for M in (64, 256):
        x = torch.randn(M, in_f)
        with torch.no_grad():
            y_cpu = x @ ref_w.t()
        x_np = np.ascontiguousarray(x.numpy().astype(np.float32))
        xb = buf(x_np)
        cb_out = cl.Buffer(rt.ctx, mf.READ_WRITE, M * out_f * 4)
        gsz = ((M + 15) // 16 * 16, (out_f + 15) // 16 * 16)
        kern(rt.queue, gsz, (16, 16), xb, pb, ab, cb, cb_out,
             np.int32(M), np.int32(out_f), np.int32(in_f),
             np.int32(qstate.blocksize))
        rt.queue.finish()
        out = np.empty((M, out_f), dtype=np.float32)
        cl.enqueue_copy(rt.queue, out, cb_out, is_blocking=True)
        rt.queue.finish()
        rel = np.linalg.norm(out - y_cpu.numpy()) / (np.linalg.norm(y_cpu.numpy()) + 1e-9)
        t0 = time.perf_counter()
        for _ in range(3):
            kern(rt.queue, gsz, (16, 16), xb, pb, ab, cb, cb_out,
                 np.int32(M), np.int32(out_f), np.int32(in_f),
                 np.int32(qstate.blocksize))
            rt.queue.finish()
        dt = (time.perf_counter() - t0) / 3
        print('M=%3d | fused rel %.3e | %.1f ms | %.1f GFLOP/s'
              % (M, rel, dt * 1e3, 2 * M * out_f * in_f / dt / 1e9), flush=True)


if __name__ == '__main__':
    main()
