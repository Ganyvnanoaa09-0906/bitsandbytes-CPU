"""probe_qwen_igpu_dequant.py — OpenCL NF4 dequant + Hgemm for one Qwen-Image layer.

CPU bnb dequantizes each layer on every forward (0.19 s for 4096^2), which kills the
hybrid path.  This probe moves the NF4 dequant itself to the iGPU: packed bytes + the
per-64-block absmax (computed once on the CPU) stay resident; an OpenCL kernel expands
them to fp16, then CLBlast Hgemm runs.  We compare against bnb's CPU 4bit matmul.
"""
from __future__ import annotations

import ctypes
import json
import time

import numpy as np
import pyopencl as cl
import torch

import bitsandbytes as bnb
import bitsandbytes.functional as BF
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16, ROW, NO, YES, HALF_ONE, HALF_ZERO  # noqa: E402

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.0.attn.to_q.weight'

KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void dequant4(__global const uchar* packed, __global const float* absmax,
                       __global const float* map, __global half* out,
                       const int n, const int blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    uchar b = packed[i >> 1];
    int nib = (i & 1) ? (b >> 4) : (b & 0xF);
    float a = absmax[i / blocksize];
    out[i] = (half)(map[nib] * a);
}
"""

# variant with the opposite nibble order, to identify the packing convention
KERNEL_SWAP = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void dequant4s(__global const uchar* packed, __global const float* absmax,
                        __global const float* map, __global half* out,
                        const int n, const int blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    uchar b = packed[i >> 1];
    int nib = (i & 1) ? (b & 0xF) : (b >> 4);
    float a = absmax[i / blocksize];
    out[i] = (half)(map[nib] * a);
}
"""


def load_packed(name):
    idx = json.load(open(P + r'\diffusion_pytorch_model.safetensors.index.json'))
    shard = idx['weight_map'][name]
    qs = {}
    with safe_open(P + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name).clone()
        for suffix in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                       'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suffix
            if k in keys:
                qs[suffix] = f.get_tensor(k).clone()
    return packed, qs


def main():
    torch.set_num_threads(6)
    packed, qs = load_packed(NAME)
    meta = json.load(open(P + r'\quant_meta.json'))['layers'][NAME]
    out_f, in_f = meta['shape']
    blocksize = 64
    # CPU bnb path for reference
    lin = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float32,
                     compress_statistics=True, quant_type='nf4')
    p = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu')
    lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False, device='cpu',
                                              module=lin)
    qstate = p.quant_state
    # per-block absmax (nested blockwise dequant)
    absmax = BF.dequantize_blockwise(qstate.absmax, qstate.state2)
    absmax = absmax.float() + qstate.offset
    code = qstate.code.float()  # NF4 code map (16 values)
    print('packed', tuple(packed.shape), '| blocks', tuple(absmax.shape),
          '| code', code.tolist()[:4], '...', flush=True)
    ref = BF.dequantize_4bit(packed.reshape(-1, 1), qstate).float().reshape(out_f, in_f)

    rt = OpenCLFP16.get()
    prog = cl.Program(rt.ctx, KERNEL + KERNEL_SWAP)
    prog.build(options=['-cl-mad-enable'])
    print('kernel names', prog.kernel_names, flush=True)
    k1 = cl.Kernel(prog, 'dequant4')
    k2 = cl.Kernel(prog, 'dequant4s')
    mf = cl.mem_flags
    n = out_f * in_f
    packed_np = np.ascontiguousarray(packed.numpy().reshape(-1))
    absmax_np = np.ascontiguousarray(absmax.numpy().astype(np.float32))
    code_np = np.ascontiguousarray(code.numpy().astype(np.float32))
    pb = rt.to_gpu(packed_np)
    ab = rt.to_gpu(absmax_np)
    cb = rt.to_gpu(code_np)
    outb = rt.empty_half(n)
    for name, kern in (('normal', k1), ('swap', k2)):
        kern(rt.queue, (n,), None, pb, ab, cb, outb, np.int32(n), np.int32(blocksize))
        rt.queue.finish()
        got = rt.to_host(outb, (out_f, in_f)).astype(np.float32)
        rel = np.linalg.norm(got - ref.numpy()) / (np.linalg.norm(ref.numpy()) + 1e-9)
        print('%s nibble order: relL2 %.5f' % (name, rel), flush=True)
        if rel < 1e-3:
            break

    # benchmark dequant + transposed Hgemm for several seq lengths
    wb = rt.empty_half(n)  # fp16 weights (out,in)
    qp = ctypes.c_void_p(int(rt.queue.int_ptr))
    for seq in (16, 64, 256, 1024):
        x = torch.randn(seq, in_f)
        xb = rt.to_gpu(x.numpy().astype(np.float16))
        outmat = rt.empty_half(seq * out_f)
        # warmup
        for _ in range(2):
            kern(rt.queue, (n,), None, pb, ab, cb, wb, np.int32(n), np.int32(blocksize))
            ctypes.CDLL
            rc = rt.bl.CLBlastHgemm(
                ROW, NO, YES, seq, out_f, in_f, HALF_ONE,
                ctypes.c_void_p(int(xb.int_ptr)), 0, in_f,
                ctypes.c_void_p(int(wb.int_ptr)), 0, in_f,
                HALF_ZERO,
                ctypes.c_void_p(int(outmat.int_ptr)), 0, out_f,
                ctypes.byref(qp), None)
            rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(5):
            kern(rt.queue, (n,), None, pb, ab, cb, wb, np.int32(n), np.int32(blocksize))
            rc = rt.bl.CLBlastHgemm(
                ROW, NO, YES, seq, out_f, in_f, HALF_ONE,
                ctypes.c_void_p(int(xb.int_ptr)), 0, in_f,
                ctypes.c_void_p(int(wb.int_ptr)), 0, in_f,
                HALF_ZERO,
                ctypes.c_void_p(int(outmat.int_ptr)), 0, out_f,
                ctypes.byref(qp), None)
            rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 5
        # CPU bnb reference
        xx = x.reshape(1, seq, in_f)
        with torch.no_grad():
            lin(xx)
            t0 = time.perf_counter()
            for _ in range(3):
                lin(xx)
            cpu_s = (time.perf_counter() - t0) / 3
        print('seq=%4d | CPU bnb %.3f s | GPU dequant+gemm %.3f s | speedup %.2fx'
              % (seq, cpu_s, gpu_s, cpu_s / gpu_s), flush=True)
        xb.release(); outmat.release()


if __name__ == '__main__':
    main()
