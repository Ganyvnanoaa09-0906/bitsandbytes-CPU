"""probe_igpu_absmax_dequant.py — compute NF4 absmax on the iGPU from raw uint8 codes.

Persistent fp32 absmax (3 MB/layer) is what made the correct hybrid slow. Here the kernel
takes the raw uint8 absmax codes + the nested blockwise state2 (absmax 3072 floats, code
256 floats) and reconstructs the per-64-block absmax on the GPU, then dequantizes NF4.
Validated against bnb's dequantize_4bit.
"""
from __future__ import annotations

import json

import numpy as np
import pyopencl as cl
import torch
from safetensors import safe_open

import bitsandbytes.functional as BF
from bitsandbytes.nn import Linear4bit, Params4bit

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.3.img_mlp.gate_layer.weight'

KERNEL = r"""
__kernel void dequant_nf4_abs(__global const uchar* packed,
                              __global const uchar* abs_codes,
                              __global const float* s2_absmax,
                              __global const float* s2_code,
                              const float offset,
                              __global const float* nf4,
                              __global float* out,
                              const int n, const int blocksize, const int s2blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    int b = i / blocksize;
    int g = b / s2blocksize;
    uchar raw = abs_codes[b];
    float a = s2_absmax[g] * s2_code[raw] + offset;
    uchar v = packed[i >> 1];
    int nib = (i & 1) ? (v & 0xF) : (v >> 4);
    out[i] = a * nf4[nib];
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
    ref = BF.dequantize_4bit(lin.weight.data, qstate).float()
    print('shape', tuple(ref.shape), 's2 code', tuple(qstate.state2.code.shape),
          's2 absmax', tuple(qstate.state2.absmax.shape), flush=True)

    rt = OpenCLFP16.get()
    prog = cl.Program(rt.ctx, KERNEL).build(options=['-cl-mad-enable'])
    kern = cl.Kernel(prog, 'dequant_nf4_abs')
    mf = cl.mem_flags
    n = out_f * in_f

    def buf(arr):
        arr = np.ascontiguousarray(arr)
        b = cl.Buffer(rt.ctx, mf.READ_WRITE, max(4, arr.nbytes))
        cl.enqueue_copy(rt.queue, b, arr, is_blocking=True)
        rt.queue.finish()
        return b

    pb = buf(packed.numpy().reshape(-1))
    cb = buf(qstate.absmax.numpy().reshape(-1).astype(np.uint8))
    s2a = buf(qstate.state2.absmax.numpy().astype(np.float32))
    s2c = buf(qstate.state2.code.numpy().astype(np.float32))
    nf4 = buf(qstate.code.numpy().astype(np.float32))
    ob = cl.Buffer(rt.ctx, mf.READ_WRITE, n * 4)
    kern(rt.queue, (n,), None, pb, cb, s2a, s2c, np.float32(qstate.offset.item()),
         nf4, ob, np.int32(n), np.int32(qstate.blocksize),
         np.int32(qstate.state2.blocksize))
    rt.queue.finish()
    out = np.empty(n, dtype=np.float32)
    cl.enqueue_copy(rt.queue, out, ob, is_blocking=True)
    rt.queue.finish()
    ref_np = ref.reshape(-1).numpy()
    rel = np.linalg.norm(out - ref_np) / (np.linalg.norm(ref_np) + 1e-9)
    print('absmax-on-GPU dequant rel %.6e | max %.3e' % (rel, np.abs(out - ref_np).max()),
          flush=True)

    import time
    t0 = time.perf_counter()
    for _ in range(5):
        kern(rt.queue, (n,), None, pb, cb, s2a, s2c, np.float32(qstate.offset.item()),
             nf4, ob, np.int32(n), np.int32(qstate.blocksize),
             np.int32(qstate.state2.blocksize))
        rt.queue.finish()
    print('dequant time %.1f ms' % ((time.perf_counter() - t0) / 5 * 1000), flush=True)


if __name__ == '__main__':
    main()
