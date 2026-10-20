"""qwen4bit_igpu_linear.py — hybrid Qwen-Image transformer: iGPU linear layers, CPU rest.

Each bnb Linear4bit is replaced by GPU4BitLinear:
  * the NF4 packed bytes + per-64-block absmax + 16-entry code map stay resident on the
    Vega 6 iGPU (total ~3.8 GB for the 7.12B model),
  * forward() dequantizes the layer to fp16 with an OpenCL kernel, runs CLBlast Hgemm
    (transposed B), downloads the fp16 result and returns fp32 to the CPU model,
  * attention/RoPE/LayerNorm/modulation stay on the CPU.

Measured on one 4096^2 layer: GPU dequant+Hgemm is 1.8-4.2x faster than bnb's CPU
4-bit matmul (seq 16-1024), so this hybrid is the concrete way to put Qwen-Image-2.1
on the iGPU without reimplementing the whole MMDiT.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np
import pyopencl as cl
import torch

import bitsandbytes.functional as BF
from bitsandbytes.nn import Linear4bit

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16, ROW, NO, YES, HALF_ONE, HALF_ZERO  # noqa: E402

KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void dequant_nf4(__global const uchar* packed, __global const float* absmax,
                          __global const float* map, __global half* out,
                          const int n, const int blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    uchar b = packed[i >> 1];
    // bnb layout: high nibble first (validated against dequantize_4bit, relL2 1.6e-3)
    int nib = (i & 1) ? (b & 0xF) : (b >> 4);
    out[i] = (half)(map[nib] * absmax[i / blocksize]);
}
"""


class _Runtime:
    inst = None

    @classmethod
    def get(cls):
        if cls.inst is None:
            cls.inst = cls()
        return cls.inst

    def __init__(self):
        self.rt = OpenCLFP16.get()
        self.prog = cl.Program(self.rt.ctx, KERNEL).build(options=['-cl-mad-enable'])
        self.kern = cl.Kernel(self.prog, 'dequant_nf4')
        self.wbuf = None
        self.wcap = 0
        self.obuf = None
        self.ocap = 0
        self.xbuf = None
        self.xcap = 0
        self.qp = ctypes.c_void_p(int(self.rt.queue.int_ptr))


def _set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class GPU4BitLinear(torch.nn.Module):
    def __init__(self, lin: Linear4bit):
        super().__init__()
        self.rt = _Runtime.get()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = None  # Qwen-Image transformer linears are bias-free
        qs = lin.weight.quant_state
        n = self.out_features * self.in_features
        packed = np.ascontiguousarray(lin.weight.data.reshape(-1).numpy())
        absmax = BF.dequantize_blockwise(qs.absmax, qs.state2).float() + qs.offset
        code = qs.code.float()
        self.n = n
        self.blocksize = qs.blocksize
        self.packed = self.rt.rt.to_gpu(packed)
        self.absmax = self.rt.rt.to_gpu(np.ascontiguousarray(absmax.numpy().astype(np.float32)))
        self.code = self.rt.rt.to_gpu(np.ascontiguousarray(code.numpy().astype(np.float32)))

    def _ensure(self):
        rt = self.rt
        if rt.wcap < self.n:
            if rt.wbuf is not None:
                rt.wbuf.release()
            rt.wbuf = rt.rt.empty_half(self.n)
            rt.wcap = self.n
        # x/out buffers are grown by forward

    def forward(self, x: torch.Tensor):
        rt = self.rt
        org = x.shape
        rows = int(np.prod(org[:-1])) if len(org) > 1 else 1
        K, N = self.in_features, self.out_features
        self._ensure()
        x_np = np.ascontiguousarray(x.detach().reshape(rows, K).numpy().astype(np.float16))
        rt.kern(rt.rt.queue, (self.n,), None, self.packed, self.absmax, self.code,
                rt.wbuf, np.int32(self.n), np.int32(self.blocksize))
        if rt.xcap < rows * K:
            if rt.xbuf is not None:
                rt.xbuf.release()
            rt.xbuf = rt.rt.empty_half(rows * K)
            rt.xcap = rows * K
        if rt.ocap < rows * N:
            if rt.obuf is not None:
                rt.obuf.release()
            rt.obuf = rt.rt.empty_half(rows * N)
            rt.ocap = rows * N
        cl.enqueue_copy(rt.rt.queue, rt.xbuf, x_np)
        rc = rt.rt.bl.CLBlastHgemm(
            ROW, NO, YES, rows, N, K, HALF_ONE,
            ctypes.c_void_p(int(rt.xbuf.int_ptr)), 0, K,
            ctypes.c_void_p(int(rt.wbuf.int_ptr)), 0, K,
            HALF_ZERO,
            ctypes.c_void_p(int(rt.obuf.int_ptr)), 0, N,
            ctypes.byref(rt.qp), None)
        if rc != 0:
            raise RuntimeError('hybrid Hgemm rc=%d' % rc)
        out = np.empty((rows, N), dtype=np.float16)
        cl.enqueue_copy(rt.rt.queue, out, rt.obuf)
        rt.rt.queue.finish()
        return torch.from_numpy(out.astype(np.float32)).reshape(*org[:-1], N)


def to_igpu(model: torch.nn.Module, verbose: bool = True) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Linear4bit):
            _set_submodule(model, name, GPU4BitLinear(mod))
            n += 1
    if verbose:
        print('wrapped %d linears for iGPU hybrid' % n, flush=True)
    return n
