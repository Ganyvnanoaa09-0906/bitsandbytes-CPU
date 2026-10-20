"""qwen4bit_igpu_linear_fp32.py — fp32-precision hybrid linears for Qwen-Image 4bit.

The fp16 variant was ~9x faster but per-layer 0.9% error compounded to garbage (whole
forward rel 2.9). This variant dequantizes NF4 to float32 and uses CLBlast Sgemm with
fp32 inputs/outputs, so the CPU model sees numerically the same values it would get from
bnb's CPU Linear4bit path. The iGPU is still fast: fp32 GEMM ~644 GFLOPS peak.
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
from opencl_fp16_ops import OpenCLFP16, ROW, NO, YES  # noqa: E402

KERNEL = r"""
__kernel void dequant_nf4_f32(__global const uchar* packed,
                              __global const float* absmax,
                              __global const float* map,
                              __global float* out,
                              const int n, const int blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    uchar b = packed[i >> 1];
    int nib = (i & 1) ? (b & 0xF) : (b >> 4);
    out[i] = map[nib] * absmax[i / blocksize];
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
        self.kern = cl.Kernel(self.prog, 'dequant_nf4_f32')
        self.bl = self.rt.bl
        self.bl.CLBlastSgemm.restype = ctypes.c_int
        self.bl.CLBlastSgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.qp = ctypes.c_void_p(int(self.rt.queue.int_ptr))
        self.wbuf = None
        self.wcap = 0
        self.xbuf = None
        self.xcap = 0
        self.obuf = None
        self.ocap = 0

    def f32(self, numel):
        return cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(4, numel * 4))


def _set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class GPU4BitLinearF32(torch.nn.Module):
    def __init__(self, lin: Linear4bit):
        super().__init__()
        self.r = _RT.get()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        qs = lin.weight.quant_state
        self.n = self.out_features * self.in_features
        self.blocksize = qs.blocksize
        self.packed = self.r.rt.to_gpu(np.ascontiguousarray(lin.weight.data.reshape(-1).numpy()))
        absmax = BF.dequantize_blockwise(qs.absmax, qs.state2).float() + qs.offset
        self.absmax = self.r.rt.to_gpu(np.ascontiguousarray(absmax.numpy().astype(np.float32)))
        self.code = self.r.rt.to_gpu(np.ascontiguousarray(qs.code.float().numpy().astype(np.float32)))

    def forward(self, x: torch.Tensor):
        r = self.r
        org = x.shape
        rows = int(np.prod(org[:-1])) if len(org) > 1 else 1
        K, N = self.in_features, self.out_features
        # per-call buffers: no shared runtime state across 232 modules
        wbuf = r.f32(self.n)
        xbuf = r.f32(rows * K)
        obuf = r.f32(rows * N)
        r.kern(r.rt.queue, (self.n,), None, self.packed, self.absmax, self.code,
               wbuf, np.int32(self.n), np.int32(self.blocksize))
        x_np = np.ascontiguousarray(x.detach().reshape(rows, K).numpy().astype(np.float32))
        cl.enqueue_copy(r.rt.queue, xbuf, x_np, is_blocking=True)
        r.rt.queue.finish()
        rc = r.bl.CLBlastSgemm(
            ROW, NO, YES, rows, N, K, 1.0,
            ctypes.c_void_p(int(xbuf.int_ptr)), 0, K,
            ctypes.c_void_p(int(wbuf.int_ptr)), 0, K,
            0.0,
            ctypes.c_void_p(int(obuf.int_ptr)), 0, N,
            ctypes.byref(r.qp), None)
        if rc != 0:
            raise RuntimeError('Sgemm rc=%d' % rc)
        out = np.empty((rows, N), dtype=np.float32)
        cl.enqueue_copy(r.rt.queue, out, obuf, is_blocking=True)
        r.rt.queue.finish()
        wbuf.release(); xbuf.release(); obuf.release()
        return torch.from_numpy(out).reshape(*org[:-1], N)


def to_igpu_f32(model, verbose=True):
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Linear4bit):
            _set_submodule(model, name, GPU4BitLinearF32(mod))
            n += 1
    if verbose:
        print('wrapped %d linears for fp32 iGPU hybrid' % n, flush=True)
    return n