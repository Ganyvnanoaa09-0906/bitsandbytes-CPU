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
from opencl_fp16_ops import OpenCLFP16, ROW, NO, YES, HALF_ONE, HALF_ZERO  # noqa: E402

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

KERNEL_H = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void dequant_nf4_h(__global const uchar* packed,
                            __global const float* absmax,
                            __global const float* map,
                            __global half* out,
                            const int n, const int blocksize)
{
    int i = get_global_id(0);
    if (i >= n) return;
    uchar b = packed[i >> 1];
    int nib = (i & 1) ? (b & 0xF) : (b >> 4);
    out[i] = (half)(map[nib] * absmax[i / blocksize]);
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
        self.prog_h = cl.Program(self.rt.ctx, KERNEL_H).build(options=['-cl-mad-enable'])
        self.kern_h = cl.Kernel(self.prog_h, 'dequant_nf4_h')
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
        self.ascratch = None
        self.acap = 0
        self.cscratch = None
        self.ccap = 0
        self.hwbuf = None
        self.hwcap = 0
        self.hxbuf = None
        self.hxcap = 0
        self.hobuf = None
        self.hocap = 0

    def f32(self, numel):
        return cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(4, numel * 4))

    def half(self, numel):
        return cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(2, numel * 2))

    def upload(self, arr):
        arr = np.ascontiguousarray(arr)
        buf = cl.Buffer(self.rt.ctx, cl.mem_flags.READ_WRITE, max(4, arr.nbytes))
        cl.enqueue_copy(self.rt.queue, buf, arr, is_blocking=True)
        self.rt.queue.finish()
        return buf


def _set_submodule(model, name, new):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class GPU4BitLinearF32(torch.nn.Module):
    def __init__(self, lin: Linear4bit, mode: str = 'fp32'):
        super().__init__()
        self.r = _RT.get()
        self.mode = mode
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        qs = lin.weight.quant_state
        self.n = self.out_features * self.in_features
        self.blocksize = qs.blocksize
        self.packed = self.r.upload(lin.weight.data.reshape(-1).numpy())
        absmax = BF.dequantize_blockwise(qs.absmax, qs.state2).float() + qs.offset
        self.absmax_host = np.ascontiguousarray(absmax.numpy().astype(np.float32))
        self.code_host = np.ascontiguousarray(qs.code.float().numpy().astype(np.float32))
        self.absmax = None
        self.code = None

    def prepare(self):
        if self.absmax is None:
            self.absmax = self.r.upload(self.absmax_host)
        if self.code is None:
            self.code = self.r.upload(self.code_host)

    def forward(self, x: torch.Tensor):
        r = self.r
        org = x.shape
        rows = int(np.prod(org[:-1])) if len(org) > 1 else 1
        K, N = self.in_features, self.out_features
        if r.wcap < self.n:
            if r.wbuf is not None:
                r.wbuf.release()
            r.wbuf = r.f32(self.n)
            r.wcap = self.n
        na = self.absmax_host.size
        if r.acap < na:
            if r.ascratch is not None:
                r.ascratch.release()
            r.ascratch = r.f32(na)
            r.acap = na
        nc = self.code_host.size
        if r.ccap < nc:
            if r.cscratch is not None:
                r.cscratch.release()
            r.cscratch = r.f32(nc)
            r.ccap = nc
        cl.enqueue_copy(r.rt.queue, r.ascratch, self.absmax_host, is_blocking=True)
        cl.enqueue_copy(r.rt.queue, r.cscratch, self.code_host, is_blocking=True)
        if self.mode == 'fp16':
            if r.hwcap < self.n:
                if r.hwbuf is not None:
                    r.hwbuf.release()
                r.hwbuf = r.half(self.n)
                r.hwcap = self.n
            r.kern_h(r.rt.queue, (self.n,), None, self.packed, r.ascratch, r.cscratch,
                     r.hwbuf, np.int32(self.n), np.int32(self.blocksize))
            if r.hxcap < rows * K:
                if r.hxbuf is not None:
                    r.hxbuf.release()
                r.hxbuf = r.half(rows * K)
                r.hxcap = rows * K
            if r.hocap < rows * N:
                if r.hobuf is not None:
                    r.hobuf.release()
                r.hobuf = r.half(rows * N)
                r.hocap = rows * N
            x_np = np.ascontiguousarray(
                x.detach().reshape(rows, K).numpy().astype(np.float16))
            cl.enqueue_copy(r.rt.queue, r.hxbuf, x_np, is_blocking=True)
            r.rt.queue.finish()
            rc = r.bl.CLBlastHgemm(
                ROW, NO, YES, rows, N, K, HALF_ONE,
                ctypes.c_void_p(int(r.hxbuf.int_ptr)), 0, K,
                ctypes.c_void_p(int(r.hwbuf.int_ptr)), 0, K,
                HALF_ZERO,
                ctypes.c_void_p(int(r.hobuf.int_ptr)), 0, N,
                ctypes.byref(r.qp), None)
            if rc != 0:
                raise RuntimeError('Hgemm rc=%d' % rc)
            out16 = np.empty((rows, N), dtype=np.float16)
            cl.enqueue_copy(r.rt.queue, out16, r.hobuf, is_blocking=True)
            r.rt.queue.finish()
            return torch.from_numpy(out16.astype(np.float32)).reshape(*org[:-1], N)
        r.kern(r.rt.queue, (self.n,), None, self.packed, r.ascratch, r.cscratch,
               r.wbuf, np.int32(self.n), np.int32(self.blocksize))
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
        r.rt.queue.finish()
        rc = r.bl.CLBlastSgemm(
            ROW, NO, YES, rows, N, K, 1.0,
            ctypes.c_void_p(int(r.xbuf.int_ptr)), 0, K,
            ctypes.c_void_p(int(r.wbuf.int_ptr)), 0, K,
            0.0,
            ctypes.c_void_p(int(r.obuf.int_ptr)), 0, N,
            ctypes.byref(r.qp), None)
        if rc != 0:
            raise RuntimeError('Sgemm rc=%d' % rc)
        out = np.empty((rows, N), dtype=np.float32)
        cl.enqueue_copy(r.rt.queue, out, r.obuf, is_blocking=True)
        r.rt.queue.finish()
        return torch.from_numpy(out).reshape(*org[:-1], N)


def to_igpu_f32(model, mode_fn=None, verbose=True):
    n = 0
    modes = {'fp32': 0, 'fp16': 0}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Linear4bit):
            mode = 'fp32' if mode_fn is None else mode_fn(name)
            _set_submodule(model, name, GPU4BitLinearF32(mod, mode=mode))
            modes[mode] += 1
            n += 1
    if verbose:
        print('wrapped %d linears for iGPU hybrid %s' % (n, modes), flush=True)
    return n