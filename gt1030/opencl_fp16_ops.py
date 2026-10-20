"""opencl_fp16_ops.py — device-resident fp16 conv2d for the iGPU (OpenCL + CLBlast).

The probes showed that only a fully device-resident fp16 path beats the CPU oneDNN conv:
  - per-op hybrid (im2col on CPU + transfers): 2-3x slower
  - fp32 resident: parity at best
  - fp16 resident im2col + Hgemm: 0.50-0.97x of CPU time (1.3-2x faster)

This module is the first brick of the OpenCL image engine: convolutions run as
im2col (OpenCL kernel) + CLBlast Hgemm, with weights/columns/output kept on the GPU.
It supports the subset used by SD/AnimeDiff UNets: NCHW fp16, groups=1, square kernel,
stride/padding/dilation.  Correctness is checked against the CPU conv (fp32 reference,
fp16 inputs) in test_opencl_fp16_conv.py.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112          # CLBlast layout/transpose enums (documented constants)
HALF_ONE, HALF_ZERO = 0x3C00, 0x0000
mf = cl.mem_flags

IM2COL = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void im2col_h(
    const int n_batch, const int C, const int H, const int W,
    const int KH, const int KW, const int pad, const int stride, const int dil,
    const int Hout, const int Wout, const int K, const int L,
    __global const half* x, __global half* cols)
{
    int idx = get_global_id(0);
    if (idx >= K * L) return;
    int k = idx / L;
    int l = idx % L;
    int c = k / (KH * KW);
    int rem = k % (KH * KW);
    int kh = rem / KW;
    int kw = rem % KW;
    int oh = l / Wout;
    int ow = l % Wout;
    int ih = oh * stride - pad + kh * dil;
    int iw = ow * stride - pad + kw * dil;
    half v = (half)0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = x[((n_batch * C + c) * H + ih) * W + iw];
    cols[idx] = v;
}
"""

GROUPNORM = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void add_h(__global const half* a, __global const half* b, __global half* y,
                    const int n)
{
    int i = get_global_id(0);
    if (i < n) y[i] = (half)((float)a[i] + (float)b[i]);
}

__kernel void group_norm_silu_h(
    __global const half* x, __global half* y,
    __global const half* gamma, __global const half* beta,
    const int N, const int C, const int H, const int W, const int G, const float eps)
{
    int group = get_group_id(0);
    int n = group / G;
    int g = group % G;
    int cg = C / G;
    int hw = H * W;
    int count = cg * hw;
    __local float lsum[256];
    __local float lsq[256];
    int lid = get_local_id(0);
    int ls = get_local_size(0);
    float s = 0.f, ss = 0.f;
    for (int i = lid; i < count; i += ls) {
        int off = n * C * hw + (g * cg) * hw + i;
        float v = (float)x[off];
        s += v;
        ss += v * v;
    }
    lsum[lid] = s;
    lsq[lid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = ls / 2; step > 0; step >>= 1) {
        if (lid < step) {
            lsum[lid] += lsum[lid + step];
            lsq[lid] += lsq[lid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = lsum[0] / (float)count;
    float var = lsq[0] / (float)count - mean * mean;
    float inv = rsqrt(var + eps);
    for (int i = lid; i < count; i += ls) {
        int c = i / hw;
        int q = i - c * hw;
        int off = n * C * hw + (g * cg + c) * hw + q;
        float v = ((float)x[off] - mean) * inv * (float)gamma[g * cg + c] + (float)beta[g * cg + c];
        y[off] = (half)(v / (1.f + exp(-v)));
    }
}
"""


ATTENTION = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'kernels_attention.cl'), encoding='utf-8').read()


class OpenCLFP16:
    """Singleton-ish runtime: one context/queue, compiled kernels, CLBlast handle."""

    _inst = None

    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.prog = cl.Program(self.ctx, IM2COL + GROUPNORM + ATTENTION).build(
            options=['-cl-mad-enable'])
        self.im2col = cl.Kernel(self.prog, 'im2col_h')
        self.group_norm_silu = cl.Kernel(self.prog, 'group_norm_silu_h')
        self.add = cl.Kernel(self.prog, 'add_h')
        self.split_heads = cl.Kernel(self.prog, 'split_heads')
        self.merge_heads = cl.Kernel(self.prog, 'merge_heads')
        self.softmax_rows = cl.Kernel(self.prog, 'softmax_rows')
        self.upsample = cl.Kernel(self.prog, 'upsample_nearest2x')
        self.bl = ctypes.CDLL(DLL)
        self.bl.CLBlastHgemm.restype = ctypes.c_int
        self.bl.CLBlastHgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.bl.CLBlastHgemmStridedBatched.restype = ctypes.c_int
        self.bl.CLBlastHgemmStridedBatched.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_uint16,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        self.device_name = cl.get_platforms()[0].get_devices()[0].name
        self.max_alloc = cl.get_platforms()[0].get_devices()[0].max_mem_alloc_size

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    # -- buffers -------------------------------------------------------------
    def to_gpu(self, arr: np.ndarray):
        return cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=np.ascontiguousarray(arr))

    def empty_half(self, numel: int):
        return cl.Buffer(self.ctx, mf.READ_WRITE, max(4, numel * 2))

    def to_host(self, buf, shape) -> np.ndarray:
        out = np.empty(int(np.prod(shape)), dtype=np.float16)
        cl.enqueue_copy(self.queue, out, buf)
        self.queue.finish()
        return out.reshape(shape)

    # -- ops ------------------------------------------------------------------
    def conv2d_h(self, x_buf, w_buf, shape, weight_shape, stride=1, pad=1, dil=1,
                 out_buf=None, cols_buf=None):
        """x_buf (N,C,H,W) half device buffer, w_buf (O,C,KH,KW) half device buffer.

        Returns (out_buf (N,O,Hout,Wout) half, (N,C,H,W), (O,C,KH,KW)).  Buffers are
        left on the device; caller owns them.  groups=1.
        """
        N, C, H, W = shape
        O, Cw, KH, KW = weight_shape
        assert Cw == C and KH == KW
        hout = (H + 2 * pad - dil * (KH - 1) - 1) // stride + 1
        wout = (W + 2 * pad - dil * (KW - 1) - 1) // stride + 1
        K = C * KH * KW
        L = hout * wout
        if out_buf is None:
            out_buf = self.empty_half(N * O * L)
        if cols_buf is not None and N == 1:
            # reuse the caller's buffer only when it is large enough
            pass
        qp = ctypes.c_void_p(int(self.queue.int_ptr))
        for n in range(N):
            cols = self.empty_half(K * L)
            self.im2col(self.queue, (K * L,), None,
                        np.int32(n), np.int32(C), np.int32(H), np.int32(W),
                        np.int32(KH), np.int32(KW),
                        np.int32(pad), np.int32(stride), np.int32(dil),
                        np.int32(hout), np.int32(wout), np.int32(K), np.int32(L),
                        x_buf, cols)
            rc = self.bl.CLBlastHgemm(
                ROW, NO, NO, O, L, K, HALF_ONE,
                ctypes.c_void_p(int(w_buf.int_ptr)), 0, K,
                ctypes.c_void_p(int(cols.int_ptr)), 0, L,
                HALF_ZERO,
                ctypes.c_void_p(int(out_buf.int_ptr)), n * O * L, L,
                ctypes.byref(qp), None)
            cols.release()
            if rc != 0:
                raise RuntimeError('CLBlastHgemm rc=%d' % rc)
        self.queue.finish()
        return out_buf, (N, O, hout, wout)

    def group_norm_silu_buf(self, x_buf, shape, gamma_buf, beta_buf, groups=32, eps=1e-6):
        """In-place-able grouped norm + SiLU: x (N,C,H,W) half -> y half (new buffer)."""
        N, C, H, W = shape
        assert C % groups == 0
        y = self.empty_half(N * C * H * W)
        ls = 256
        self.group_norm_silu(self.queue, (N * groups * ls,), (ls,),
                             x_buf, y, gamma_buf, beta_buf,
                             np.int32(N), np.int32(C), np.int32(H), np.int32(W),
                             np.int32(groups), np.float32(eps))
        self.queue.finish()
        return y

    def upsample2x_buf(self, x_buf, shape):
        """Nearest 2x upsample: (N,C,H,W) half -> (N,C,2H,2W) half."""
        N, C, H, W = shape
        y = self.empty_half(N * C * 4 * H * W)
        self.upsample(self.queue, (N * C * 4 * H * W,), None, x_buf, y,
                      np.int32(N), np.int32(C), np.int32(H), np.int32(W))
        self.queue.finish()
        return y, (N, C, 2 * H, 2 * W)

    def add_buf(self, a_buf, b_buf, numel: int):
        y = self.empty_half(numel)
        self.add(self.queue, (numel,), None, a_buf, b_buf, y, np.int32(numel))
        self.queue.finish()
        return y

    # -- linear / attention ---------------------------------------------------
    def linear_h(self, x_buf, w_t_buf, M: int, K: int, N: int, out_buf=None):
        """y (M,N) = x (M,K) @ w_t (K,N), all fp16 device buffers."""
        if out_buf is None:
            out_buf = self.empty_half(M * N)
        qp = ctypes.c_void_p(int(self.queue.int_ptr))
        rc = self.bl.CLBlastHgemm(
            ROW, NO, NO, M, N, K, HALF_ONE,
            ctypes.c_void_p(int(x_buf.int_ptr)), 0, K,
            ctypes.c_void_p(int(w_t_buf.int_ptr)), 0, N,
            HALF_ZERO,
            ctypes.c_void_p(int(out_buf.int_ptr)), 0, N,
            ctypes.byref(qp), None)
        if rc != 0:
            raise RuntimeError('linear Hgemm rc=%d' % rc)
        self.queue.finish()
        return out_buf, (M, N)

    def attention_h(self, q_in, kv_in, B: int, S: int, Skv: int, C: int, heads: int,
                    wq, wk, wv, wo):
        """Multi-head attention on the device.

        q_in: (B*S,C); kv_in: (B*Skv,C). wq/wk/wv/wo: fp16 (C,C) transposed-weight
        buffers (in_features, out_features). Returns the (B*S,C) output buffer.
        Cross-attention is just Skv != S.
        """
        D = C // heads
        scale = 1.0 / np.sqrt(float(D))
        q, _ = self.linear_h(q_in, wq, B * S, C, C)
        k, _ = self.linear_h(kv_in, wk, B * Skv, C, C)
        v, _ = self.linear_h(kv_in, wv, B * Skv, C, C)
        qh = self.empty_half(B * heads * S * D)
        kh = self.empty_half(B * heads * Skv * D)
        vh = self.empty_half(B * heads * Skv * D)
        self.split_heads(self.queue, (B * S * C,), None, q, qh, np.int32(C),
                         np.int32(S), np.int32(heads), np.int32(D), np.float32(scale))
        self.split_heads(self.queue, (B * Skv * C,), None, k, kh, np.int32(C),
                         np.int32(Skv), np.int32(heads), np.int32(D), np.float32(1.0))
        self.split_heads(self.queue, (B * Skv * C,), None, v, vh, np.int32(C),
                         np.int32(Skv), np.int32(heads), np.int32(D), np.float32(1.0))
        scores = self.empty_half(B * heads * S * Skv)
        qp = ctypes.c_void_p(int(self.queue.int_ptr))
        rc = self.bl.CLBlastHgemmStridedBatched(
            ROW, NO, YES, S, Skv, D, HALF_ONE,
            ctypes.c_void_p(int(qh.int_ptr)), 0, D, S * D,
            ctypes.c_void_p(int(kh.int_ptr)), 0, D, Skv * D,
            HALF_ZERO,
            ctypes.c_void_p(int(scores.int_ptr)), 0, Skv, S * Skv,
            B * heads, ctypes.byref(qp), None)
        if rc != 0:
            raise RuntimeError('score batched Hgemm rc=%d' % rc)
        probs = self.empty_half(B * heads * S * Skv)
        self.softmax_rows(self.queue, (B * heads * S,), None, scores, probs,
                          np.int32(B * heads * S), np.int32(Skv))
        attn = self.empty_half(B * heads * S * D)
        rc = self.bl.CLBlastHgemmStridedBatched(
            ROW, NO, NO, S, D, Skv, HALF_ONE,
            ctypes.c_void_p(int(probs.int_ptr)), 0, Skv, S * Skv,
            ctypes.c_void_p(int(vh.int_ptr)), 0, D, Skv * D,
            HALF_ZERO,
            ctypes.c_void_p(int(attn.int_ptr)), 0, D, S * D,
            B * heads, ctypes.byref(qp), None)
        if rc != 0:
            raise RuntimeError('value batched Hgemm rc=%d' % rc)
        merged = self.empty_half(B * S * C)
        self.merge_heads(self.queue, (B * S * C,), None, attn, merged, np.int32(C),
                         np.int32(S), np.int32(heads), np.int32(D))
        out, _ = self.linear_h(merged, wo, B * S, C, C)
        for b in (q, k, v, qh, kh, vh, scores, probs, attn, merged):
            b.release()
        return out


def demo():
    """Small self-check: conv2d on the GPU vs the CPU reference."""
    import torch
    import torch.nn.functional as F
    rt = OpenCLFP16.get()
    torch.set_num_threads(6)
    for cin, h, w, k, stride, pad in [(320, 64, 64, 3, 1, 1), (640, 32, 32, 3, 1, 1),
                                      (1280, 16, 16, 3, 1, 1), (320, 32, 32, 1, 1, 0),
                                      (64, 32, 32, 3, 2, 1)]:
        x = np.random.randn(1, cin, h, w).astype(np.float32)
        conv = torch.nn.Conv2d(cin, cin, k, stride=stride, padding=pad).eval()
        wt = conv.weight.detach().numpy().astype(np.float32)
        xh = x.astype(np.float16)
        wh = wt.astype(np.float16)
        xb = rt.to_gpu(xh)
        wb = rt.to_gpu(wh)
        out_buf, oshape = rt.conv2d_h(xb, wb, xh.shape, wh.shape, stride, pad)
        got = rt.to_host(out_buf, oshape).astype(np.float32)
        ref = F.conv2d(torch.from_numpy(xh.astype(np.float32)),
                       torch.from_numpy(wh.astype(np.float32)),
                       stride=stride, padding=pad).numpy()
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9))
        print('shape=(%d,%d,%d) k=%d s=%d p=%d -> out%s | relL2 %.4f | max %.4f'
              % (cin, h, w, k, stride, pad, oshape, rel, float(np.abs(got - ref).max())))
        for b in (xb, wb, out_buf):
            b.release()


if __name__ == '__main__':
    demo()
