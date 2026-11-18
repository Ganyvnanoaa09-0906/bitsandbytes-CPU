# -*- coding: utf-8 -*-
"""igpu_time_emb.py — UNet 的时间步嵌入：**sinusoidal 先算，再 Linear→SiLU→Linear**。

注意（容易被写漏）：timestep 部分**不是 MLP**——
`get_timestep_embedding` 是解析式 sinusoidal（没有可训练参数，flip_sin_to_cos=True、
downscale_freq_shift=0），之后才是 `TimestepEmbedding = Linear → SiLU → Linear`。

本模块：
  · `sinusoidal(t, dim, out_f32)`   —— 设备上用 fp32 算三角函数，输出 **fp16**（与激活一致）；
  · `TimeEmbed`                     —— 两段 Linear + SiLU，权重来自真实 UNet。

    python gt1030/igpu_time_emb.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402
from gt1030.igpu_nn_ops import _kernels as NNK  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// sinusoidal timestep embedding（diffusers 语义）：
//   前半 [0, dim/2) 是 cos(t·f_i)，后半 [dim/2, dim) 是 sin(t·f_i)
//   —— 注意是**成块**排列，不是 cos/sin 交错（flip_sin_to_cos 交换的是两个半块）。
__kernel void sinusoidal(__global const float* t,  // (N,)
                         __global half* out,       // (N, dim)
                         int N, int dim, float log_base) {
    long idx = get_global_id(0);
    long total = (long)N * dim;
    if (idx >= total) return;
    int n = (int)(idx / dim);
    int c = (int)(idx % dim);
    int halfdim = dim / 2;
    int i = (c < halfdim) ? c : (c - halfdim);
    float freq = exp(-log_base * (float)i / (float)halfdim);
    float a = t[n] * freq;
    out[idx] = (half)((c < halfdim) ? cos(a) : sin(a));
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build()
        _C[key] = {n: cl.Kernel(prog, n) for n in ("sinusoidal",)}
    return _C[key]


def sinusoidal(dev, t_f32_buf, out, N, dim):
    K(dev)["sinusoidal"](dev.q, (N * dim,), None, t_f32_buf, out,
                         np.int32(N), np.int32(dim), np.float32(np.log(10000.0)))
    return out


class TimeEmbed(object):
    """`TimestepEmbedding`：Linear(dim→D) → SiLU → Linear(D→D)，权重可用真实值。"""

    def __init__(self, dev, n, dim, D):
        self.dev, self.N, self.dim, self.D = dev, int(n), int(dim), int(D)
        self.w1 = dev.buf(max(4096, D * dim * 2))
        self.b1 = dev.buf(max(4096, D * 2))
        self.w2 = dev.buf(max(4096, D * D * 2))
        self.b2 = dev.buf(max(4096, D * 2))
        self.h = dev.buf(max(4096, n * D * 2))
        self.out = dev.buf(max(4096, n * D * 2))

    def set_weights(self, w1, b1, w2, b2):
        for buf, w in ((self.w1, w1), (self.b1, b1), (self.w2, w2), (self.b2, b2)):
            cl.enqueue_copy(self.dev.q, buf,
                            np.ascontiguousarray(np.asarray(w).reshape(-1), np.float16))

    def forward(self, temb):
        d = self.dev
        GEMM.hgemm_nt(temb, self.w1, self.h, self.N, self.D, self.dim)   # H = temb · W1ᵀ
        NNK(d)["bias_add"](d.q, (self.N * self.D,), None, self.h, self.b1,
                           np.int64(self.N * self.D)) if False else None
        _bias_add(d, self.h, self.b1, self.N, self.D)
        _silu(d, self.h, self.N * self.D)
        GEMM.hgemm_nt(self.h, self.w2, self.out, self.N, self.D, self.D)
        _bias_add(d, self.out, self.b2, self.N, self.D)
        return self.out


def _bias_add(dev, buf, bias, rows, cols):
    """out[r,c] += bias[c]：复用 conv 的 bias_add_oc（按 (rows, cols) 解释）。"""
    from gt1030.igpu_conv import K as CVK2

    CVK2(dev)["bias_add_oc"](dev.q, (rows, cols), None, buf, bias,
                             np.int32(rows), np.int32(cols))


def _silu(dev, buf, n):
    """SiLU 用现成 kernel（igpu_nn_ops 的 SILU_SRC 由 igpu_fx 编译；这里直接内联一份）。"""
    key = ("silu", int(dev.ctx.int_ptr))
    if key not in _C:
        prog = cl.Program(dev.ctx, r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void silu_inplace(__global half* x, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    x[i] = (half)(v / (1.f + exp(-v)));
}
""").build(options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _C[key] = cl.Kernel(prog, "silu_inplace")
    _C[key](dev.q, (int(n),), None, buf, np.int64(n))


def check(N=2, dim=320, D=1280, seed=0):
    import torch
    from diffusers.models.embeddings import get_timestep_embedding

    rng = np.random.default_rng(seed)
    t = (np.arange(N, dtype=np.float32) * 100.0 + 1.0).astype(np.float32)
    w1 = (rng.standard_normal((D, dim)) * 0.05).astype(np.float32)
    b1 = (rng.standard_normal(D) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((D, D)) * 0.02).astype(np.float32)
    b2 = (rng.standard_normal(D) * 0.02).astype(np.float32)

    dev = CTX.DevCtx(arena_mb=800, name="temb")
    dev.activate()
    tb = dev.buf(max(4096, N * 4))
    cl.enqueue_copy(dev.q, tb, np.ascontiguousarray(t))
    sinb = dev.buf(max(4096, N * dim * 2))
    sinusoidal(dev, tb, sinb, N, dim)
    dev.q.finish()
    sinv = dev.down(sinb, (N, dim)).astype(np.float32)

    ref_sin = get_timestep_embedding(torch.from_numpy(t).long(), dim,
                                     flip_sin_to_cos=True, downscale_freq_shift=0).numpy()
    rel_sin = float(np.linalg.norm(sinv - ref_sin) / np.linalg.norm(ref_sin))

    te = TimeEmbed(dev, N, dim, D)
    te.set_weights(w1, b1, w2, b2)
    out = te.forward(sinb)
    dev.q.finish()
    ov = dev.down(out, (N, D)).astype(np.float32)

    ts = torch.from_numpy(ref_sin)
    ref = (torch.nn.functional.silu(ts @ torch.from_numpy(w1).T + torch.from_numpy(b1))
           @ torch.from_numpy(w2).T + torch.from_numpy(b2)).numpy()
    rel = float(np.linalg.norm(ov - ref) / np.linalg.norm(ref))
    print("  sinusoidal rel %.2e · MLP rel %.2e  (N%d dim%d D%d)" % (rel_sin, rel, N, dim, D))
    return rel_sin < 2e-3 and rel < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    ok &= check(N=3, dim=320, D=320, seed=1)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
