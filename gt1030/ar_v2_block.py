# -*- coding: utf-8 -*-
"""M2: one V2 block forward on iGPU vs torch BlockV2 (cond_mode=none, grid=32, T=1024)."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import pyopencl as cl

import gt1030.attention_gpu as A
# override dims BEFORE importing full_gpu_layer (it snapshots A.B/A.T)
A.B, A.T = 2, 1024
A.M = A.B * A.T
A.BH = A.B * A.NH

import gt1030.full_gpu_layer as L
import gt1030.full_gpu_step_v2 as V2
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, BlockV2

B, T, MD, FF, WROW, BH, HD, NH, NKV = L.B, L.T, L.MD, L.FF, L.WROW, L.BH, L.HD, L.NH, L.NKV
M = L.M
scale = L.scale

EXTRA = r"""
__kernel void qk_norm_fwd(__global const half* x, __global const float* w,
                          __global half* y, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global half* out = y + (long)r * D;
    float ss = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) out[i] = (half)((float)row[i] * inv * w[i]);
}
__kernel void split_silu(__global const half* a2, __global half* g, int rows, int FF) {
    int i = get_global_id(0);
    if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    g[i] = (half)(a / (1.f + exp(-a)) * b);
}
__kernel void scale_add(__global const half* x, __global const half* y,
                        __global const float* s, __global half* o, int rows, int D) {
    int i = get_global_id(0);
    if (i >= rows * D) return;
    o[i] = (half)((float)x[i] + (float)y[i] * s[i % D]);
}
"""


def main():
    ctx, q, mf = A.ctx, A.queue, A.mf
    prog = cl.Program(ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + EXTRA).build(
        options=['-cl-mad-enable'])
    K = {n: cl.Kernel(prog, n) for n in
         ('rms_stats', 'rms_apply', 'split_q', 'split_kv', 'softmax_causal', 'merge_o',
          'add_vec', 'qk_norm_fwd', 'split_silu', 'scale_add')}

    torch.manual_seed(0)
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=1, n_head=8,
                             n_kv_head=4, ffn_dim=2048, total_tokens=1024, grid=32)
    blk = BlockV2(cfg, cond_dim=None).eval()
    sd = blk.state_dict()
    WQKV = torch.cat([blk.attn.q.weight, blk.attn.k.weight, blk.attn.v.weight], 0).T.detach().contiguous()
    WO = blk.attn.o.weight.T.detach().contiguous()
    W12 = blk.mlp.w12.weight.T.detach().contiguous()
    W3 = blk.mlp.w3.weight.T.detach().contiguous()
    qw = blk.attn.q_norm.w.detach().numpy().astype(np.float32)
    kw = blk.attn.k_norm.w.detach().numpy().astype(np.float32)
    ls1 = blk.ls1.detach().numpy().astype(np.float32)
    ls2 = blk.ls2.detach().numpy().astype(np.float32)
    ln1 = blk.ln1.w.detach().numpy().astype(np.float32)
    ln2 = blk.ln2.w.detach().numpy().astype(np.float32)

    def up(a):
        return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))

    def buf(nb):
        return cl.Buffer(ctx, mf.READ_WRITE, nb)

    dWqkv = up(WQKV.numpy().astype(np.float16))
    dWo = up(WO.numpy().astype(np.float16))
    dW12 = up(W12.numpy().astype(np.float16))
    dW3 = up(W3.numpy().astype(np.float16))
    # rms_apply / rms_bwd_* 里 w 声明为 half*，必须传 fp16；
    # 传 float32 会被解释成 0.0 / 1.875 交替的垃圾半精度数。
    dln1 = up(ln1.astype(np.float16)); dln2 = up(ln2.astype(np.float16))
    dqw = up(qw); dkw = up(kw)
    dls1 = up(ls1); dls2 = up(ls2)   # scale_add 的 s 声明为 float*，这里就是 float32

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, MD)) * 0.5).astype(np.float16)
    xb = up(x)
    h1 = buf(M * MD * 2); qkv = buf(M * WROW * 2)
    Q = buf(BH * T * HD * 2); Kb = buf(BH * T * HD * 2); Vb = buf(BH * T * HD * 2)
    S = buf(BH * T * T * 2); P = buf(BH * T * T * 2); O = buf(BH * T * HD * 2)
    o_tok = buf(M * MD * 2); attn = buf(M * MD * 2); x1 = buf(M * MD * 2)
    h2 = buf(M * MD * 2); a2 = buf(M * 2 * FF * 2); gbuf = buf(M * FF * 2)
    y = buf(M * MD * 2); x2 = buf(M * MD * 2)
    ss = buf(M * 4)

    K['rms_stats'](q, (M,), (64,), xb, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), xb, dln1, ss, h1, np.int32(M), np.int32(MD))
    L.hgemm_nn(h1, dWqkv, qkv, M, WROW, MD)
    K['split_q'](q, (BH * T * HD,), (256,), qkv, Q, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
    K['split_kv'](q, (BH * T * HD,), (256,), qkv, Kb, Vb, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
    # q/k per-head RMSNorm。split_q/split_kv 写出的都是【已按 NH 展开】的布局：
    #   Q: (B*NH, T, HD)   K/V: (B*NH, T, HD)（GQA 已经 repeat 过）
    # 所以 K 也要按 BH*T 行归一化；再套一层 expand_kv 会把展开后的索引当原始索引用，
    # 只有前 B*NKV 行被归一化过 —— 之前这里就是错在这里（块贡献 cosine 只有 0.87）。
    K['qk_norm_fwd'](q, (BH * T,), None, Q, dqw, Q, np.int32(BH * T), np.int32(HD))
    K['qk_norm_fwd'](q, (BH * T,), None, Kb, dkw, Kb, np.int32(BH * T), np.int32(HD))

    A.hgemm_batched(Q, Kb, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_causal'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, Vb, O, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    K['merge_o'](q, (BH * T * HD,), (256,), O, o_tok, np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
    L.hgemm_nn(o_tok, dWo, attn, M, MD, MD)
    K['scale_add'](q, (M * MD,), (256,), xb, attn, dls1, x1, np.int32(M), np.int32(MD))

    K['rms_stats'](q, (M,), (64,), x1, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), x1, dln2, ss, h2, np.int32(M), np.int32(MD))
    L.hgemm_nn(h2, dW12, a2, M, 2 * FF, MD)
    K['split_silu'](q, (M * FF,), (256,), a2, gbuf, np.int32(M), np.int32(FF))
    L.hgemm_nn(gbuf, dW3, y, M, MD, FF)
    K['scale_add'](q, (M * MD,), (256,), x1, y, dls2, x2, np.int32(M), np.int32(MD))
    q.finish()
    out = np.empty((M, MD), dtype=np.float16)
    cl.enqueue_copy(q, out, x2); q.finish()

    xt = torch.from_numpy(x.astype(np.float32)).view(B, T, MD)
    with torch.no_grad():
        ref = blk(xt).view(M, MD)
    # 注意：直接比 x2 是【无意义】的 —— x2 = x + ls1*attn + ls2*mlp，
    # x 占了 99.9% 的幅度，ls 又是 1e-2，任何块内错误都会被残差掩盖（第一版就是这么假通过的）。
    # 必须比块贡献 x2 - x；而它只有 1.4e-2 量级、x2 又是 fp16 存储（ulp ~1e-3），
    # 所以只能按绝对误差判定。
    dev = out.astype(np.float32)
    refd = ref.numpy()
    dev_delta = dev - x.astype(np.float32)
    ref_delta = refd - x.astype(np.float32)
    de = float(np.abs(dev_delta - ref_delta).max())
    print('  [dbg] dev delta max=%.4e  ref delta max=%.4e  cosine=%.5f'
          % (float(np.abs(dev_delta).max()), float(np.abs(ref_delta).max()),
             float((dev_delta * ref_delta).sum() /
                   (np.linalg.norm(dev_delta) * np.linalg.norm(ref_delta) + 1e-30))))
    print('V2 block fwd delta: max_abs=%.4e refmax=%.4e %s (abs-only, fp16 floor ~1e-3)'
          % (de, float(np.abs(refd - x.astype(np.float32)).max()),
             'OK' if de < 5e-3 else 'FAIL'))
    print('V2 block fwd x2   : max_abs=%.4e (residual-dominated, informational only)'
          % float(np.abs(dev - refd).max()))


if __name__ == '__main__':
    main()
