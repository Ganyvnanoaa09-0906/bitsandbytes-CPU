# -*- coding: utf-8 -*-
"""M3: one V2 block backward on iGPU vs torch autograd (ln/qk-norm weights frozen)."""
from __future__ import annotations
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import pyopencl as cl

import gt1030.attention_gpu as A
A.B, A.T = 2, 1024
A.M = A.B * A.T
A.BH = A.B * A.NH
import gt1030.full_gpu_layer as L
import gt1030.full_gpu_step_v2 as V2
from small_image_model_v2 import SmallImageConfigV2, BlockV2

B, T, MD, FF, WROW, BH, HD, NH, NKV = L.B, L.T, L.MD, L.FF, L.WROW, L.BH, L.HD, L.NH, L.NKV
M = L.M
scale = L.scale

KSRC = r"""
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
__kernel void qk_norm_dx(__global const half* x, __global const float* w,
                         __global const half* dy, __global half* dx, int rows, int D) {
    int r = get_global_id(0);
    __global const half* row = x + (long)r * D;
    __global const half* drow = dy + (long)r * D;
    __global half* ox = dx + (long)r * D;
    float ss = 0.f, dot = 0.f;
    for (int i = 0; i < D; ++i) { float v = (float)row[i]; ss += v * v; }
    float inv = rsqrt(ss / D + 1e-6f);
    for (int i = 0; i < D; ++i) dot += (float)row[i] * (float)drow[i];
    float c = dot * inv * inv * inv / D;
    for (int i = 0; i < D; ++i)
        ox[i] = (half)((float)drow[i] * inv * w[i] - (float)row[i] * c * w[i]);
}
__kernel void split_silu(__global const half* a2, __global half* g, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    g[i] = (half)(a / (1.f + exp(-a)) * b);
}
__kernel void split_silu_bwd(__global const half* a2, __global const half* dg,
                             __global half* da2, int rows, int FF) {
    int i = get_global_id(0); if (i >= rows * FF) return;
    int r = i / FF, j = i % FF;
    float a = (float)a2[(long)r * 2 * FF + j];
    float b = (float)a2[(long)r * 2 * FF + FF + j];
    float g = (float)dg[i];
    float sg = 1.f / (1.f + exp(-a));
    float silu_a = a * sg;
    float dsilu = sg * (1.f + a * (1.f - sg));
    da2[(long)r * 2 * FF + j] = (half)(g * b * dsilu);
    da2[(long)r * 2 * FF + FF + j] = (half)(g * silu_a);
}
__kernel void scale_vec(__global const half* x, __global const float* s, __global half* y, int rows, int D) {
    int i = get_global_id(0); if (i >= rows * D) return;
    y[i] = (half)((float)x[i] * s[i % D]);
}
"""


def main():
    ctx, q, mf = A.ctx, A.queue, A.mf
    prog = cl.Program(ctx, A.KERNELS + V2.KERNELS + L.EXTRA_KERNELS + KSRC).build(
        options=['-cl-mad-enable'])
    names = ('rms_stats', 'rms_apply', 'rms_bwd_stats', 'rms_bwd_apply', 'split_q', 'split_kv',
             'softmax_causal', 'softmax_bwd_causal', 'merge_o', 'split_dout', 'merge_dqkv',
             'add_vec', 'silu_gate', 'qk_norm_fwd', 'qk_norm_dx', 'split_silu', 'split_silu_bwd',
             'scale_vec')
    K = {n: cl.Kernel(prog, n) for n in names}

    torch.manual_seed(0)
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=1, n_head=8,
                             n_kv_head=4, ffn_dim=2048, total_tokens=1024, grid=32)
    blk = BlockV2(cfg, cond_dim=None)
    # freeze non-tested params
    for p in (blk.ln1.w, blk.ln2.w, blk.attn.q_norm.w, blk.attn.k_norm.w, blk.ls1, blk.ls2):
        p.requires_grad_(False)
    sd = blk.state_dict()
    WQKV = torch.cat([blk.attn.q.weight, blk.attn.k.weight, blk.attn.v.weight], 0).T.detach().numpy().astype(np.float16)
    WO = blk.attn.o.weight.T.detach().numpy().astype(np.float16)
    W12 = blk.mlp.w12.weight.T.detach().numpy().astype(np.float16)
    W3 = blk.mlp.w3.weight.T.detach().numpy().astype(np.float16)
    qw = blk.attn.q_norm.w.detach().numpy().astype(np.float32)
    kw = blk.attn.k_norm.w.detach().numpy().astype(np.float32)
    ln1 = blk.ln1.w.detach().numpy().astype(np.float32)
    ln2 = blk.ln2.w.detach().numpy().astype(np.float32)
    ls1 = blk.ls1.detach().numpy().astype(np.float32)
    ls2 = blk.ls2.detach().numpy().astype(np.float32)

    def up(a):
        return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))

    def buf(nb):
        return cl.Buffer(ctx, mf.READ_WRITE, nb)

    def cp(h, b, n, off=0):
        cl.enqueue_copy(q, b, h, byte_count=n, dst_offset=off)

    dWqkv, dWo, dW12, dW3 = up(WQKV), up(WO), up(W12), up(W3)
    uq, uk = up(qw), up(kw)
    uln1, uln2 = up(ln1.astype(np.float16)), up(ln2.astype(np.float16))   # rms_* want half*
    uls1, uls2 = up(ls1), up(ls2)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, MD)) * 0.5).astype(np.float16)
    xb = up(x)
    h1, qkv = buf(M * MD * 2), buf(M * WROW * 2)
    Q, Kb, Vb = buf(BH * T * HD * 2), buf(BH * T * HD * 2), buf(BH * T * HD * 2)
    Qe, Ke, Ve = buf(BH * T * HD * 2), buf(BH * T * HD * 2), buf(BH * T * HD * 2)
    Qn, Kn = buf(BH * T * HD * 2), buf(BH * T * HD * 2)
    S, P, O = buf(BH * T * T * 2), buf(BH * T * T * 2), buf(BH * T * HD * 2)
    o_tok, attn, x1, h2 = buf(M * MD * 2), buf(M * MD * 2), buf(M * MD * 2), buf(M * MD * 2)
    a2, gbuf, y = buf(M * 2 * FF * 2), buf(M * FF * 2), buf(M * MD * 2)
    x2 = buf(M * MD * 2)
    ss, sc = buf(M * 4), buf(M * 4)
    EXP = r"""
__kernel void expand_kv(__global const half* k, __global half* out, int T, int D, int B, int nkv, int nh) {
    int i = get_global_id(0); int total = B * nh * T * D; if (i >= total) return;
    int d = i % D, t = (i / D) % T, hp = i / (D * T);
    int b = hp / nh, h = hp % nh, kh = h / (nh / nkv);
    out[i] = k[((b * nkv + kh) * T + t) * D + d];
}
"""
    prog2 = cl.Program(ctx, EXP).build()
    kexp = cl.Kernel(prog2, 'expand_kv')

    # ---------- forward ----------
    K['rms_stats'](q, (M,), (64,), xb, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), xb, uln1, ss, h1, np.int32(M), np.int32(MD))
    L.hgemm_nn(h1, dWqkv, qkv, M, WROW, MD)
    K['split_q'](q, (BH * T * HD,), (256,), qkv, Q, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
    K['split_kv'](q, (BH * T * HD,), (256,), qkv, Kb, Vb, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
    K['qk_norm_fwd'](q, (BH * T,), None, Q, uq, Qn, np.int32(BH * T), np.int32(HD))
    K['qk_norm_fwd'](q, (BH * T,), None, Kb, uk, Kn, np.int32(BH * T), np.int32(HD))
    # split_kv already writes NH-expanded K/V; Ke/Ve unused
    # (Ve unused)
    A.hgemm_batched(Qn, Kn, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_causal'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, Vb, O, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    K['merge_o'](q, (BH * T * HD,), (256,), O, o_tok, np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
    L.hgemm_nn(o_tok, dWo, attn, M, MD, MD)
    # x1 = x + ls1*attn  (scale_add)
    K['scale_vec'](q, (M * MD,), (256,), attn, uls1, x1, np.int32(M), np.int32(MD))
    K['add_vec'](q, (M * MD,), (256,), xb, x1, x1, np.int32(M * MD))
    K['rms_stats'](q, (M,), (64,), x1, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), x1, uln2, ss, h2, np.int32(M), np.int32(MD))
    L.hgemm_nn(h2, dW12, a2, M, 2 * FF, MD)
    K['split_silu'](q, (M * FF,), (256,), a2, gbuf, np.int32(M), np.int32(FF))
    L.hgemm_nn(gbuf, dW3, y, M, MD, FF)
    K['scale_vec'](q, (M * MD,), (256,), y, uls2, y, np.int32(M), np.int32(MD))
    K['add_vec'](q, (M * MD,), (256,), x1, y, x2, np.int32(M * MD))

    # ---------- backward (dout provided from outside; use ones-ish random) ----------
    dout = (rng.standard_normal((M, MD)) * 0.5).astype(np.float16)
    doutb = up(dout)
    dW3b, dW12b, dWob, dWqkvb = up(np.zeros_like(W3)), up(np.zeros_like(W12)), up(np.zeros_like(WO)), up(np.zeros_like(WQKV))
    dy = buf(M * MD * 2); dg = buf(M * FF * 2); da2 = buf(M * 2 * FF * 2)
    dh2 = buf(M * MD * 2); dx1_mlp = buf(M * MD * 2)
    dattn = buf(M * MD * 2); dO_tok = buf(M * MD * 2); dO = buf(BH * T * HD * 2)
    dP = buf(BH * T * T * 2); dS = buf(BH * T * T * 2)
    dQ = buf(BH * T * HD * 2); dK = buf(BH * T * HD * 2); dV = buf(BH * T * HD * 2)
    dqkv = buf(M * WROW * 2); dh1 = buf(M * MD * 2); dx_attn = buf(M * MD * 2)
    dx = buf(M * MD * 2); dQraw = buf(BH * T * HD * 2); dKraw = buf(BH * T * HD * 2)

    # MLP bwd
    K['scale_vec'](q, (M * MD,), (256,), doutb, uls2, dy, np.int32(M), np.int32(MD))
    L.hgemm_tn(gbuf, dy, dW3b, FF, MD, M)
    L.hgemm_nt(dy, dW3, dg, M, FF, MD)
    K['split_silu_bwd'](q, (M * FF,), (256,), a2, dg, da2, np.int32(M), np.int32(FF))
    L.hgemm_tn(h2, da2, dW12b, MD, 2 * FF, M)
    L.hgemm_nt(da2, dW12, dh2, M, MD, 2 * FF)
    K['rms_bwd_stats'](q, (M,), (64,), x1, dh2, uln2, ss, sc, np.int32(M), np.int32(MD))
    K['rms_bwd_apply'](q, (M * MD,), (256,), x1, dh2, uln2, ss, sc, dx1_mlp, np.int32(M), np.int32(MD))
    # attn bwd
    K['scale_vec'](q, (M * MD,), (256,), doutb, uls1, dattn, np.int32(M), np.int32(MD))
    L.hgemm_tn(o_tok, dattn, dWob, MD, MD, M)
    L.hgemm_nt(dattn, dWo, dO_tok, M, MD, MD)
    K['split_dout'](q, (BH * T * HD,), (256,), dO_tok, dO, np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
    A.hgemm_batched(dO, Vb, dP, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_bwd_causal'](q, (T * 256, BH), (256, 1), P, dP, dS, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, dO, dV, T, HD, T, A.YES, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    A.hgemm_batched(dS, Kn, dQ, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    A.hgemm_batched(dS, Qn, dK, T, HD, T, A.YES, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    # qk norm bwd
    K['qk_norm_dx'](q, (BH * T,), None, Q, uq, dQ, dQraw, np.int32(BH * T), np.int32(HD))
    K['qk_norm_dx'](q, (BH * T,), None, Kb, uk, dK, dKraw, np.int32(BH * T), np.int32(HD))
    # for merge, expanded dK (NH) must be reduced to NKV; merge_dqkv expects dQ,dK,dV expanded
    K['merge_dqkv'](q, (M * WROW,), (256,), dQraw, dKraw, dV, dqkv,
                    np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
    L.hgemm_tn(h1, dqkv, dWqkvb, MD, WROW, M)
    L.hgemm_nt(dqkv, dWqkv, dh1, M, MD, WROW)
    K['rms_bwd_stats'](q, (M,), (64,), xb, dh1, uln1, ss, sc, np.int32(M), np.int32(MD))
    K['rms_bwd_apply'](q, (M * MD,), (256,), xb, dh1, uln1, ss, sc, dx_attn, np.int32(M), np.int32(MD))
    K['add_vec'](q, (M * MD,), (256,), doutb, dx1_mlp, dx, np.int32(M * MD))
    K['add_vec'](q, (M * MD,), (256,), dx, dx_attn, dx, np.int32(M * MD))
    q.finish()

    def dn(b, shape, dtype=np.float16):
        a = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(q, a, b); q.finish(); return a

    dx_np = dn(dx, (M, MD)).astype(np.float32)
    dWq_np = dn(dWqkvb, WQKV.shape).astype(np.float32)
    dWo_np = dn(dWob, WO.shape).astype(np.float32)
    dW12_np = dn(dW12b, W12.shape).astype(np.float32)
    dW3_np = dn(dW3b, W3.shape).astype(np.float32)
    x2_np = dn(x2, (M, MD)).astype(np.float32)
    y_np = dn(y, (M, MD)).astype(np.float32)     # = ls2 * (g @ W3)

    # torch ref
    xt = torch.from_numpy(x.astype(np.float32)).view(B, T, MD).requires_grad_(True)
    out = blk(xt)
    dot = torch.from_numpy(dout.astype(np.float32)).view(B, T, MD)
    out.backward(dot)
    grads = {'dx': xt.grad.view(M, MD).numpy()}
    gq = blk.attn.q.weight.grad.T.numpy(); gk = blk.attn.k.weight.grad.T.numpy(); gv = blk.attn.v.weight.grad.T.numpy()
    dWqkv_ref = np.concatenate([gq, gk, gv], axis=1)
    dWo_ref = blk.attn.o.weight.grad.T.numpy()
    dW12_ref = blk.mlp.w12.weight.grad.T.numpy()
    dW3_ref = blk.mlp.w3.weight.grad.T.numpy()

    def cmp(name, a, b):
        err = float(np.abs(a - b).max()); rel = err / (float(np.abs(b).max()) + 1e-12)
        print('%-8s max_abs=%.3e rel=%.3e refmax=%.3e %s'
              % (name, err, rel, float(np.abs(b).max()), 'OK' if rel < 5e-2 else 'FAIL'))

    print('--- fwd (block contribution, NOT dominated by residual x) ---')
    # x2 - x 是块的真实贡献；直接比 x2 会被残差 x 盖住（两个残差项各带 ls=1e-2 缩放）。
    # 但块贡献只有 1.4e-2 量级，而 x2 以 fp16 存储（|x|~2.5 时 ulp ~1e-3），
    # 所以这里只能按【绝对误差】判定，相对误差没有意义。
    x2_ref = out.detach().view(M, MD).numpy()
    delta_ref = x2_ref - x.astype(np.float32)
    de = float(np.abs((x2_np - x.astype(np.float32)) - delta_ref).max())
    print('delta    max_abs=%.3e refmax=%.3e %s  (abs-only, fp16 storage floor ~1e-3)'
          % (de, float(np.abs(delta_ref).max()), 'OK' if de < 5e-3 else 'FAIL'))
    print('--- bwd ---')
    cmp('dx', dx_np, grads['dx'])
    cmp('dWqkv', dWq_np, dWqkv_ref)
    cmp('dWo', dWo_np, dWo_ref)
    cmp('dW12', dW12_np, dW12_ref)
    cmp('dW3', dW3_np, dW3_ref)


if __name__ == '__main__':
    main()



