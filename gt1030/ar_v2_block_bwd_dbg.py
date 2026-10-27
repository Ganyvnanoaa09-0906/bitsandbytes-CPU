# -*- coding: utf-8 -*-
"""M3 debug: locate the wrong weight-gradient by comparing every intermediate."""
from __future__ import annotations
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import torch.nn.functional as F
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


def rms_t(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


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
    for p in (blk.ln1.w, blk.ln2.w, blk.attn.q_norm.w, blk.attn.k_norm.w, blk.ls1, blk.ls2):
        p.requires_grad_(False)
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

    dWqkv, dWo, dW12, dW3 = up(WQKV), up(WO), up(W12), up(W3)
    # rms_* kernels declare w as `half*` -> must upload fp16 (float32 shows up as
    # alternating 0.0/1.875 garbage halves). qk_norm kernels below are float*.
    uq, uk = up(qw), up(kw)
    uln1, uln2 = up(ln1.astype(np.float16)), up(ln2.astype(np.float16))
    uls1, uls2 = up(ls1), up(ls2)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, MD)) * 0.5).astype(np.float16)
    xb = up(x)
    h1, qkv = buf(M * MD * 2), buf(M * WROW * 2)
    Q, Kb, Vb = buf(BH * T * HD * 2), buf(BH * T * HD * 2), buf(BH * T * HD * 2)
    Qn, Kn = buf(BH * T * HD * 2), buf(BH * T * HD * 2)
    S, P, O = buf(BH * T * T * 2), buf(BH * T * T * 2), buf(BH * T * HD * 2)
    o_tok, attn, x1, h2 = buf(M * MD * 2), buf(M * MD * 2), buf(M * MD * 2), buf(M * MD * 2)
    a2, gbuf, y = buf(M * 2 * FF * 2), buf(M * FF * 2), buf(M * MD * 2)
    x2 = buf(M * MD * 2)
    ss, sc = buf(M * 4), buf(M * 4)

    # ---------- forward ----------
    K['rms_stats'](q, (M,), (64,), xb, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), xb, uln1, ss, h1, np.int32(M), np.int32(MD))
    L.hgemm_nn(h1, dWqkv, qkv, M, WROW, MD)
    K['split_q'](q, (BH * T * HD,), (256,), qkv, Q, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV))
    K['split_kv'](q, (BH * T * HD,), (256,), qkv, Kb, Vb, np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV), np.int32(B))
    K['qk_norm_fwd'](q, (BH * T,), None, Q, uq, Qn, np.int32(BH * T), np.int32(HD))
    K['qk_norm_fwd'](q, (BH * T,), None, Kb, uk, Kn, np.int32(BH * T), np.int32(HD))
    A.hgemm_batched(Qn, Kn, S, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_causal'](q, (T * 256, BH), (256, 1), S, P, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, Vb, O, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    K['merge_o'](q, (BH * T * HD,), (256,), O, o_tok, np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
    L.hgemm_nn(o_tok, dWo, attn, M, MD, MD)
    K['scale_vec'](q, (M * MD,), (256,), attn, uls1, x1, np.int32(M), np.int32(MD))
    K['add_vec'](q, (M * MD,), (256,), xb, x1, x1, np.int32(M * MD))
    K['rms_stats'](q, (M,), (64,), x1, ss, np.int32(M), np.int32(MD))
    K['rms_apply'](q, (M * MD,), (256,), x1, uln2, ss, h2, np.int32(M), np.int32(MD))
    L.hgemm_nn(h2, dW12, a2, M, 2 * FF, MD)
    K['split_silu'](q, (M * FF,), (256,), a2, gbuf, np.int32(M), np.int32(FF))
    L.hgemm_nn(gbuf, dW3, y, M, MD, FF)
    K['scale_vec'](q, (M * MD,), (256,), y, uls2, y, np.int32(M), np.int32(MD))
    K['add_vec'](q, (M * MD,), (256,), x1, y, x2, np.int32(M * MD))

    # ---------- backward ----------
    dout = (rng.standard_normal((M, MD)) * 0.5).astype(np.float16)
    doutb = up(dout)
    dW3b, dW12b, dWob, dWqkvb = (up(np.zeros_like(W3)), up(np.zeros_like(W12)),
                                 up(np.zeros_like(WO)), up(np.zeros_like(WQKV)))
    dy = buf(M * MD * 2); dg = buf(M * FF * 2); da2 = buf(M * 2 * FF * 2)
    dh2 = buf(M * MD * 2); dx1_mlp = buf(M * MD * 2)
    dattn = buf(M * MD * 2); dO_tok = buf(M * MD * 2); dO = buf(BH * T * HD * 2)
    dP = buf(BH * T * T * 2); dS = buf(BH * T * T * 2)
    dQ = buf(BH * T * HD * 2); dK = buf(BH * T * HD * 2); dV = buf(BH * T * HD * 2)
    dqkv = buf(M * WROW * 2); dh1 = buf(M * MD * 2); dx_attn = buf(M * MD * 2)
    dx = buf(M * MD * 2); dQraw = buf(BH * T * HD * 2); dKraw = buf(BH * T * HD * 2)

    K['scale_vec'](q, (M * MD,), (256,), doutb, uls2, dy, np.int32(M), np.int32(MD))
    L.hgemm_tn(gbuf, dy, dW3b, FF, MD, M)          # fixed order
    L.hgemm_nt(dy, dW3, dg, M, FF, MD)
    K['split_silu_bwd'](q, (M * FF,), (256,), a2, dg, da2, np.int32(M), np.int32(FF))
    L.hgemm_tn(h2, da2, dW12b, MD, 2 * FF, M)
    L.hgemm_nt(da2, dW12, dh2, M, MD, 2 * FF)
    K['rms_bwd_stats'](q, (M,), (64,), x1, dh2, uln2, ss, sc, np.int32(M), np.int32(MD))
    K['rms_bwd_apply'](q, (M * MD,), (256,), x1, dh2, uln2, ss, sc, dx1_mlp, np.int32(M), np.int32(MD))
    K['scale_vec'](q, (M * MD,), (256,), doutb, uls1, dattn, np.int32(M), np.int32(MD))
    L.hgemm_tn(o_tok, dattn, dWob, MD, MD, M)
    L.hgemm_nt(dattn, dWo, dO_tok, M, MD, MD)
    K['split_dout'](q, (BH * T * HD,), (256,), dO_tok, dO, np.int32(T), np.int32(HD), np.int32(NH), np.int32(B))
    A.hgemm_batched(dO, Vb, dP, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)
    K['softmax_bwd_causal'](q, (T * 256, BH), (256, 1), P, dP, dS, np.int32(T), np.float32(scale))
    A.hgemm_batched(P, dO, dV, T, HD, T, A.YES, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    # S = Qn Kn^T -> dQ = dS @ Kn, dK = dS^T @ Qn  (NORMALIZED, not raw!)
    A.hgemm_batched(dS, Kn, dQ, T, HD, T, A.NO, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    A.hgemm_batched(dS, Qn, dK, T, HD, T, A.YES, A.NO, T, HD, HD, T * T, T * HD, T * HD)
    K['qk_norm_dx'](q, (BH * T,), None, Q, uq, dQ, dQraw, np.int32(BH * T), np.int32(HD))
    K['qk_norm_dx'](q, (BH * T,), None, Kb, uk, dK, dKraw, np.int32(BH * T), np.int32(HD))
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
        cl.enqueue_copy(q, a, b); q.finish(); return a.astype(np.float32)

    dev = dict(
        dx=dn(dx, (M, MD)), dy=dn(dy, (M, MD)), dg=dn(dg, (M, FF)), da2=dn(da2, (M, 2 * FF)),
        dh2=dn(dh2, (M, MD)), dx1_mlp=dn(dx1_mlp, (M, MD)), dattn=dn(dattn, (M, MD)),
        dO_tok=dn(dO_tok, (M, MD)), dO=dn(dO, (B, NH, T, HD)), dP=dn(dP, (B, NH, T, T)),
        dS=dn(dS, (B, NH, T, T)), dQ=dn(dQ, (B, NH, T, HD)), dK=dn(dK, (B, NH, T, HD)),
        dV=dn(dV, (B, NH, T, HD)), dqkv=dn(dqkv, (M, WROW)), dh1=dn(dh1, (M, MD)),
        dx_attn=dn(dx_attn, (M, MD)),
        dWqkv=dn(dWqkvb, WQKV.shape), dWo=dn(dWob, WO.shape),
        dW12=dn(dW12b, W12.shape), dW3=dn(dW3b, W3.shape),
        h1=dn(h1, (M, MD)), qkv=dn(qkv, (M, WROW)), o_tok=dn(o_tok, (M, MD)),
        attn=dn(attn, (M, MD)), x1=dn(x1, (M, MD)), h2=dn(h2, (M, MD)),
        a2=dn(a2, (M, 2 * FF)), g=dn(gbuf, (M, FF)), y=dn(y, (M, MD)),
        Q=dn(Q, (B, NH, T, HD)), Kb=dn(Kb, (B, NH, T, HD)), Vb=dn(Vb, (B, NH, T, HD)),
        Qn=dn(Qn, (B, NH, T, HD)), Kn=dn(Kn, (B, NH, T, HD)), P=dn(P, (B, NH, T, T)),
        O=dn(O, (B, NH, T, HD)),
    )

    # ---------- torch reference (same forward, fp32, grads of everything) ----------
    tt = lambda a, r=True: torch.tensor(np.asarray(a, dtype=np.float32), requires_grad=r)
    xt = tt(x, True); tWQKV = tt(WQKV); tWO = tt(WO); tW12 = tt(W12); tW3 = tt(W3)
    tln1 = torch.tensor(ln1); tln2 = torch.tensor(ln2); tls1 = torch.tensor(ls1); tls2 = torch.tensor(ls2)
    tqw = torch.tensor(qw); tkw = torch.tensor(kw)

    h1t = rms_t(xt, tln1); h1t.retain_grad()
    qkvt = h1t @ tWQKV; qkvt.retain_grad()
    qr = qkvt[:, :MD].view(B, T, NH, HD).permute(0, 2, 1, 3); qr.retain_grad()
    kr = qkvt[:, MD:MD + NKV * HD].view(B, T, NKV, HD).permute(0, 2, 1, 3); kr.retain_grad()
    vr = qkvt[:, MD + NKV * HD:].view(B, T, NKV, HD).permute(0, 2, 1, 3); vr.retain_grad()
    qn = rms_t(qr, tqw); qn.retain_grad()
    kn = rms_t(kr, tkw); kn.retain_grad()
    kn = kn.repeat_interleave(NH // NKV, dim=1)
    vv = vr.repeat_interleave(NH // NKV, dim=1)
    St = qn @ kn.transpose(-1, -2); St.retain_grad()
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    Pt = torch.softmax(St.masked_fill(~mask, float('-inf')) * scale, dim=-1); Pt.retain_grad()
    Ot = Pt @ vv; Ot.retain_grad()
    ot = Ot.permute(0, 2, 1, 3).reshape(M, MD); ot.retain_grad()
    at = ot @ tWO; at.retain_grad()
    x1t = xt + tls1 * at; x1t.retain_grad()
    h2t = rms_t(x1t, tln2); h2t.retain_grad()
    a2t = h2t @ tW12; a2t.retain_grad()
    gt = F.silu(a2t[:, :FF]) * a2t[:, FF:]; gt.retain_grad()
    yt = gt @ tW3; yt.retain_grad()
    x2t = x1t + tls2 * yt
    dot_t = torch.tensor(dout.astype(np.float32))
    leaves = [xt, tWQKV, tWO, tW12, tW3]
    gr = torch.autograd.grad(x2t, leaves + [h1t, qkvt, yt, gt, a2t, h2t, x1t, at, ot, Ot, Pt, St, qn, kn, qr, kr],
                             dot_t, retain_graph=True)
    (gx, gWQKV, gWO, gW12, gW3), g_rest = gr[:5], gr[5:]
    gmap = dict(zip(['h1', 'qkv', 'y', 'g', 'a2', 'h2', 'x1', 'attn', 'o_tok', 'O', 'P', 'S',
                     'Qn', 'Kn', 'Q', 'K'],
                    [t.numpy() for t in g_rest]))

    def cmp(name, d, t):
        d = np.asarray(d, np.float32); t = np.asarray(t, np.float32)
        if d.shape != t.shape:
            print('%-10s SHAPE %s vs %s' % (name, d.shape, t.shape)); return
        err = float(np.abs(d - t).max()); rel = err / (float(np.abs(t).max()) + 1e-12)
        print('%-8s max_abs=%.3e rel=%.3e refmax=%.3e devmax=%.3e %s'
              % (name, err, rel, float(np.abs(t).max()), float(np.abs(d).max()),
                 'OK' if rel < 5e-2 else 'FAIL'))

    print('--- forward parity ---')
    for nm, key in (('h1', 'h1'), ('qkv', 'qkv'), ('o_tok', 'o_tok'), ('attn', 'attn'),
                    ('x1', 'x1'), ('h2', 'h2'), ('a2', 'a2'), ('g', 'g'), ('y', 'y')):
        pass
    cmp('h1f', dev['h1'], h1t.detach().numpy())
    cmp('qkvf', dev['qkv'], qkvt.detach().numpy())
    cmp('Qnf', dev['Qn'], qn.detach().numpy())
    cmp('Knf', dev['Kn'], kn.detach().numpy())
    cmp('Pf', dev['P'], Pt.detach().numpy())
    cmp('Of', dev['O'], Ot.detach().numpy())
    cmp('otokf', dev['o_tok'], ot.detach().numpy())
    cmp('attnf', dev['attn'], at.detach().numpy())
    cmp('x1f', dev['x1'], x1t.detach().numpy())
    cmp('h2f', dev['h2'], h2t.detach().numpy())
    cmp('a2f', dev['a2'], a2t.detach().numpy())
    cmp('gf', dev['g'], gt.detach().numpy())
    cmp('yf', dev['y'], (tls2 * yt).detach().numpy())
    print('--- backward ---')
    cmp('dx', dev['dx'], gx.numpy())
    cmp('dy', dev['dy'], gmap['y'])
    cmp('dg', dev['dg'], gmap['g'])
    cmp('da2', dev['da2'], gmap['a2'])
    cmp('dh2', dev['dh2'], gmap['h2'])
    cmp('dattn', dev['dattn'], gmap['attn'])
    cmp('dO_tok', dev['dO_tok'], gmap['o_tok'])
    cmp('dO', dev['dO'], gmap['O'])
    cmp('dP', dev['dP'], gmap['P'])
    cmp('dS', dev['dS'], gmap['S'])
    cmp('dQ', dev['dQ'], gmap['Qn'])
    cmp('dK', dev['dK'], gmap['Kn'])
    cmp('dqkv', dev['dqkv'], gmap['qkv'])
    cmp('dh1', dev['dh1'], gmap['h1'])
    print('--- weights ---')
    cmp('dWqkv', dev['dWqkv'], gWQKV.numpy())
    cmp('dWo', dev['dWo'], gWO.numpy())
    cmp('dW12', dev['dW12'], gW12.numpy())
    cmp('dW3', dev['dW3'], gW3.numpy())

    # also compare against the real module autograd
    xt2 = torch.tensor(x.astype(np.float32)).view(B, T, MD).requires_grad_(True)
    out = blk(xt2)
    out.backward(torch.tensor(dout.astype(np.float32)).view(B, T, MD))
    print('--- vs module autograd ---')
    cmp('dx@blk', dev['dx'], xt2.grad.view(M, MD).numpy())
    ref = np.concatenate([blk.attn.q.weight.grad.T.numpy(), blk.attn.k.weight.grad.T.numpy(),
                          blk.attn.v.weight.grad.T.numpy()], axis=1)
    cmp('dWqkv@blk', dev['dWqkv'], ref)
    cmp('dWo@blk', dev['dWo'], blk.attn.o.weight.grad.T.numpy())
    cmp('dW12@blk', dev['dW12'], blk.mlp.w12.weight.grad.T.numpy())
    cmp('dW3@blk', dev['dW3'], blk.mlp.w3.weight.grad.T.numpy())


if __name__ == '__main__':
    main()

