# -*- coding: utf-8 -*-
"""M4 剖面：把一步拆成 GEMM / attention / 小 kernel / 数据传输，看时间花在哪。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A
import gt1030.full_gpu_layer as L

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
model = MM.Model(CKPT)
model.forward(tokens)          # 预热 + 调优
model.backward(tokens)
d = model.dev
q, K = d.q, d.K
M, MD, FF, T, HD, NH, NKV, BH, WROW = (MM.M, MM.MD, MM.FF, MM.T, MM.HD, MM.NH, MM.NKV,
                                       MM.BH, MM.WROW)
blk = model.blocks[0]
b = blk.b


def tm(tag, fn, n=10):
    fn()
    d.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    d.finish()
    print('%-28s %7.2f ms/次  (x%d)' % (tag, (time.time() - t0) / n * 1000, n))


# 1) 大 GEMM 4 种
tm('GEMM Wqkv (M x 1024 x 512)', lambda: L.hgemm_nn(b['h1'], blk.W['Wqkv'].w, b['qkv'],
                                                   M, WROW, MD))
tm('GEMM Wo   (M x 512 x 512)', lambda: L.hgemm_nn(b['o_tok'], blk.W['Wo'].w, b['attn'],
                                                  M, MD, MD))
tm('GEMM W12  (M x 2736 x 512)', lambda: L.hgemm_nn(b['h2'], blk.W['W12'].w, b['a2'],
                                                    M, 2 * FF, MD))
tm('GEMM W3   (M x 512 x 1368)', lambda: L.hgemm_nn(b['g'], blk.W['W3'].w, b['y'],
                                                    M, MD, FF))
# 2) attention
tm('bGEMM S = Qn Kn^T (16x T T 64)',
   lambda: A.hgemm_batched(b['Qn'], b['Kn'], b['S'], T, T, HD, A.NO, A.YES,
                           HD, HD, T, T * HD, T * HD, T * T))
tm('softmax_causal (T=1024)',
   lambda: K['softmax_causal'](q, (T * 256, BH), (256, 1), b['S'], b['P'],
                               np.int32(T), np.float32(MM.scale)))
tm('bGEMM O = P Vb (16x T 64 T)',
   lambda: A.hgemm_batched(b['P'], b['Vb'], b['O'], T, HD, T, A.NO, A.NO,
                           T, HD, HD, T * T, T * HD, T * HD))
# 3) 小 kernel
tm('rms_stats + rms_apply',
   lambda: (K['rms_stats'](q, (M,), (64,), b['xin'], b['ss'], np.int32(M), np.int32(MD)),
            K['rms_apply'](q, (M * MD,), (256,), b['xin'], blk.ln1, b['ss'], b['h1'],
                           np.int32(M), np.int32(MD))))
tm('split_q + split_kv',
   lambda: (K['split_q'](q, (BH * T * HD,), (256,), b['qkv'], b['Qraw'],
                         np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV)),
            K['split_kv'](q, (BH * T * HD,), (256,), b['qkv'], b['Kraw'], b['Vb'],
                          np.int32(T), np.int32(HD), np.int32(NH), np.int32(NKV),
                          np.int32(MM.B))))
tm('qk_norm x2',
   lambda: (K['qk_norm_fwd'](q, (BH * T,), None, b['Qraw'], blk.qw, b['Qn'],
                             np.int32(BH * T), np.int32(HD)),
            K['qk_norm_fwd'](q, (BH * T,), None, b['Kraw'], blk.kw, b['Kn'],
                             np.int32(BH * T), np.int32(HD))))
tm('merge_o', lambda: K['merge_o'](q, (BH * T * HD,), (256,), b['O'], b['o_tok'],
                                   np.int32(T), np.int32(HD), np.int32(NH), np.int32(MM.B)))
tm('split_silu', lambda: K['split_silu'](q, (M * FF,), (256,), b['a2'], b['g'],
                                         np.int32(M), np.int32(FF)))
tm('scale_vec + add_vec', lambda: (
    K['scale_vec'](q, (M * MD,), (256,), b['attn'], blk.ls1, b['x1'], np.int32(M), np.int32(MD)),
    K['add_vec'](q, (M * MD,), (256,), b['xin'], b['x1'], b['x1'], np.int32(M * MD))))
# 4) 层间 x 传输
xh = np.zeros((M, MD), np.float16)


def xfer():
    import pyopencl as cl
    cl.enqueue_copy(q, b['xin'], xh, is_blocking=True)
    d.down(b['xout'], (M, MD), np.float16)


tm('层间 x 传输 (up 2MB + down 2MB)', xfer)
