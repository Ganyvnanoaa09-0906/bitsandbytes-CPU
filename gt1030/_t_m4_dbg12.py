# -*- coding: utf-8 -*-
"""探针 12：逐算子前向第 0 层，每一步都回读，找出第一个变 0 / 变错的量。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM
import gt1030.full_gpu_layer as L

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

model = MM.V2ModelDev(ckpt)
d, K = model.dev, model.dev.K
B, T, MD, FF, HD, NH, NKV, BH, M, WROW = (MM.B, MM.T, MM.MD, MM.FF, MM.HD, MM.NH,
                                          MM.NKV, MM.BH, MM.M, MM.WROW)
x_host = model.embed(tokens)
print('host embed absmax=%.4f' % np.abs(x_host).max())
xa = d.down(model.xa, (M, MD), np.float16).astype(np.float32)
print('device xa    absmax=%.4f mean=%.6f' % (np.abs(xa).max(), xa.mean()))

ly = model.layers[0]
b = ly.b
K['rms_stats'](d.q, (M,), (64,), model.xa, model.ss, np.int32(M), np.int32(MD))
d.finish()
ss = d.down(model.ss, (M,), np.float32)
print('ss           min=%.4f max=%.4f mean=%.4f (期望 ≈ E[x^2] ~ 5)' % (ss.min(), ss.max(), ss.mean()))
K['rms_apply'](d.q, (M * MD,), (256,), model.xa, ly.ln1, model.ss, b['h1'], np.int32(M), np.int32(MD))
d.finish()
h1 = d.down(b['h1'], (M, MD), np.float16).astype(np.float32)
print('h1           absmax=%.4f mean=%.6f' % (np.abs(h1).max(), h1.mean()))
ln1 = d.down(ly.ln1, (MD,), np.float16).astype(np.float32)
print('ln1(d)       min=%.4f max=%.4f' % (ln1.min(), ln1.max()))
L.hgemm_nn(b['h1'], ly.W['Wqkv'].w, model.sc['qkv'], M, WROW, MD)
d.finish()
qkv = d.down(model.sc['qkv'], (M, WROW), np.float16).astype(np.float32)
print('qkv          absmax=%.4f mean=%.6f' % (np.abs(qkv).max(), qkv.mean()))
import torch as th
h1t = th.from_numpy(h1)
W = th.from_numpy(d.down(ly.W['Wqkv'].w, (MD, WROW), np.float16).astype(np.float32))
qkv_ref = h1t @ W
print('qkv (torch 同 h1) absmax=%.4f  与 device qkv 的 rel=%.3e'
      % (qkv_ref.abs().max(), float((th.from_numpy(qkv) - qkv_ref).abs().max() / qkv_ref.abs().max())))
