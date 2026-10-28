# -*- coding: utf-8 -*-
"""M4 debug 4: 逐中间量与 torch 对拍（整机 10 层，看第 0 层和最终 logits）。"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy
import torch
import torch.nn.functional
import gt1030.ar_v2_model
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
ck = torch.load(ckpt, map_location='cpu', weights_only=False)
cfg = SmallImageConfigV2(**ck['cfg'])
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'], strict=True)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
xt = torch.from_numpy(tokens)

dev = MM.V2ModelDev(ckpt)
d = dev.dev
B, T, MD, FF, HD, NH = MM.B, MM.T, MM.MD, MM.FF, MM.HD, MM.NH
print('=== run 1: fresh process, first forward ===')
dev.forward(tokens)
lg_dev = d.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('logits absmax=%.4f mean=%.4f' % (np.abs(lg_dev).max(), lg_dev.mean()))
dev.forward(tokens)
lg_dev2 = d.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('logits absmax(2nd)=%.4f  identical to 1st: %s'
      % (np.abs(lg_dev2).max(), bool(np.allclose(lg_dev, lg_dev2))))
print('=== torch reference ===')
with torch.no_grad():
    x = m.embed(xt)                                   # (B,T,MD)
    xf = x
    b0 = m.blocks[0]
    h1 = b0.ln1(xf)
    # 手写 attention（走模块自身，便于拿中间量）
    q, k, v = b0.attn._qkv(h1, B, T)
    kk = k.repeat_interleave(b0.attn.nh // b0.attn.n_kv, dim=1)
    vv = v.repeat_interleave(b0.attn.nh // b0.attn.n_kv, dim=1)
    y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
    attn_t = b0.attn.o(y.transpose(1, 2).contiguous().view(B, T, NH * HD))
    x1_t = xf + b0.ls1 * attn_t
    h2_t = b0.ln2(x1_t)
    a2_t = b0.mlp.w12(h2_t)
    g_t = F.silu(a2_t[..., :FF]) * a2_t[..., FF:]
    y_t = b0.mlp.w3(g_t)
    x2_t = x1_t + b0.ls2 * y_t

def cmp(name, a, t):
    a = np.asarray(a, np.float32); t = np.asarray(t, np.float32)
    err = float(np.abs(a - t).max()); rel = err / (float(np.abs(t).max()) + 1e-12)
    print('%-10s dev_absmax=%.4f torch_absmax=%.4f  max_abs=%.3e rel=%.3e %s'
          % (name, float(np.abs(a).max()), float(np.abs(t).max()), err, rel,
             'OK' if rel < 5e-2 else 'FAIL'))

cmp('h1[0]', d.down(dev.layers[0].b['h1'], (MM.M, MD)), h1.view(MM.M, MD).numpy())
cmp('attn[0]', d.down(dev.sc['attn'], (MM.M, MD)), attn_t.view(MM.M, MD).numpy())
cmp('x1[0]', d.down(dev.layers[0].b['x1'], (MM.M, MD)), x1_t.view(MM.M, MD).numpy())
cmp('h2[0]', d.down(dev.layers[0].b['h2'], (MM.M, MD)), h2_t.view(MM.M, MD).numpy())
cmp('a2[0]', d.down(dev.layers[0].b['a2'], (MM.M, 2 * FF)), a2_t.view(MM.M, 2 * FF).numpy())
cmp('g[0]', d.down(dev.layers[0].b['g'], (MM.M, FF)), g_t.view(MM.M, FF).numpy())
print('note: x2[0] 没保存，看整机 logits')
with torch.no_grad():
    lg_t = m.forward_logits(xt, None)[1][0]
cmp('logits', lg_dev2, lg_t.view(MM.M, -1).numpy())
