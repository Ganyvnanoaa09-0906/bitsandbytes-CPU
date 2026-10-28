# -*- coding: utf-8 -*-
"""探针 30：1 层多 context 模型，逐层比较 dx（模型返回的 embedding 梯度）与 torch。"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
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
import numpy as np
import torch
import torch.nn.functional as F
import gt1030.ar_v2_multi as MM
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = SmallImageConfigV2(**ck['cfg'])
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'], strict=True)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
xt = torch.from_numpy(tokens)

model = MM.Model(CKPT)
loss = model.forward(tokens)
dx_dev = model.backward(tokens)                 # (M,MD) fp16

lg = m.forward_logits(xt, None)[1][0]
lo = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
lo.backward()
P = dict(m.named_parameters())
dx_t = P['token_emb.weight'].grad       # 只用来对照 embed 之外的链路
# 直接拿 torch 的 block 输入梯度：用 embed 输出做叶子
xe = m.embed(xt).detach().requires_grad_(True)
xb = m.blocks[0](xe)
lgt = m.gen_head.lin(xb)
lgt = lgt[0] if isinstance(lgt, (tuple, list)) else lgt
F.cross_entropy(lgt[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1)).backward()
dx_ref = xe.grad.view(MM.M, MM.MD).numpy()
print('dx: max_abs=%.3e rel=%.3e' % (np.abs(dx_dev.astype(np.float32) - dx_ref).max(),
                                     np.abs(dx_dev.astype(np.float32) - dx_ref).max()
                                     / (np.abs(dx_ref).max() + 1e-12)))
# 顺带看 dW3 直接用手写的 block 反向（对照 M3 结论）
print('torch dW3 max=%.3e  device dW3 max=%.3e'
      % (np.abs(P['blocks.0.mlp.w3.weight'].grad).max(),
         np.abs(model.blocks[0].dev.down(model.blocks[0].W['W3'].grad,
                                        (MM.FF, MM.MD), np.float16).astype(np.float32)).max()))
