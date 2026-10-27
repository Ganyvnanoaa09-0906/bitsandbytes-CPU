# -*- coding: utf-8 -*-
"""探针 32：1 层，定位 dWo 的 15% 误差来自 GEMM 还是输入（o_tok / dattn）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import torch.nn.functional as F
import gt1030.ar_v2_multi as MM
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = SmallImageConfigV2(**ck['cfg'])
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'], strict=True)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
xt = torch.from_numpy(tokens)
M, MD = MM.M, MM.MD

model = MM.Model(CKPT)
_ = model.forward(tokens)
model.backward(tokens)
blk = model.blocks[0]
d, b = blk.dev, blk.b

o_tok = d.down(b['o_tok'], (M, MD), np.float16).astype(np.float32)
dattn = d.down(b['dattn'], (M, MD), np.float16).astype(np.float32)
dWo_dev = d.down(blk.W['Wo'].grad, (MD, MD), np.float16).astype(np.float32) / MM.CE_SCALE

# torch 侧参考
xe = m.embed(xt).detach().requires_grad_(True)
blk0 = m.blocks[0]
xb = blk0(xe)
lgt = m.gen_head.lin(xb)
lgt = lgt[0] if isinstance(lgt, (tuple, list)) else lgt
loss = F.cross_entropy(lgt[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
dout = torch.autograd.grad(loss, xb, retain_graph=True)[0].view(M, MD)
ls1 = blk0.ls1
dattn_ref = (dout * ls1).detach().numpy()

def rel(a, bb):
    return float(np.abs(a - bb).max() / (np.abs(bb).max() + 1e-30))

print('dattn   rel=%.3e  devmax=%.4g refmax=%.4g' % (rel(dattn, dattn_ref),
                                                    np.abs(dattn).max(), np.abs(dattn_ref).max()))
# o_tok 参考
with torch.no_grad():
    h1 = blk0.ln1(xe.detach())
    q, k, v = blk0.attn._qkv(h1, MM.B, MM.T)
    kk = k.repeat_interleave(blk0.attn.nh // blk0.attn.n_kv, dim=1)
    vv = v.repeat_interleave(blk0.attn.nh // blk0.attn.n_kv, dim=1)
    y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)
    yt = y.transpose(1, 2).contiguous().view(M, blk0.attn.nh * blk0.attn.hd)
    o_tok_ref = yt.numpy()
print('o_tok   rel=%.3e' % rel(o_tok, o_tok_ref))
# device 侧 GEMM 自洽性
dWo_np = o_tok.T @ dattn
print('dWo(device GEMM) vs numpy(o_tok^T dattn) rel=%.3e' % rel(dWo_dev, dWo_np))
print('dWo(numpy 用 device 输入) vs torch rel=%.3e' % rel(dWo_np, dWo_dev))
# torch 的 dWo 参考
Wor = torch.autograd.grad(loss, blk0.attn.o.weight)[0].T.numpy()
print('dWo vs torch 参考 rel=%.3e  refmax=%.4g devmax=%.4g'
      % (rel(dWo_dev, Wor), np.abs(Wor).max(), np.abs(dWo_dev).max()))
# 用 torch 的输入重算
dWo_t = (torch.from_numpy(o_tok_ref).T @ torch.from_numpy(dattn_ref)).numpy()
print('numpy(torch 输入) vs torch 参考 rel=%.3e' % rel(dWo_t, Wor))
