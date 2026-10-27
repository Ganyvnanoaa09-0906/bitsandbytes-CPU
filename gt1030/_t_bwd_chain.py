# -*- coding: utf-8 -*-
"""探针 31：1 层模型，逐量对拍反向链：dout(block 输入梯度) / dy / dW3 / dg。"""
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
M, MD, FF = MM.M, MM.MD, MM.FF

model = MM.Model(CKPT)
_ = model.forward(tokens)
blk = model.blocks[0]
d, b = blk.dev, blk.b
model.backward(tokens)

# --- torch 侧：拿 block 输出梯度、dy、dg、dW3 ---
xe = m.embed(xt).detach().requires_grad_(True)
blk0 = m.blocks[0]
xb = blk0(xe)
lgt = m.gen_head.lin(xb)
lgt = lgt[0] if isinstance(lgt, (tuple, list)) else lgt
loss = F.cross_entropy(lgt[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
dout_t = torch.autograd.grad(loss, xb)[0].view(M, MD)
# dy = dout * ls2, dg = dy @ W3^T, dW3 = g^T dy
W3 = blk0.mlp.w3.weight                      # (MD, FF)
ls2 = blk0.ls2
dy_t = dout_t * ls2
# 复算 a2/g 以便拿 g
with torch.no_grad():
    h1 = blk0.ln1(xe.detach())
    q = blk0.attn.q(h1) if False else None
dW3_t = P = None
# 用 autograd 直接拿 W3.grad（已由 loss.backward 算好？没有，需要 retain）
loss2 = None
del dout_t, dy_t

def gv(x):
    return x.detach().numpy() if torch.is_tensor(x) else np.asarray(x)

# 只比较 dout（device 的 din_a 在 backward 里被当作 dout）
din_dev = d.down(b['din_a'], (M, MD), np.float16).astype(np.float32)
# torch 的 dout
xe2 = m.embed(xt).detach().requires_grad_(True)
xb2 = m.blocks[0](xe2)
lg2 = m.gen_head.lin(xb2)
lg2 = lg2[0] if isinstance(lg2, (tuple, list)) else lg2
l2 = F.cross_entropy(lg2[:, :-1].reshape(-1, cfg.vocab_size), xt[:, 1:].reshape(-1))
dout_ref = torch.autograd.grad(l2, xb2)[0].view(M, MD).numpy()
print('dout  : rel=%.3e  devmax=%.4g refmax=%.4g' %
      (np.abs(din_dev - dout_ref).max() / (np.abs(dout_ref).max() + 1e-12),
       np.abs(din_dev).max(), np.abs(dout_ref).max()))
dy_dev = d.down(b['dy'], (M, MD), np.float16).astype(np.float32)
ls2n = blk0.ls2.detach().numpy()
dy_ref = dout_ref * ls2n
print('dy    : rel=%.3e' % (np.abs(dy_dev - dy_ref).max() / (np.abs(dy_ref).max() + 1e-12)))
dg_dev = d.down(b['dg'], (M, FF), np.float16).astype(np.float32)
W3f = blk0.mlp.w3.weight.detach().numpy().T          # (FF, MD)
dg_ref = dy_ref @ W3f.T
print('dg    : rel=%.3e  devmax=%.4g refmax=%.4g' %
      (np.abs(dg_dev - dg_ref).max() / (np.abs(dg_ref).max() + 1e-12),
       np.abs(dg_dev).max(), np.abs(dg_ref).max()))
g_dev = d.down(b['g'], (M, FF), np.float16).astype(np.float32)
dW3_dev = d.down(blk.W['W3'].grad, (FF, MD), np.float16).astype(np.float32)
dW3_ref = g_dev.T @ dy_ref
print('dW3   : dev vs g_dev^T dy_ref rel=%.3e  devmax=%.4g' %
      (np.abs(dW3_dev - dW3_ref).max() / (np.abs(dW3_ref).max() + 1e-12), np.abs(dW3_dev).max()))
print('  ratio dev/ref = %.4f' % (np.abs(dW3_dev).max() / (np.abs(dW3_ref).max() + 1e-30)))
