# -*- coding: utf-8 -*-
"""探针 24：CE kernel 逐行对拍（device loss_row vs numpy 用同一份 device logits 算的 CE）。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ckpt_1l.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

dev = MM.V2ModelDev(CKPT)
d = dev.dev
dev.forward(tokens)
lg = d.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
ic = dev.make_targets(tokens)
d.K['ce_softmax'](d.q, (MM.M,), (256,), dev.sc['logits'], dev.tgt, dev.msk,
                  dev.sc['dlogits'], dev.loss_row, np.int32(dev.V), np.float32(ic))
d.finish()
lrow = d.down(dev.loss_row, (MM.M,), np.float32)
tgt = d.down(dev.tgt, (MM.M,), np.int32)
msk = d.down(dev.msk, (MM.M,), np.float32)

mx = lg.max(1, keepdims=True)
e = np.exp(lg - mx)
p = e / e.sum(1, keepdims=True)
ce_ref = -np.log(p[np.arange(MM.M), np.clip(tgt, 0, dev.V - 1)] + 1e-12) * msk
print('kernel sum=%.6f  numpy sum=%.6f' % (lrow.sum(), ce_ref.sum()))
print('kernel mean=%.6f numpy mean=%.6f' % (lrow.sum() / msk.sum(), ce_ref.sum() / msk.sum()))
print('row-wise 前 8 行:')
for r in range(8):
    print('  r=%4d tgt=%6d mask=%.0f  kernel=%10.5f numpy=%10.5f' %
          (r, tgt[r], msk[r], lrow[r], ce_ref[r]))
print('kernel 非零但 numpy 为 0 的行:', int(((lrow != 0) & (msk == 0)).sum()))
print('numpy 非零但 kernel 为 0 的行:', int(((ce_ref != 0) & (lrow == 0)).sum()))
print('loss_row 尾部 4 行:', np.round(lrow[-4:], 4), 'tgt', tgt[-4:], 'mask', msk[-4:])
