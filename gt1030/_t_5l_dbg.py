# -*- coding: utf-8 -*-
"""探针 21：5 层 ckpt，dump logits / tgt / loss_row，定位 4.5e23 从哪来。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ckpt_5l.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

dev = MM.V2ModelDev(CKPT)
dev.embed(tokens)
dev.forward(tokens)
lg = dev.dev.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('logits: absmax=%.4g  nan=%d  inf=%d  mean=%.4g'
      % (np.abs(lg).max(), int(np.isnan(lg).sum()), int(np.isinf(lg).sum()), lg.mean()))
xf = dev.dev.down(dev.x_final, (MM.M, MM.MD), np.float16).astype(np.float32)
print('x_final: absmax=%.4g' % np.abs(xf).max())
wh = dev.dev.down(dev.head.w, (MM.MD, dev.V), np.float16).astype(np.float32)
print('head.w : absmax=%.4g' % np.abs(wh).max())
print('ref logits (numpy, 同输入同权重) absmax=%.4g'
      % np.abs(xf @ wh).max())
ic = dev.make_targets(tokens)
tgt = dev.dev.down(dev.tgt, (MM.M,), np.int32)
msk = dev.dev.down(dev.msk, (MM.M,), np.float32)
print('tgt: min=%d max=%d 前 6 个=%s' % (tgt.min(), tgt.max(), tgt[:6]))
print('mask: sum=%.0f  val at 1023=%d' % (msk.sum(), int(msk[1023])))
dev.dev.K['ce_softmax'](dev.dev.q, (MM.M,), (256,), dev.sc['logits'], dev.tgt, dev.msk,
                        dev.sc['dlogits'], dev.loss_row, np.int32(dev.V), np.float32(ic))
dev.dev.finish()
lrow = dev.dev.down(dev.loss_row, (MM.M,), np.float32)
print('loss_row: sum=%.6g  max=%.6g  非零个数=%d' % (lrow.sum(), lrow.max(), int((lrow != 0).sum())))
print('  lrow[:6]=', np.round(lrow[:6], 4))
# numpy 参考
m2 = lg - lg.max(1, keepdims=True)
p = np.exp(m2); p /= p.sum(1, keepdims=True)
ce = -np.log(p[np.arange(MM.M), np.clip(tgt, 0, dev.V - 1)] + 1e-12) * msk
print('numpy CE = %.6f' % (ce.sum() / msk.sum()))
dl = dev.dev.down(dev.sc['dlogits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('dlogits absmax=%.4g (期望 ~%.4g)' % (np.abs(dl).max(), (1.0 / msk.sum())))
