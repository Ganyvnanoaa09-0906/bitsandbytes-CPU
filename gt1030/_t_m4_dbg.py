# -*- coding: utf-8 -*-
"""M4 debug: check logits buffer + CE kernel output."""
import sys, time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_model as MM

ckpt = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

dev = MM.V2ModelDev(ckpt)
t0 = time.time(); dev.forward(tokens); print('fwd %.2fs' % (time.time() - t0))
lg = dev.dev.down(dev.sc['logits'], (MM.M, dev.V), np.float16).astype(np.float32)
print('logits: min=%.3f max=%.3f mean=%.3f  finite=%s  row0[:6]=%s'
      % (lg.min(), lg.max(), lg.mean(), np.isfinite(lg).all(), np.round(lg[0, :6], 3)))
tgt = np.zeros(MM.M, np.int64)
msk = np.zeros(MM.M, np.float32)
for b in range(MM.B):
    r0 = b * MM.T
    tgt[r0:r0 + MM.T - 1] = tokens[b, 1:]
    msk[r0:r0 + MM.T - 1] = 1.0
m = lg - lg.max(1, keepdims=True)
p = np.exp(m); p /= p.sum(1, keepdims=True)
ce = -np.log(p[np.arange(MM.M), tgt] + 1e-12) * msk
print('numpy CE mean=%.6f  valid=%d' % (ce.sum() / msk.sum(), int(msk.sum())))
ic = dev.make_targets(tokens)
dev.dev.K['ce_softmax'](dev.dev.q, (MM.M,), (256,), dev.sc['logits'], dev.tgt, dev.msk,
                        dev.sc['dlogits'], dev.loss_row, np.int32(dev.V), np.float32(ic))
dev.dev.finish()
lrow = dev.dev.down(dev.loss_row, (MM.M,), np.float32)
print('kernel loss_row: sum=%.6f  nonzero rows=%d  first valid=%s'
      % (lrow.sum(), int((lrow != 0).sum()), np.round(lrow[:4], 4)))
dl = dev.dev.down(dev.sc['dlogits'], (MM.M, dev.V), np.float16).astype(np.float32)
ref_dl = (p - np.eye(dev.V)[tgt]) * msk[:, None] * ic
print('dlogits: max_abs=%.3e vs numpy %.3e -> rel=%.3e'
      % (np.abs(dl).max(), np.abs(ref_dl).max(),
         np.abs(dl - ref_dl).max() / (np.abs(ref_dl).max() + 1e-12)))
