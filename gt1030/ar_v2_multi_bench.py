# -*- coding: utf-8 -*-
"""M4 吞吐测量：device-only 的 fwd / bwd / opt 各多少秒，并和 CPU 对比。"""
import sys
import time
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'
tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

t0 = time.time()
model = MM.Model(CKPT)
print('build %.1fs  arena used %.0f MB' % (time.time() - t0, model.dev._off / 1e6))

# 预热（首次会触发 CLBlast 调优）
t0 = time.time(); model.forward(tokens); print('fwd#1 %.2fs (含调优)' % (time.time() - t0))
model.backward(tokens)
model.opt_step(5e-5, 1)

for tag in ('warm2',):
    t0 = time.time(); l = model.forward(tokens); t_f = time.time() - t0
    t0 = time.time(); model.backward(tokens); t_b = time.time() - t0
    t0 = time.time(); gn = model.opt_step(5e-5, 2); t_o = time.time() - t0
    print('%s: fwd=%.3fs bwd=%.3fs opt=%.3fs total=%.3fs  loss=%.4f gnorm=%.4f'
          % (tag, t_f, t_b, t_o, t_f + t_b + t_o, l, gn))
    # 再来三次取中位
    fs, bs, os_ = [], [], []
    for _ in range(3):
        t0 = time.time(); model.forward(tokens); fs.append(time.time() - t0)
        t0 = time.time(); model.backward(tokens); bs.append(time.time() - t0)
        t0 = time.time(); model.opt_step(5e-5, 3); os_.append(time.time() - t0)
    print('median: fwd=%.3f bwd=%.3f opt=%.3f total=%.3f'
          % (np.median(fs), np.median(bs), np.median(os_),
             np.median(fs) + np.median(bs) + np.median(os_)))
    print('tokens/s = %.0f' % (MM.M / (np.median(fs) + np.median(bs) + np.median(os_))))
