# -*- coding: utf-8 -*-
"""bench_ar_batch.py - 标定 batch size 对训练吞吐的影响。

batch=1 时每个 matmul 只有 (T,512)@(512,2048)，矩阵小到喂不饱缓存，
CPU 虽然 100% 但有效算力很低。本脚本实测不同 batch 下的
  样本/秒 与 token/秒，找出真正的吞吐拐点。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from scripts.lib.small_image_model import SmallImageConfig, SmallARImageModel
import bitsandbytes as bnb

THREADS = int(os.environ.get('BENCH_THREADS', '6'))
torch.set_num_threads(THREADS)
torch.backends.mkldnn.enabled = True

tokens = torch.load(r'D:\work\bitsandbytes-CPU\.autopilot\tokens_pretrain.pt', map_location='cpu')
N = tokens.shape[0]
print('tokens %s, threads=%d' % (tuple(tokens.shape), THREADS))
print()
print('%6s %5s %10s %12s %12s %10s' % ('batch', 'T', 'ms/step', '样本/秒', 'token/秒', '峰值内存GB'))
print('-' * 66)

for T in (64, 256):
    for B in (1, 2, 4, 8, 16):
        cfg = SmallImageConfig(d_model=512, n_layer=10, n_head=8, ffn_dim=2048, total_tokens=256)
        model = SmallARImageModel(cfg)
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
        text = torch.zeros(B, cfg.text_dim)
        ITERS = 6
        try:
            for it in range(ITERS + 2):          # 前 2 步热身不计
                idx = torch.randint(0, N, (B,))
                x = tokens[idx, :T].long()
                t0 = time.perf_counter()
                th, ge = model(x, text)
                gen = F.cross_entropy(ge[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
                loss = gen
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                dt = time.perf_counter() - t0
                if it >= 2:
                    pass
                if it == 2:
                    samples = []
                if it >= 2:
                    samples.append(dt)
            ms = sum(samples) / len(samples) * 1000
            print('%6d %5d %10.1f %12.1f %12.0f' % (B, T, ms, B / (ms / 1000), B * T / (ms / 1000)))
        except Exception as e:
            print('%6d %5d  FAIL %s' % (B, T, type(e).__name__))
        del model, opt
print()
print('注: PyTorch 的 AdamW8bit 在这里是 CPU 实现，已计入耗时。')
