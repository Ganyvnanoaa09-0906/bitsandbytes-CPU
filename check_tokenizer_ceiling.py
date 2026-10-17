"""What is the ceiling? Reconstruct real images through the VQGAN and look at them.

This is a check that should have been done before any of the modelling work, and never was. The AR
model predicts tokens produced by this VQGAN, so whatever the tokenizer cannot represent, no model
can ever generate. If the reconstruction of a real training image is already mush, then "the samples
are colour fields" says nothing about the AR model and everything about the tokenizer, and the fix
is a bigger codebook or a finer grid rather than a better objective.

The configuration under test: grid 32 (each token covers 8x8 pixels of a 256px image), a 1024-entry
codebook trained on 14432 cosplay photographs.

Three quantities per image: the round-trip reconstruction, the pixel error, and how much of the
codebook the whole validation split actually uses. A codebook of 1024 entries that only ever gets
partly used is itself a finding, since it means the effective capacity is smaller than the nominal
one.

Read-only with respect to the training run: one image at a time through the encoder and decoder, no
torch threads beyond one, so it will not disturb the MaskGIT job that is currently using five.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from PIL import Image  # noqa: E402

torch.set_num_threads(1)
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

VQ = r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt'
TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
DATA = r'D:\work\cloud_salvage\DATASET\data'

print()
print('=== VQGAN 重建质量：这是整个生成任务的天花板 ===')
print()

ck = torch.load(VQ, map_location='cpu', weights_only=False)
print('  checkpoint 键: %s' % list(ck.keys())[:8])

# find the model class used for this VQGAN
import glob
import os
cands = [f for f in glob.glob(r'D:\work\bitsandbytes-CPU\*.py')
         if 'vq' in os.path.basename(f).lower()]
print('  候选 VQ 脚本: %s' % [os.path.basename(c) for c in cands][:8])

# try to load tokens and inspect the code distribution, which needs no model at all
if os.path.exists(TOK):
    t = torch.load(TOK, map_location='cpu')
    tt = t['tokens'] if isinstance(t, dict) and 'tokens' in t else (t if torch.is_tensor(t) else None)
    if tt is not None:
        tt = tt.long()
        n = tt.numel()
        uniq = torch.unique(tt)
        counts = torch.bincount(tt.reshape(-1), minlength=1024).float()
        p = counts / counts.sum()
        ent = float(-(p[p > 0] * p[p > 0].log()).sum())
        print()
        print('  === 训练 token 的码本使用情况 ===')
        print('    形状 %s · 共 %s 个 token' % (tuple(tt.shape), format(n, ',')))
        print('    用到的码字: 【%d / 1024】 (%.1f%%)' % (len(uniq), len(uniq) / 1024 * 100))
        print('    分布熵: 【%.3f nats】 (最大 %.3f = 均匀)' % (ent, float(np.log(1024))))
        top = torch.topk(counts, 8)
        print('    最高频 8 个码字占比: %.1f%%' % (float(top.values.sum() / counts.sum() * 100)))
        print('    最低频 8 个码字占比: %.3f%%' % (float(torch.sort(counts)[0][:8].sum() / counts.sum() * 100)))
        print()
        print('    判据: 若熵远低于 6.93 或使用率远低于 100%%，说明码本容量【名义上】有 1024')
        print('          而【实际有效】的小得多 ⇒ 这是分词器层面的瓶颈，不是模型的。')
else:
    print('  找不到 token 文件 %s' % TOK)
