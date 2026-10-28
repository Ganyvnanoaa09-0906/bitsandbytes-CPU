# -*- coding: utf-8 -*-
"""diag_ms_ema.py — 为什么本地 dead=0、云端 dead=990？（直接测，不推理）

打印：self.training / update_ema / ema_count 的统计 / 低于阈值的个数。
"""
import os, sys
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
import sys, glob, random
import torch, torch.nn, torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
from scripts.lib.small_vqvae import Encoder, Decoder
from scripts.lib.ms_vq import MSVQVAE

B, SIZE = 4, 256
files = sorted(glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg', recursive=True))
random.seed(0)


def load_batch(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
        out.append(torch.from_numpy(np.asarray(im, dtype='uint8').copy()).float()
                   .permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(out)


torch.manual_seed(7)
m = MSVQVAE(embed_dim=64, n_codes=1024, ch=(64, 128, 256, 256),
            use_attn=True, two_3x3=True, size=256)
ck = torch.load(r'D:\work\cloud_results\P_16x16_attn_small_17820.pt',
                map_location='cpu', weights_only=False)
sd = ck['model']
m.load_state_dict({k: v for k, v in sd.items() if k.startswith(('enc.', 'dec.'))}, strict=False)
with torch.no_grad():
    m.vq_codebook.weight.data.copy_(
        F.normalize(sd['vq.codebook.weight'].float(), dim=1) * (64 ** 0.5))

opt = torch.optim.AdamW(list(m.enc.parameters()) + list(m.dec.parameters()), lr=1e-4)

print('=' * 92)
print('%-5s %-9s %-6s %-11s %-9s %-9s %-9s %-7s %s' %
      ('step', 'training', 'dead', 'ema_sum', 'ema_min', 'ema_med', 'ema_max', 'thr', '低于 thr 的码数'))
print('=' * 92)
for step in range(12):
    m.train()
    x = load_batch([random.choice(files) for _ in range(B)])
    f = m.enc(x)
    lt = m.training
    idxs, fh, vq = m.ms.f_to_idx(f)
    # 复制 _ema_step 的阈值逻辑，看真实的 ema_count
    ec = m.ms.ema_count
    K = m.vq_codebook.num_embeddings
    n_tok = sum(t.numel() for t in idxs)
    thr = m.ms.dead_thresh * (n_tok / K if m.ms.dead_thresh_rel else 1.0)
    below = int((ec < thr).sum().item())
    med = float(ec.median())
    print('%-5d %-9s %-6d %-11.2f %-9.5f %-9.5f %-9.4f %-7.3f %d' %
          (step, lt, m.ms.last_dead, float(ec.sum()), float(ec.min()), med,
           float(ec.max()), thr, below))
    rec = m.dec(fh)
    loss = F.mse_loss(rec, x) + 0.5 * F.l1_loss(rec, x) + vq
    opt.zero_grad(); loss.backward(); opt.step()

print()
print('判读：')
print('  若 ema_sum 恒为 0 且 ema_max 恒为 0  ⇒ _ema_step 从未执行（update_ema=False）')
print('  若 ema_sum ≈ %d 附近            ⇒ EMA 正常在跑' % (B * 341))
print('  "低于 thr 的码数" 就是 dead 应有的值')
print('DONE')
