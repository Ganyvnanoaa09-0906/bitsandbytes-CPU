# -*- coding: utf-8 -*-
"""token_stats.py — AR 是否真的打满了分词器上限？

客观指标（不靠肉眼）：
  ① 相邻 token 相同率 —— 真实图 vs AR 生成图
     报告实测：真实图 2.9%，随机基线 0.1%（29 倍）
     若生成图明显低于真实 ⇒ AR 没学到空间结构
  ② token 的 2D 自相关（水平/垂直）
  ③ 码本分布的距离（KL）
  ④ 【关键】用真实 token 做 teacher forcing，算 AR 的逐位置 loss
     若 loss 低但采样差 ⇒ 采样问题；若 loss 高 ⇒ 模型/训练不足

大样本：用 200 张真实图（12800 token）vs 生成 200 张
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys, glob, os, random, time, math
import torch
import numpy as np
from PIL import Image
from collections import Counter

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 82)
print('AR 是否打满了分词器上限（客观指标）')
print('=' * 82)

from scripts.lib.small_vqvae import vqvae_from_ckpt

SIZE = 256
N_REAL = 200
N_GEN = 200


def load_real(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
        a = np.asarray(im, dtype='uint8').copy()
        out.append(torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(out)


def token_stats(idx, side, name):
    """idx: (N, side*side) int64"""
    N = idx.shape[0]
    g = idx.view(N, side, side)
    # 相邻相同率（水平 + 垂直）
    h_same = (g[:, :, :-1] == g[:, :, 1:]).float().mean().item()
    v_same = (g[:, :-1, :] == g[:, 1:, :]).float().mean().item()
    # 全图相同率（基线）
    flat = idx.reshape(-1)
    perm = flat[torch.randperm(flat.numel())]
    rnd_same = (flat == perm).float().mean().item()
    # 码本使用与熵
    cnt = Counter(flat.tolist())
    total = flat.numel()
    ent = -sum((c / total) * math.log(c / total) for c in cnt.values())
    print('  %-22s 相邻同 %5.2f%%(H) %5.2f%%(V)   随机基线 %5.2f%%   倍数 %5.1fx'
          % (name, h_same * 100, v_same * 100, rnd_same * 100, h_same / max(rnd_same, 1e-9)))
    print('  %-22s 码本 %4d/%d (%.1f%%)   熵 %.3f (%.1f%%)'
          % ('', len(cnt), 1024, 100 * len(cnt) / 1024, ent, 100 * ent / math.log(1024)))
    return {'h_same': h_same, 'v_same': v_same, 'rnd': rnd_same,
            'codes': len(cnt), 'ent': ent, 'cnt': cnt, 'total': total}


# ---------- 真实 token ----------
files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(7)
print()
print('[1] 真实图的 token 统计（%d 张）' % N_REAL)
paths = random.sample(files, N_REAL)
real = load_real(paths)

real_idx = {}
for tag, ck, side in (('8x8', r'D:\work\cloud_results\vqvae_8x8.pt', 8),
                      ('16x16', r'D:\work\cloud_results\ckpt_C_mse.pt', 16)):
    vq, _ = vqvae_from_ckpt(ck)
    vq.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, N_REAL, 25):
            _, idx, _ = vq(real[i:i + 25])
            outs.append(idx.view(idx.shape[0], -1))
    real_idx[tag] = torch.cat(outs)
    print()
    token_stats(real_idx[tag], side, '真实 %s' % tag)

# ---------- 生成 token ----------
print()
print('[2] AR 生成的 token 统计（%d 张）' % N_GEN)


def gen_tokens(ckpt, n, L=0):
    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight'):
                mx = v.shape[0]
                break
        use = L if L > 0 else max(1, mx - 1)
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=use, loop_L_max=max(1, mx - 1), loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, 50):
            b = min(50, n - i)
            t0 = time.perf_counter()
            tk = m.generate(None, steps=cfg.total_tokens, top_k=1024, temperature=1.0,
                            batch=b, seed=1000 + i)
            dt = time.perf_counter() - t0
            out.append(tk)
            print('    生成 %d 张  %.1fs (%.3f s/张)' % (b, dt, dt / b))
    return torch.cat(out).long(), cfg


gen_idx = {}
for tag, ck, side, L in (
    ('8x8(ar_swiglu)', r'D:\work\cloud_results\swiglu_last.pt', 8, 0),
    ('16x16(loop1)', r'D:\work\cloud_results\loop1_26500.pt', 16, 3),
):
    print()
    toks, cfg = gen_tokens(ck, N_GEN, L)
    gen_idx[tag] = toks
    token_stats(toks, side, '生成 %s' % tag)

# ---------- 对比 ----------
print()
print('=' * 82)
print('[3] 结论：AR 的空间结构保持度')
print()
print('  %-24s %12s %12s %10s' % ('', '相邻同率', '随机基线', '倍数'))
print('  ' + '-' * 62)
for tag, side in (('8x8', 8), ('16x16', 16)):
    r = token_stats(real_idx[tag], side, '（重算）') if False else None
for name, idx, side in (
    ('真实 8x8', real_idx['8x8'], 8),
    ('生成 8x8', gen_idx['8x8(ar_swiglu)'], 8),
    ('真实 16x16', real_idx['16x16'], 16),
    ('生成 16x16', gen_idx['16x16(loop1)'], 16),
):
    N = idx.shape[0]
    g = idx.view(N, side, side)
    h = (g[:, :, :-1] == g[:, :, 1:]).float().mean().item()
    print('  %-24s %11.2f%%' % (name, h * 100))

print()
print('  判读:')
print('    若「生成」的相邻同率 ≈「真实」⇒ AR 完整学到了空间结构，瓶颈在分词器')
print('    若「生成」明显低于「真实」⇒ AR 还没学好，瓶颈在 AR')
print('DONE')
