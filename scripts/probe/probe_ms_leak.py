# -*- coding: utf-8 -*-
"""probe_ms_leak.py — 多尺度残差量化的【同尺度独立性】判决（多尺度路线成立与否的前提）

## 为什么这是最关键的一个实验

VAR 的 block-causal mask 规则是 `scale(i) >= scale(j)`，即【尺度内双向】。
我们直接抄这个 mask 到单尺度 token 上，得到了 teacher-forcing top-1 = 88.1%
的抄答案行为 —— 因为单尺度下同一块内的 token 是图像真实内容、强相关。

VAR 之所以敢用尺度内双向，是因为它的同一尺度 token 是【残差】
（把上一尺度重建减掉之后的余量），**近似互相独立**。

⇒ 所以整个多尺度路线成立的前提是：**我们的多尺度残差 token 也近似独立。**
   如果不独立，block mask 一样会泄漏，10 小时的分词器训练 + AR 训练全部白费。

## 判决指标（直接测，不猜）

用【条件熵】度量"知道同尺度邻居能不能预测当前 token"：

    H(X)            = 该尺度 token 的边缘熵
    H(X | Y)        = 已知同尺度邻居 Y 时 X 的条件熵（按 Y 的取值分组求经验熵再加权）
    泄漏比 leak     = 1 − H(X|Y) / H(X)

    leak ≈ 0    ⇒ 独立（block mask 安全）
    leak ≈ 1    ⇒ 完全可预测（block mask 会抄答案）

对照组：**单尺度 16×16 的真实 token**（用原 VQ 码本），我们已知它强相关
（相邻同码率 4.47%，随机基线 ~0.1%）。若多尺度的 leak 显著低于单尺度，
则结论成立。

同时打印【相邻同码率 / 随机基线】倍数作为第二个独立佐证 —— 两个指标互相印证。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, glob, random, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
from small_vqvae import Encoder, Decoder, VQ
from ms_vq import MSVQVAE

SIZE = 256
N_IMG = 384          # 要足够多才能估条件熵
BATCH = 8
PATCH = (1, 2, 4, 8, 16)
CKPT_MS = r'D:\work\bitsandbytes-CPU\.autopilot\_ms_b16\ckpt_last.pt'
CKPT_SS = r'D:\work\cloud_results\P_16x16_attn_small_17820.pt'


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


def entropy(counts):
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0)


def cond_entropy(pairs, min_group=8):
    """pairs: list of (y, x)。H(X|Y)，只对样本数 >= min_group 的 y 分组计入。"""
    groups = defaultdict(Counter)
    for y, x in pairs:
        groups[y][x] += 1
    num, den = 0.0, 0
    for y, c in groups.items():
        n = sum(c.values())
        if n < min_group:
            continue
        num += n * entropy(c)
        den += n
    return (num / den) if den > 0 else float('nan'), den


def leak_report(name, seqs, shape_fn, note=''):
    """seqs: LongTensor (N, total)。shape_fn(i) -> (h, w) for scale i"""
    print()
    print('  ── %s ──' % name)
    print('     尺度  token数  唯一码  边缘熵H(X)  H(X|Y)  泄漏比leak   相邻同码率  随机基线  倍数')
    for i, pn in enumerate(PATCH):
        g = shape_fn(i)                              # (N, pn, pn) 或 None
        if g is None:
            continue
        flat = g.reshape(g.shape[0], -1)
        # 水平邻居对
        pairs_h = [(int(flat[n, j]), int(flat[n, j + 1]))
                   for n in range(flat.shape[0]) for j in range(flat.shape[1] - 1)]
        pairs_v = [(int(flat[n, j]), int(flat[n + 0, j]))
                   for n in range(flat.shape[0]) for j in range(flat.shape[1])]
        # 垂直：跨图行，这里用同一图内上下行
        g2 = g
        pairs_v = [(int(g2[n, r, c]), int(g2[n, r + 1, c]))
                   for n in range(g2.shape[0]) for r in range(g2.shape[1] - 1)
                   for c in range(g2.shape[2])]
        mx = Counter(int(v) for v in flat.reshape(-1).tolist())
        Hx = entropy(mx)
        hxy_h, n_h = cond_entropy(pairs_h)
        hxy_v, n_v = cond_entropy(pairs_v)
        hxys = [v for v in (hxy_h, hxy_v) if not math.isnan(v)]
        Hxy = float(np.mean(hxys)) if hxys else float('nan')
        leak = 1 - Hxy / Hx if (Hx > 0 and not math.isnan(Hxy)) else float('nan')
        same_h = np.mean([a == b for a, b in pairs_h]) if pairs_h else float('nan')
        same_v = np.mean([a == b for a, b in pairs_v]) if pairs_v else float('nan')
        same = float(np.mean([same_h, same_v]))
        # 随机基线：把 flat 全局打乱后的同码率
        shuf = flat.reshape(-1).clone()
        perm = torch.randperm(shuf.numel())
        rnd = float((shuf == shuf[perm]).float().mean())
        print('     %-4d %5d %7d %10.3f %8.3f %10.3f %11.4f%% %8.4f%% %6.1fx'
              % (pn, flat.shape[1], len(mx), Hx, Hxy, leak, same * 100, rnd * 100,
                 same / max(rnd, 1e-9)))
    if note:
        print('     %s' % note)


# ---------------- 数据 ----------------
files = sorted(glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg', recursive=True))
random.seed(0)
random.shuffle(files)
files = files[:N_IMG]
print('=' * 104)
print('多尺度残差量化的同尺度独立性判决   图片 %d 张' % len(files))
print('=' * 104)

# ---------------- 多尺度 ----------------
ms = MSVQVAE(embed_dim=64, n_codes=1024, ch=(64, 128, 256, 256),
             use_attn=True, two_3x3=True, size=256)
ck = torch.load(CKPT_MS, map_location='cpu', weights_only=False)
try:
    ms.load_state_dict(ck['model'])
    print('  多尺度 ckpt 载入: %s (step %s)' % (CKPT_MS.split('\\')[-2], ck.get('step')))
except Exception as e:
    print('  ⚠️ 载入失败(%s)，改用迁移初始化' % str(e)[:50])
ms.eval()

ms_scales = [[] for _ in PATCH]
with torch.no_grad():
    for s in range(0, len(files), BATCH):
        x = load_batch(files[s:s + BATCH])
        seq = ms.encode_idx(x)                     # (B, 341)
        for i, (a, b) in enumerate(ms.ms.begin_ends):
            ms_scales[i].append(seq[:, a:b].clone())


def ms_shape(i):
    pn = PATCH[i]
    tot = sum(t.shape[0] for t in ms_scales[i])
    return torch.cat(ms_scales[i], 0).view(tot, pn, pn)


print()
print('[A] 多尺度残差 token（尺度内应该【独立】，否则 block mask 会泄漏）')
leak_report('多尺度残差 (ms_vq)', None, ms_shape)

# ---------------- 单尺度对照 ----------------
# 必须用真实单尺度 VQVAE（P_16x16_attn_small）的编码器 + 码本，
# 否则跟多尺度不是同一份图像、同一条管线，数字不可比。
from small_vqvae import vqvae_from_ckpt
print()
print('[B] 单尺度 16x16 真实 token（对照组：已知强相关）')
ss_model, ss_ck = vqvae_from_ckpt(CKPT_SS, map_location='cpu')
ss_model.eval()
print('    ckpt: %s  ch=%s  embed_dim=%s  use_attn=%s'
      % (CKPT_SS.split('\\')[-1], ss_ck.get('ch'), ss_ck.get('embed_dim'),
         ss_ck.get('use_attn')))
ss_all = []
with torch.no_grad():
    for s in range(0, len(files), BATCH):
        x = load_batch(files[s:s + BATCH])
        xr, idx, _ = ss_model(x)
        ss_all.append(idx.clone())
ss_g = torch.cat(ss_all, 0)


def ss_shape(i):
    return ss_g if i == 4 else None


leak_report('单尺度 16x16 (对照)', None, ss_shape)

print()
print('=' * 104)
print('判读：')
print('  ① 多尺度的 leak 应【显著低于】单尺度对照 ⇒ block-causal mask 安全')
print('  ② 多尺度的"倍数"应接近 1x，单尺度应远离 1x（已知单尺度 4.47% vs 基线 ~0.1%）')
print('  ③ 若多尺度的 leak 也接近 1，则【多尺度路线失败】，必须放弃 block mask')
print('DONE')
