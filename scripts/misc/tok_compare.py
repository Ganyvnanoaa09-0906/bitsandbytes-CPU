# -*- coding: utf-8 -*-
"""tok_compare.py — 三个分词器的"可学习性"对比（一个变量、一个可靠指标）

## 为什么是这个实验

`probe_ceiling.py` 的判决：AR（val 5.2114）比最强的 8-tap 局部探针（5.6671）还好
⇒ 局部上下文方向全部收摊，墙是 token 流自身的熵。
结合"两个表示都只榨出 ~520-560 bits/图"，杠杆只能在【分词器】。

## 指标只用一个：H(x | 左邻)

理由：这是本项目唯一被证明【可信】的条件熵 —— 1024 个格子、每格 ~7200 个样本，
稠密。H(x|左+上) 之类会因表稀疏而严重偏低（已在 §10.13⑦ 撤回）。

H(x|左邻) 越低 ⇒ 这个表示的 token 越好预测 ⇒ AR 越容易榨出结构。

## 公平性

三个分词器用【同一份 paths.json】（即同一批图、同一顺序），所以差异只来自分词器。

用法:
  python tok_compare.py --ckpt <分词器 ckpt> --out <tokens 输出> --name <名字>
"""
from __future__ import annotations
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
import argparse, os, sys, time, math, json
import numpy
import torch
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt

ROOT = r'D:\work\bitsandbytes-CPU'
REF_PATHS = os.path.join(ROOT, r'.autopilot\tokens_16x16_full.pt.paths.json')
SIZE = 256


class PathDS(Dataset):
    def __init__(self, paths, size=SIZE):
        self.paths = paths
        self.tf = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert('RGB'))


def entropy_bits(counts):
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0)


def cond_entropy_bits(pairs):
    g = defaultdict(Counter)
    for y, x in pairs:
        g[y][x] += 1
    num, den = 0.0, 0
    for _k, c in g.items():
        n = sum(c.values())
        num += n * entropy_bits(c)
        den += n
    return num / max(den, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--name', default='')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    name = a.name or os.path.basename(a.ckpt)

    paths = json.load(open(REF_PATHS, encoding='utf-8'))
    if a.limit and a.limit < len(paths):
        # ⚠️ 必须【均匀抽样】：paths 是按路径排序的，取前 N 张会拿到同一个目录的
        # 同质图片，条件熵会被系统性压低（本地踩过：max_items 64 让 rec 好一个量级）。
        step = len(paths) / a.limit
        paths = [paths[int(i * step)] for i in range(a.limit)]
    miss = sum(1 for p in paths[:500] if not os.path.exists(p))
    print('[tok] %s  图 %d 张（均匀抽样）  前500缺失 %d' % (name, len(paths), miss), flush=True)

    model, ck = vqvae_from_ckpt(a.ckpt)
    model.eval()
    print('[tok] ch=%s emb=%s attn=%s two=%s step=%s'
          % (ck.get('ch'), ck.get('embed_dim'), ck.get('use_attn'),
             ck.get('two_3x3'), ck.get('step')), flush=True)

    ds = PathDS(paths)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    toks = []
    t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(dl):
            _, idx, _ = model(x)
            toks.append(idx.reshape(idx.shape[0], -1).cpu())
            if (i + 1) % 200 == 0:
                print('[tok] %d/%d  %.0fs' % ((i + 1) * a.batch, len(ds), time.time() - t0),
                      flush=True)
    t = torch.cat(toks, 0).long()
    print('[tok] tokens %s  %.0fs' % (tuple(t.shape), time.time() - t0), flush=True)
    if a.out:
        torch.save(t, a.out)
        print('[tok] saved %s' % a.out, flush=True)

    # ---- 统计 ----
    g = int(round(t.shape[1] ** 0.5))
    t = t.view(t.shape[0], g, g)
    flat = t.reshape(-1).tolist()
    cnt = Counter(flat)
    V = 1024
    H1 = entropy_bits(cnt)
    same = float((t[:, :, :-1] == t[:, :, 1:]).float().mean())
    perm = torch.randperm(t.numel())
    rnd = float((t.reshape(-1) == t.reshape(-1)[perm]).float().mean())
    # 左邻 / 上邻（稠密表，可信）
    left = list(zip(t[:, :, :-1].reshape(-1).tolist(), t[:, :, 1:].reshape(-1).tolist()))
    up = list(zip(t[:, :-1, :].reshape(-1).tolist(), t[:, 1:, :].reshape(-1).tolist()))
    Hl = cond_entropy_bits(left)
    Hu = cond_entropy_bits(up)
    print()
    print('=' * 88)
    print('%-22s %8s %10s %12s %12s %10s %8s' %
          ('分词器', '唯一码', '一元熵', 'H(x|左邻)', 'H(x|上邻)', '相邻同率', '倍数'))
    print('-' * 88)
    print('%-22s %8d %10.3f %12.3f %12.3f %9.3f%% %7.1fx'
          % (name, len(cnt), H1, Hl, Hu, same * 100, same / max(rnd, 1e-9)))
    print('  ⇒ H(x|左邻) 单位 bits，越低表示这个表示的 token 越好预测')
    print('  ⇒ 参考：C_mse 分词器的 H(x|左邻) = 8.674 bits，其 AR val = 5.2114 nats')
    print('=' * 88)


if __name__ == '__main__':
    main()
