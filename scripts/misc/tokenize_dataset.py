# -*- coding: utf-8 -*-
"""tokenize_dataset.py - 用 small_vqvae 的 checkpoint 把图片编码成 16x16 token。

支持多目录（--data 用 | 分隔）或清单文件（--manifest，jsonl 每行含 path 字段）。

**路径清单会一并写出**（<out>.paths.json），因为 mosaic mask 必须和 token 严格同序，
没有这份清单就无法验证对齐。

用法:
  python tokenize_dataset.py --data "dirA|dirB" --ckpt ckpt_last.pt --out tokens.pt
  python tokenize_dataset.py --manifest mix.jsonl --ckpt ckpt_last.pt --out tokens.pt
"""
from __future__ import annotations
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
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import VQVAE, vqvae_from_ckpt

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def collect(data_arg, manifest):
    if manifest:
        paths = []
        with open(manifest, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                p = r.get('path') or r.get('image') or r.get('src')
                if p and os.path.exists(p):
                    paths.append(p)
        return sorted(paths)
    roots = [x for x in data_arg.split('|') if x.strip()]
    paths = []
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(EXTS):
                    paths.append(os.path.join(dp, fn))
    return sorted(paths)


class PathDS(Dataset):
    def __init__(self, paths, size=256):
        self.paths = paths
        self.tf = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # 必须用 with：Image.open 是惰性的，不关会持续累积文件句柄与缓冲区
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='', help='图片目录，多个用 | 分隔')
    ap.add_argument('--manifest', default='', help='jsonl 清单（每行含 path 字段）')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--ch', default='',
                    help='编码器通道。留空则从 checkpoint 里读（新 checkpoint 会记）。'
                         '长度决定下采样倍率 2**len：4->16x->16x16网格, 5->32x->8x8网格')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0,
                    help='只处理前 N 张（0=全部）。用于快速冒烟验证管线')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = collect(a.data, a.manifest)
    if not paths:
        raise SystemExit('没有找到图片')
    if a.limit and a.limit < len(paths):
        # 均匀抽样而不是取前 N 张：取前 N 会因目录序拿到同质的一批，
        # 冒烟测出来的码本统计会失真（本地就踩过：max_items 64 让 rec 好一个量级）
        step = len(paths) / a.limit
        paths = [paths[int(i * step)] for i in range(a.limit)]
        print('[tokenize] --limit %d：均匀抽样' % a.limit, flush=True)
    print('[tokenize] 图片 %d 张' % len(paths), flush=True)

    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    # ch 决定下采样倍率（进而决定网格大小）。优先从 checkpoint 里读，
    # 保证「训练的什么架构，分词就用什么架构」，避免 state_dict 对不上。
    if a.ch:
        ch = tuple(int(x) for x in a.ch.split(','))
    else:
        ch = tuple(ckpt['ch']) if ckpt.get('ch') else (64, 128, 256, 256)
    n_codes = int(ckpt.get('n_codes', 1024))
    grid = a.size // (2 ** len(ch))
    print('[tokenize] ch=%s -> 下采样 %dx -> 网格 %dx%d = %d token/图'
          % (ch, 2 ** len(ch), grid, grid, grid * grid), flush=True)

    # ---- 多尺度残差量化（ms_vq）走另一条路 ----
    # ckpt 里有 ms_vq=True 就用 MSVQVAE.encode_idx，产出 (N, Σpn²=341) 而不是 (N,256)。
    # 判别依据必须来自 ckpt 而不是命令行，否则"训练的什么架构、分词就用什么架构"
    # 这条原则会再次被破坏（8x8 那次 ch 硬编码就踩过）。
    is_ms = bool(ckpt.get('ms_vq', False))
    if is_ms:
        from scripts.lib.ms_vq import msvqvae_from_ckpt
        model, _ = msvqvae_from_ckpt(ckpt)
        patch_nums = tuple(model.patch_nums)
        total = int(model.total_tokens)
        print('[tokenize] ★多尺度残差量化: patch_nums=%s 总 token=%d（单尺度是 %d）'
              % (patch_nums, total, grid * grid), flush=True)
    else:
        model, _ = vqvae_from_ckpt(ckpt)
        patch_nums, total = (), grid * grid
    m_ch = tuple(_['ch']) if _.get('ch') else ch
    if m_ch != ch:
        print('[tokenize] 注意: --ch %s 与 checkpoint 里的 %s 不一致，用 checkpoint 的'
              % (ch, m_ch), flush=True)

    ds = PathDS(paths, a.size)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    all_tokens = []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for x in dl:
            if is_ms:
                idx = model.encode_idx(x)          # (B, total)
            else:
                _, idx, _ = model(x)
                idx = idx.reshape(idx.shape[0], -1)
            all_tokens.append(idx.cpu())
            done += x.shape[0]
            if done % 512 < a.batch:
                print('[tokenize] %d/%d %.1fs' % (done, len(ds), time.time() - t0), flush=True)

    tokens = torch.cat(all_tokens, 0)
    torch.save(tokens, a.out)
    with open(a.out + '.paths.json', 'w', encoding='utf-8') as f:
        json.dump(paths, f, ensure_ascii=False)
    # 架构 sidecar：AR 训练必须知道 patch_nums 才能建对 mask 和位置编码，
    # 靠命令行手输迟早错（和 ch 硬编码是同一类事故）。
    with open(a.out + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'ms_vq': is_ms, 'patch_nums': list(patch_nums),
                   'total_tokens': total, 'grid': grid, 'ch': list(ch),
                   'n_codes': n_codes, 'size': a.size,
                   'src_ckpt': os.path.basename(a.ckpt),
                   'src_step': ckpt.get('step')}, f, ensure_ascii=False, indent=1)

    # 码本体检
    flat = tokens.reshape(-1)
    used = int(torch.unique(flat).numel())
    cnt = torch.bincount(flat.long(), minlength=1024).float()
    p = cnt / cnt.sum()
    ent = float(-(p[p > 0] * p[p > 0].log()).sum())
    import math
    print('[tokenize] saved %s -> %s in %.1fs' % (tuple(tokens.shape), a.out, time.time() - t0))
    print('[tokenize] 码本: 使用 %d/1024 (%.1f%%), 熵 %.3f/%.3f (%.1f%%)' % (
        used, used / 1024 * 100, ent, math.log(1024), ent / math.log(1024) * 100))
    print('[tokenize] 路径清单 -> %s' % (a.out + '.paths.json'))


if __name__ == '__main__':
    main()
