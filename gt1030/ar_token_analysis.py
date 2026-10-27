# -*- coding: utf-8 -*-
"""把"色块"量化：对比真实数据与各个采样点的 token 级统计。

指标（都只看 token 序列，不需要 GPU）：
  · 边际熵（bits）：token 分布的熵，V=1024 ⇒ 上限 10 bit
  · 每图不同 token 数 / 熵：衡量"是不是就几种颜色"
  · KL(样本 ‖ 真实)：样本 token 分布离真实有多远
  · 邻接一致率 h/v：相邻 token 相同的比例 —— 色块会非常高，有结构会掉下来
  · 位置熵均值：每个空间位置上的 token 熵（模型有没学到"哪里有东西"）

    python gt1030/ar_token_analysis.py
"""
from __future__ import annotations
import io
import os
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch

V = 1024
G = 32              # 32x32
F = r'F:\weibo_cos'
REAL = os.path.join(F, 'tokens_32x32_weibo.pt')

SAMPLES = [
    ('6k (CPU)', os.path.join(F, 'ar_sample_6000.tokens.pt')),
    ('12k (CPU)', os.path.join(F, 'ar_sample_12000.tokens.pt')),
    ('20k (CPU)', os.path.join(F, 'ar_sample_20000.tokens.pt')),
    ('iGPU 20100', os.path.join(F, 'ar_sample_igpu_20100.tokens.pt')),
]


def load_tokens(path, cap=2000):
    o = torch.load(path, map_location='cpu', weights_only=False)
    a = o['tokens'] if isinstance(o, dict) and 'tokens' in o else o
    a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
    a = a.reshape(len(a), -1).astype(np.int64)
    return a[:cap]


def stats(tok, name, ref_hist=None):
    n, L = tok.shape
    hist = np.bincount(tok.reshape(-1), minlength=V).astype(np.float64)
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    per_img_distinct = np.mean([len(np.unique(t)) for t in tok])
    per_img_ent = np.mean([-(lambda q: (q[q > 0] * np.log2(q[q > 0])).sum())(
        np.bincount(t, minlength=V) / L) for t in tok])
    grid = tok.reshape(n, G, G)
    h_eq = float((grid[:, :, :-1] == grid[:, :, 1:]).mean())
    v_eq = float((grid[:, :-1, :] == grid[:, 1:, :]).mean())
    # 位置熵均值
    pos_ent = []
    for r in range(G):
        row = grid[:, r, :].reshape(-1)
        q = np.bincount(row, minlength=V) / max(1, len(row))
        pos_ent.append(float(-(q[q > 0] * np.log2(q[q > 0])).sum()))
    kl = float((p[p > 0] * np.log2(p[p > 0] / np.maximum(ref_hist[p > 0], 1e-12))).sum()) if ref_hist is not None else float('nan')
    print('%-12s 边际熵 %5.2f bit | 每图不同码 %6.1f | 每图熵 %5.2f bit | KL %5.2f | '
          '邻接一致 h/v %5.1f%%/%5.1f%% | 位置熵均值 %5.2f'
          % (name, ent, per_img_distinct, per_img_ent, kl, h_eq * 100, v_eq * 100,
             float(np.mean(pos_ent))))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', default='')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    if args.tokens:
        real = load_tokens(REAL, cap=4000)
        rh = np.bincount(real.reshape(-1), minlength=V).astype(np.float64)
        rh /= rh.sum()
        stats(load_tokens(args.tokens), args.label or os.path.basename(args.tokens), rh)
        return
    real = load_tokens(REAL, cap=4000)
    rh = np.bincount(real.reshape(-1), minlength=V).astype(np.float64)
    rh /= rh.sum()
    print('真实数据 %d 张（32×32，V=1024）' % len(real))
    stats(real, 'real', rh)
    print('-' * 110)
    for name, path in SAMPLES:
        if not os.path.exists(path):
            print('%-12s (缺文件)' % name); continue
        stats(load_tokens(path), name, rh)


if __name__ == '__main__':
    main()
