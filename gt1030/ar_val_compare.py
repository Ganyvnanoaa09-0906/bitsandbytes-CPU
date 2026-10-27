# -*- coding: utf-8 -*-
"""配对比较多个 ckpt 的 val：同一批 val 图上逐 batch 求差，报 Δ ± SEM(paired)。

    python gt1030/ar_val_compare.py --dumps a.npz b.npz c.npz [--base a.npz]

输出：
  · 每个模型的 mean ± SEM（非配对）
  · 每个模型 vs base 的 mean Δ ± SEM(paired) + t 值（|t|>2 才算有差别）
"""
from __future__ import annotations
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', required=True)
    ap.add_argument('--base', default='')
    args = ap.parse_args()
    data = {}
    for p in args.dumps:
        z = np.load(p, allow_pickle=True)
        name = str(z['ckpt']).split('\\')[-1].replace('.pt', '')
        d = str(z['ckpt'])
        # 加上目录名，避免同名 ckpt 混淆
        name = d.split('\\')[-2] + '/' + name
        data[name] = np.asarray(z['losses'], np.float64)
    n = min(len(v) for v in data.values())
    print('每个模型：mean ± SEM（非配对，n=%d 个 batch）' % n)
    for k, v in data.items():
        v = v[:n]
        print('  %-34s %.4f ± %.4f' % (k, v.mean(), v.std(ddof=1) / np.sqrt(n)))
    base_key = None
    if args.base:
        for k in data:
            if args.base in k:
                base_key = k
                break
    if base_key is None:
        base_key = list(data)[0]
    b = data[base_key][:n]
    print('\n配对比较（base = %s）' % base_key)
    print('  %-34s %10s %10s %8s %s' % ('model', 'Δmean', 'SEM(paired)', 't', '判读'))
    for k, v in data.items():
        if k == base_key:
            continue
        v = v[:n]
        diff = v - b
        sem = diff.std(ddof=1) / np.sqrt(n)
        t = diff.mean() / sem if sem > 1e-12 else 0.0
        verdict = '显著' if abs(t) > 2.0 else '在噪声内'
        print('  %-34s %+10.4f %10.4f %8.2f %s' % (k, diff.mean(), sem, t, verdict))


if __name__ == '__main__':
    main()
