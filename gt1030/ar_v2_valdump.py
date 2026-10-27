# -*- coding: utf-8 -*-
# 把某个 ckpt 在固定 val 划分上的逐 batch loss 存下来（供配对比较）。
#   python gt1030/ar_v2_valdump.py --ckpt <path> --out <dir>/xxx.npz
# 为什么：单模型 val 的 SEM 有 ±0.05（batch 间 sd 0.72 / sqrt(191)），
# 两个模型用非配对误差比差 0.01~0.05 完全分不开；同一批 val 图上的配对差值 SEM 通常小一个量级。
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', default=r'F:\weibo_cos\tokens_32x32_weibo.pt')
    ap.add_argument('--frac', type=float, default=0.02)
    args = ap.parse_args()

    tok = torch.load(args.tokens, map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    n = tok.shape[0]
    n_train = int(round(n * (1.0 - args.frac)))
    va = np.ascontiguousarray(tok[n_train:, :MM.T].astype(np.int64))
    model = MM.Model(args.ckpt)
    model.forward(np.ascontiguousarray(va[:2]))          # 预热
    losses = []
    for i in range(0, len(va) - MM.B + 1, MM.B):
        blk = np.ascontiguousarray(va[i:i + MM.B])
        losses.append(float(model.forward(blk)))
    losses = np.asarray(losses, np.float64)
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(args.out, losses=losses, ckpt=os.path.abspath(args.ckpt))
    print('%s: n=%d mean=%.4f' % (os.path.basename(args.out), len(losses), losses.mean()))


if __name__ == '__main__':
    main()
