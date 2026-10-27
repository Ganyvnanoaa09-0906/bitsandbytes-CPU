# -*- coding: utf-8 -*-
"""设备端算 val loss（尾部 2% 那 382 张），和 CPU trainer 的 val 口径一致。

    python gt1030/ar_v2_val.py --ckpt ... [--frac 0.02]

训练的时候别跑（会抢 GPU）。
"""
from __future__ import annotations
import argparse
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch
import gt1030.ar_v2_multi as MM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'F:\weibo_cos\ar_run_igpu\ckpt_last.pt')
    ap.add_argument('--tokens', default=r'F:\weibo_cos\tokens_32x32_weibo.pt')
    ap.add_argument('--frac', type=float, default=0.02)
    ap.add_argument('--batch', type=int, default=2)
    args = ap.parse_args()

    tok = torch.load(args.tokens, map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    n = tok.shape[0]
    n_train = int(round(n * (1.0 - args.frac)))
    va = np.ascontiguousarray(tok[n_train:, :MM.T].astype(np.int64))
    model = MM.Model(args.ckpt)
    model.forward(np.ascontiguousarray(va[:2]))          # 预热
    tot, cnt = 0.0, 0
    for i in range(0, len(va), args.batch):
        blk = np.ascontiguousarray(va[i:i + args.batch])
        if len(blk) < 1:
            break
        # 只有 batch 恰好等于 MM.B 时才是标准路径；不足则跳过
        if len(blk) != MM.B:
            continue
        loss = model.forward(blk)
        tot += float(loss) * len(blk)
        cnt += len(blk)
    print('ckpt=%s  val_loss=%.4f  (%d 张)'
          % (args.ckpt.split('\\')[-1], tot / max(cnt, 1), cnt))


if __name__ == '__main__':
    main()
