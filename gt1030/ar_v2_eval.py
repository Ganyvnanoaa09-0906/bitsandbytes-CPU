# -*- coding: utf-8 -*-
"""单 ckpt 体检：权重里有没有 NaN/inf + 在 val 划分（尾部 2%）上的 loss。

    python gt1030/ar_v2_eval.py --ckpt <path> [--no-val]

注意：一个进程只建一个 Model（各自一个 context），要评多个 ckpt 请分进程调用。
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
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tokens', default=r'F:\weibo_cos\tokens_32x32_weibo.pt')
    ap.add_argument('--frac', type=float, default=0.02)
    ap.add_argument('--no-val', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck['model']
    bad = [k for k, v in sd.items() if not torch.isfinite(v).all()]
    absmax = max(float(v.abs().max()) for v in sd.values())
    print('%s: keys=%d step=%s absmax=%.3g  non-finite=%s'
          % (args.ckpt.split('\\')[-1], len(sd), ck.get('step'), absmax,
             (bad[:3] if bad else 'none')))
    if bad or args.no_val:
        return
    model = MM.Model(args.ckpt)
    tok = torch.load(args.tokens, map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    n = tok.shape[0]
    n_train = int(round(n * (1.0 - args.frac)))
    va = np.ascontiguousarray(tok[n_train:, :MM.T].astype(np.int64))
    model.forward(np.ascontiguousarray(va[:2]))          # 预热
    tot, cnt, per_batch = 0.0, 0, []
    for i in range(0, len(va) - MM.B + 1, MM.B):
        blk = np.ascontiguousarray(va[i:i + MM.B])
        loss = model.forward(blk)
        if not np.isfinite(loss):
            print('  val 出现非有限 loss，停止')
            break
        tot += float(loss) * len(blk)
        cnt += len(blk)
        per_batch.append(float(loss))
    mean = tot / max(cnt, 1)
    if len(per_batch) > 2:
        sd = float(np.std(per_batch, ddof=1))
        sem = sd / (len(per_batch) ** 0.5)
        print('  val_loss=%.4f ± %.4f (SEM, %d 张, %d 个 batch, batch 间 sd=%.3f)'
              % (mean, sem, cnt, len(per_batch), sd))
    else:
        print('  val_loss=%.4f (%d 张)' % (mean, cnt))


if __name__ == '__main__':
    main()
