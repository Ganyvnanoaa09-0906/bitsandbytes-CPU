# -*- coding: utf-8 -*-
"""F1-M6：在 iGPU 上续训 30M AR 模型（从 CPU trainer 的 ckpt 接着跑）。

    python gt1030/ar_v2_train.py --ckpt F:\\weibo_cos\\ar_run_full3\\ckpt_12000.pt \
        --steps 200 --lr 5e-5 --out-dir F:\\weibo_cos\\ar_run_igpu

每个 step：host embedding -> 10 层（arena）-> head+CE -> 反向 -> clip -> AdamW(device)。
loss/吞吐逐 step 打印，定期导出 CPU-format ckpt（可直接 sample_ar 采样）。
fwd/bwd 都在 device；只有 token 表和 2D pos 在 host（每步 gather 一次）。
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch

import gt1030.ar_v2_multi as MM
import gt1030.ar_v2_export as EX
import gt1030.ar_v2_optstate as OS

DEFAULT_TOKENS = r'F:\weibo_cos\tokens_32x32_weibo.pt'


def load_tokens(path, n_rows=0):
    t = torch.load(path, map_location='cpu')
    t = t.numpy() if torch.is_tensor(t) else np.asarray(t)
    if n_rows:
        t = t[:n_rows]
    # BOS 版 token 文件多一列，这里只用前 total_tokens 列（use_bos=False）
    return np.ascontiguousarray(t[:, :MM.T])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt')
    ap.add_argument('--tokens', default=DEFAULT_TOKENS)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--out-dir', default=r'F:\weibo_cos\ar_run_igpu')
    ap.add_argument('--log', default='')
    ap.add_argument('--shuffle', type=int, default=1)
    ap.add_argument('--opt-state', type=int, default=0, help='1=存取优化器状态')
    ap.add_argument('--spike', type=float, default=100.0, help='grad_norm 超过就跳过该步')
    ap.add_argument('--warmup', type=int, default=0, help='前多少步线性升温到 --lr')
    ap.add_argument('--lr-min', type=float, default=0.0, help='余弦衰减终点')
    ap.add_argument('--val-frac', type=float, default=0.02,
                    help='尾部这么多比例的图不参与训练（与 CPU trainer 的 val_frac 一致）')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    start_step = int(ck.get('step', 0))
    toks = load_tokens(args.tokens)
    n_all = toks.shape[0]
    n_train = int(round(n_all * (1.0 - args.val_frac)))
    n_img = n_train                     # 只用前 n_train 张；尾部就是 CPU 的验证集
    print('[igpu] ckpt=%s start_step=%d  训练用 %d 张（共 %d，尾部 %.1f%% 是 val）x %d token'
          % (os.path.basename(args.ckpt), start_step, n_img, n_all,
             100.0 * args.val_frac, MM.T), flush=True)

    model = MM.Model(args.ckpt)
    print('[igpu] arena 用了 %.0f MB;  batch=%d  T=%d  M=%d'
          % (model.dev._off / 1e6, MM.B, MM.T, MM.M), flush=True)

    if args.opt_state:
        try:
            OS.load_opt_state(model, args.ckpt)
        except Exception as e:
            print('[igpu] 优化器状态加载失败（继续用冷状态）: %r' % (e,), flush=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_img)
    pos = 0
    logf = open(args.log, 'a', encoding='utf-8') if args.log else None
    t_start = time.time()
    losses, step_times = [], []
    import math

    def lr_at(st):
        if args.warmup and st <= args.warmup:
            return args.lr * float(st) / float(args.warmup)
        prog = (st - args.warmup) / float(max(1, args.steps - args.warmup))
        prog = min(1.0, max(0.0, prog))
        return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))

    n_skip = 0
    n_spike = 0
    lr_used = args.lr
    for st in range(1, args.steps + 1):
        lr_used = lr_at(st)
        if pos + MM.B > len(order):
            order = rng.permutation(n_img)
            pos = 0
        idx = order[pos:pos + MM.B]
        pos += MM.B
        tokens = np.ascontiguousarray(toks[idx].astype(np.int64))

        t0 = time.time()
        loss = model.forward(tokens)
        if not np.isfinite(loss):
            n_skip += 1
            msg0 = '[igpu] step %d/%d SKIP (loss 非有限)' % (st, args.steps)
            print(msg0, flush=True)
            if logf:
                logf.write(msg0 + "\n"); logf.flush()
            continue
        model.backward(tokens)
        if args.spike:
            gpre = model.grad_norm()
            if not np.isfinite(gpre) or gpre > args.spike:
                n_spike += 1
                msg1 = '[igpu] step %d/%d SPIKE gnorm=%.3g → 跳过该步' % (st, args.steps, gpre)
                print(msg1, flush=True)
                if logf:
                    logf.write(msg1 + "\n"); logf.flush()
                continue
        gn = model.opt_step(lr_used, start_step + st)   # 真的用调度后的 lr
        dt = time.time() - t0
        losses.append(loss)
        step_times.append(dt)
        msg = ('[igpu] step %d/%d loss=%.4f gnorm=%.3f lr=%.2e %.3fs (%.0f tok/s)'
               % (st, args.steps, loss, gn, lr_used, dt, MM.M / dt))
        print(msg, flush=True)
        if logf:
            logf.write('%s\n' % msg)
            logf.flush()

        if args.save_every and st % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            p = os.path.join(args.out_dir, 'ckpt_%d.pt' % (start_step + st))
            EX.export_ckpt(model, p, step=start_step + st)
            print('[igpu] 导出 %s' % p, flush=True)
            if args.opt_state:
                try:
                    OS.save_opt_state(model, p)
                except Exception as e:
                    print('[igpu] opt state 保存失败: %r' % (e,), flush=True)

    tot = time.time() - t_start
    med = float(np.median(step_times))
    print('[igpu] 完成 %d 步 用时 %.1fs  median %.3fs/step  %.0f tok/s  loss %.4f -> %.4f'
          % (args.steps, tot, med, MM.M / med, losses[0], losses[-1]), flush=True)
    if n_skip:
        print('[igpu] 跳过了 %d 个非有限 loss 的 batch' % n_skip, flush=True)
    if logf:
        logf.write('[igpu] SUMMARY steps=%d median=%.3fs tok/s=%.0f loss %.4f -> %.4f\n'
                   % (args.steps, med, MM.M / med, losses[0], losses[-1]))
        logf.close()
    if not args.save_every:
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, 'ckpt_last.pt')
        EX.export_ckpt(model, p, step=start_step + args.steps)
        print('[igpu] 导出 %s' % p, flush=True)


if __name__ == '__main__':
    main()
