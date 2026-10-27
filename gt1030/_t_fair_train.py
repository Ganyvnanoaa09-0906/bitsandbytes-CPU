# -*- coding: utf-8 -*-
"""在 CPU trainer 挂起的状态下跑 N 步 iGPU 训练，拿到"核显独占"的持续吞吐与 loss。

    python gt1030/_t_fair_train.py [--steps 60] [--pid 16288] [--ckpt ...]

finally 里一定 NtResumeProcess。
"""
from __future__ import annotations
import argparse
import ctypes
import os
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch

PROCESS_SUSPEND_RESUME = 0x0800
ntdll = ctypes.WinDLL('ntdll')
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)


def suspend(pid):
    h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not h:
        raise OSError('OpenProcess failed')
    return h, ntdll.NtSuspendProcess(ctypes.c_void_p(h))


def resume(h):
    st = ntdll.NtResumeProcess(ctypes.c_void_p(h))
    kernel32.CloseHandle(ctypes.c_void_p(h))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--pid', type=int, default=16288)
    ap.add_argument('--ckpt', default=r'F:\weibo_cos\ar_run_full3\ckpt_16000.pt')
    ap.add_argument('--lr', type=float, default=5e-6)
    ap.add_argument('--out', default=r'F:\weibo_cos\ar_run_igpu\ckpt_excl.pt')
    args = ap.parse_args()

    import gt1030.ar_v2_multi as MM
    import gt1030.ar_v2_export as EX

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    start = int(ck.get('step', 0))
    tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    n_all = tok.shape[0]
    n_train = int(round(n_all * 0.98))
    toks = np.ascontiguousarray(tok[:n_train, :MM.T].astype(np.int64))

    print('[excl] 建模型（CPU 仍在训练，稍慢）...', flush=True)
    model = MM.Model(args.ckpt)
    rng = np.random.default_rng(0)
    order = rng.permutation(n_train)
    pos = 0

    h, st = suspend(args.pid)
    print('[excl] CPU trainer 已挂起(rc=%d)，开始 %d 步' % (st, args.steps), flush=True)
    losses, times = [], []
    try:
        time.sleep(2.0)
        for i in range(1, args.steps + 1):
            if pos + MM.B > len(order):
                order = rng.permutation(n_train); pos = 0
            idx = order[pos:pos + MM.B]; pos += MM.B
            tokens = np.ascontiguousarray(toks[idx])
            t0 = time.time()
            loss = model.forward(tokens)
            model.backward(tokens)
            gn = model.opt_step(args.lr, start + i)
            dt = time.time() - t0
            losses.append(loss); times.append(dt)
            if i % 10 == 0 or i == 1:
                print('[excl] %3d/%d loss=%.4f gnorm=%.3f %.3fs (%.0f tok/s)'
                      % (i, args.steps, loss, gn, dt, MM.M / dt), flush=True)
    finally:
        r = resume(h)
        print('[excl] CPU trainer 已恢复(rc=%d)' % r, flush=True)

    med = float(np.median(times[2:])) if len(times) > 3 else float(np.median(times))
    print('[excl] 完成 %d 步  median %.3f s/step = %.0f tok/s  (min %.3f / max %.3f)'
          % (len(times), med, MM.M / med, min(times), max(times)), flush=True)
    print('[excl] loss %.4f -> %.4f  mean %.4f' % (losses[0], losses[-1],
                                                   float(np.mean(losses))), flush=True)
    try:
        EX.export_ckpt(model, args.out, step=start + args.steps)
        print('[excl] 导出 %s' % args.out, flush=True)
    except Exception as e:
        print('[excl] 导出失败: %r' % (e,), flush=True)


if __name__ == '__main__':
    main()
