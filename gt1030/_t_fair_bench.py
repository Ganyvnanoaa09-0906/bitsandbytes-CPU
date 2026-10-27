# -*- coding: utf-8 -*-
"""公平测量：临时挂起 CPU trainer（NtSuspendProcess），在 CPU 空闲状态下测核显一步耗时，然后恢复。

    python gt1030/_t_fair_bench.py [--pid 16288]

安全性：用 ntdll 的 NtSuspendProcess/NtResumeProcess，finally 里一定会 Resume。
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
        raise OSError('OpenProcess failed %d' % ctypes.get_last_error())
    st = ntdll.NtSuspendProcess(ctypes.c_void_p(h))
    return h, st


def resume(h):
    st = ntdll.NtResumeProcess(ctypes.c_void_p(h))
    kernel32.CloseHandle(ctypes.c_void_p(h))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=16288)
    ap.add_argument('--hold', type=float, default=150.0, help='挂起多少秒（含测量时间）')
    args = ap.parse_args()

    import gt1030.ar_v2_multi as MM
    CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_16000.pt'
    tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))

    print('[fair] 建模型（此时 CPU 还在训练）...', flush=True)
    model = MM.Model(CKPT)
    model.forward(tokens); model.backward(tokens); model.opt_step(5e-6, 1)   # 预热 + CLBlast 调优
    print('[fair] 预热完成，挂起 CPU trainer pid=%d' % args.pid, flush=True)

    h, st = suspend(args.pid)
    try:
        if st != 0:
            print('[fair] NtSuspendProcess 返回 %d（继续测，但可能没停）' % st)
        time.sleep(3.0)          # 让核显把功耗/频率吃上去
        fs, bs, os_ = [], [], []
        for i in range(4):
            t0 = time.time(); l = model.forward(tokens); fs.append(time.time() - t0)
            t0 = time.time(); model.backward(tokens); bs.append(time.time() - t0)
            t0 = time.time(); model.opt_step(5e-6, 10 + i); os_.append(time.time() - t0)
        tot = [a + b + c for a, b, c in zip(fs, bs, os_)]
        print('[fair] CPU 暂停中： fwd=%.3f bwd=%.3f opt=%.3f total=%.3f s/step  (%.0f tok/s)'
              % (np.median(fs), np.median(bs), np.median(os_), np.median(tot),
                 MM.M / np.median(tot)), flush=True)
        print('[fair] loss=%.4f' % l, flush=True)
    finally:
        r = resume(h)
        print('[fair] 已恢复 CPU trainer (rc=%d)' % r, flush=True)

    # 恢复后再测一组作对比
    time.sleep(2.0)
    fs, bs = [], []
    for i in range(3):
        t0 = time.time(); model.forward(tokens); fs.append(time.time() - t0)
        t0 = time.time(); model.backward(tokens); bs.append(time.time() - t0)
    print('[fair] CPU 恢复后：  fwd=%.3f bwd=%.3f  (fwd+bwd=%.3f)'
          % (np.median(fs), np.median(bs), np.median(fs) + np.median(bs)), flush=True)


if __name__ == '__main__':
    main()
