# -*- coding: utf-8 -*-
"""独占条件：把 Block.backward 拆开（大 GEMM / attention / 小 kernel / 小向量梯度）看谁占得多。"""
from __future__ import annotations
import argparse
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import torch


def suspend(pid):
    ntdll = ctypes.WinDLL('ntdll'); k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    h = k32.OpenProcess(0x0800, False, pid)
    return h, ntdll.NtSuspendProcess(ctypes.c_void_p(h))


def resume(h):
    ntdll = ctypes.WinDLL('ntdll'); k32 = ctypes.WinDLL('kernel32')
    st = ntdll.NtResumeProcess(ctypes.c_void_p(h)); k32.CloseHandle(ctypes.c_void_p(h)); return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=16288)
    args = ap.parse_args()
    import gt1030.ar_v2_multi as MM

    tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
    model = MM.Model(r'F:\weibo_cos\ar_run_full3\ckpt_16000.pt')
    model.forward(tokens)
    blk = model.blocks[0]
    d, b = blk.dev, blk.b
    q, K = d.q, d.K
    M, MD, FF, T, HD, NH, NKV, BH, WROW = (MM.M, MM.MD, MM.FF, MM.T, MM.HD, MM.NH,
                                           MM.NKV, MM.BH, MM.WROW)

    h, _ = suspend(args.pid)
    try:
        time.sleep(3.0)

        def t(tag, fn, n=6):
            fn(); d.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            d.finish()
            print('  %-24s %7.2f ms' % (tag, (time.time() - t0) / n * 1000))

        # 小向量梯度那几个 kernel 单独测
        t('rms_dw_from_ss (ln1, 512 列)', lambda: K['rms_dw_from_ss'](
            q, (MD,), None, b['xin'], b['dh1'], b['ss'], blk.dw['ln1'],
            np.int32(M), np.int32(MD)))
        t('rms_dw_from_ss (qk, 64 列/16384 行)', lambda: K['rms_dw_from_ss'](
            q, (HD,), None, b['Qraw'], b['dQ'], b['ss_q'], blk.dw['qw'],
            np.int32(BH * T), np.int32(HD)))
        t('col_sum_prod (512 列)', lambda: K['col_sum_prod'](
            q, (MD,), None, b['attn'], b['din_a'], blk.dw['ls1'],
            np.int32(M), np.int32(MD)))
        t('rms_bwd_stats (qk, 16384 行)', lambda: K['rms_bwd_stats'](
            q, (BH * T,), (64,), b['Qraw'], b['dQ'], blk.qw, b['ss_q'], b['sc_q'],
            np.int32(BH * T), np.int32(HD)))
        # Block.backward 整体
        xh = np.zeros((M, MD), np.float16)
        t('Block.forward (整层)', lambda: blk.forward(xh), 5)
        t('Block.backward (整层)', lambda: blk.backward(xh), 5)
    finally:
        print('resume rc=%d' % resume(h))


if __name__ == '__main__':
    main()
