# -*- coding: utf-8 -*-
"""独占条件下测反向那 8 个大 GEMM（tn/nt 形状）各自的耗时，判断还有没有肉。"""
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
    import gt1030.full_gpu_layer as L

    tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
    model = MM.Model(r'F:\weibo_cos\ar_run_full3\ckpt_16000.pt')
    model.forward(tokens); model.backward(tokens)
    blk = model.blocks[0]
    d, b = blk.dev, blk.b
    M, MD, FF, WROW = MM.M, MM.MD, MM.FF, MM.WROW

    h, _ = suspend(args.pid)
    try:
        time.sleep(3.0)
        rows = []

        def t(tag, fn, n=6):
            fn(); d.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            d.finish()
            rows.append((tag, (time.time() - t0) / n * 1000))

        t('fwd Wqkv nn', lambda: L.hgemm_nn(b['h1'], blk.W['Wqkv'].w, b['qkv'], M, WROW, MD))
        t('fwd W12  nn', lambda: L.hgemm_nn(b['h2'], blk.W['W12'].w, b['a2'], M, 2 * FF, MD))
        t('bwd dW3  tn', lambda: L.hgemm_tn(b['g'], b['dy'], blk.W['W3'].grad, FF, MD, M))
        t('bwd dg   nt', lambda: L.hgemm_nt(b['dy'], blk.W['W3'].w, b['dg'], M, FF, MD))
        t('bwd dW12 tn', lambda: L.hgemm_tn(b['h2'], b['da2'], blk.W['W12'].grad, MD, 2 * FF, M))
        t('bwd dh2  nt', lambda: L.hgemm_nt(b['da2'], blk.W['W12'].w, b['dh2'], M, MD, 2 * FF))
        t('bwd dWo  tn', lambda: L.hgemm_tn(b['o_tok'], b['dattn'], blk.W['Wo'].grad, MD, MD, M))
        t('bwd dO   nt', lambda: L.hgemm_nt(b['dattn'], blk.W['Wo'].w, b['dO_tok'], M, MD, MD))
        t('bwd dWqkv tn', lambda: L.hgemm_tn(b['h1'], b['dqkv'], blk.W['Wqkv'].grad, MD, WROW, M))
        t('bwd dh1  nt', lambda: L.hgemm_nt(b['dqkv'], blk.W['Wqkv'].w, b['dh1'], M, MD, WROW))
        tot = 0.0
        for tag, ms in rows:
            print('  %-14s %7.2f ms' % (tag, ms))
            if tag.startswith('bwd'):
                tot += ms
        print('  反向大 GEMM 合计 %.1f ms/层 => %.0f ms/步' % (tot, tot * 10))
    finally:
        print('resume rc=%d' % resume(h))


if __name__ == '__main__':
    main()
