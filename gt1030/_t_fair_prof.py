# -*- coding: utf-8 -*-
"""独占条件下的逐算子剖面：CPU trainer 挂起后测 fwd/bwd 各阶段与单项算子。

    python gt1030/_t_fair_prof.py [--pid 16288]
"""
from __future__ import annotations
import argparse
import ctypes
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
    ap.add_argument('--pid', type=int, default=16288)
    args = ap.parse_args()
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A
    import gt1030.full_gpu_layer as L

    tok = torch.load(r'F:\weibo_cos\tokens_32x32_weibo.pt', map_location='cpu')
    tok = tok.numpy() if torch.is_tensor(tok) else np.asarray(tok)
    tokens = np.ascontiguousarray(tok[:MM.B, :MM.T].astype(np.int64))
    model = MM.Model(r'F:\weibo_cos\ar_run_full3\ckpt_16000.pt')
    model.forward(tokens); model.backward(tokens); model.opt_step(5e-6, 1)

    d = model.dev
    q, K = d.q, d.K
    M, MD, FF, T, HD, NH, NKV, BH, WROW = (MM.M, MM.MD, MM.FF, MM.T, MM.HD, MM.NH,
                                           MM.NKV, MM.BH, MM.WROW)
    blk = model.blocks[0]
    b = blk.b

    def tm(tag, fn, n=10):
        fn(); d.finish()
        t0 = time.time()
        for _ in range(n):
            fn()
        d.finish()
        return (time.time() - t0) / n * 1000

    h, st = suspend(args.pid)
    try:
        time.sleep(3.0)
        print('=== 独占条件（CPU trainer 挂起）===', flush=True)
        rows = []
        rows.append(('GEMM Wqkv 2048x1024x512', tm('a', lambda: L.hgemm_nn(
            b['h1'], blk.W['Wqkv'].w, b['qkv'], M, WROW, MD))))
        rows.append(('GEMM Wo   2048x512x512', tm('b', lambda: L.hgemm_nn(
            b['o_tok'], blk.W['Wo'].w, b['attn'], M, MD, MD))))
        rows.append(('GEMM W12  2048x2736x512', tm('c', lambda: L.hgemm_nn(
            b['h2'], blk.W['W12'].w, b['a2'], M, 2 * FF, MD))))
        rows.append(('GEMM W3   2048x512x1368', tm('d', lambda: L.hgemm_nn(
            b['g'], blk.W['W3'].w, b['y'], M, MD, FF))))
        rows.append(('bGEMM S=QnKn^T 16xTxTx64', tm('e', lambda: A.hgemm_batched(
            b['Qn'], b['Kn'], b['S'], T, T, HD, A.NO, A.YES, HD, HD, T,
            T * HD, T * HD, T * T))))
        rows.append(('softmax_causal', tm('f', lambda: K['softmax_causal'](
            q, (T * 256, BH), (256, 1), b['S'], b['P'], np.int32(T), np.float32(MM.scale)))))
        rows.append(('bGEMM O=PVb 16xTx64xT', tm('g', lambda: A.hgemm_batched(
            b['P'], b['Vb'], b['O'], T, HD, T, A.NO, A.NO, T, HD, HD,
            T * T, T * HD, T * HD))))
        rows.append(('bGEMM dP=dO·Vb^T', tm('h', lambda: A.hgemm_batched(
            b['dO'], b['Vb'], b['dP'], T, T, HD, A.NO, A.YES, HD, HD, T,
            T * HD, T * HD, T * T))))
        rows.append(('softmax_bwd_causal', tm('i', lambda: K['softmax_bwd_causal'](
            q, (T * 256, BH), (256, 1), b['P'], b['dP'], b['dS'],
            np.int32(T), np.float32(MM.scale)))))
        rows.append(('bGEMM dV=P^T dO', tm('j', lambda: A.hgemm_batched(
            b['P'], b['dO'], b['dV'], T, HD, T, A.YES, A.NO, T, HD, HD,
            T * T, T * HD, T * HD))))
        rows.append(('bGEMM dQ=dS Kn', tm('k', lambda: A.hgemm_batched(
            b['dS'], b['Kn'], b['dQ'], T, HD, T, A.NO, A.NO, T, HD, HD,
            T * T, T * HD, T * HD))))
        rows.append(('bGEMM dK=dS^T Qn', tm('l', lambda: A.hgemm_batched(
            b['dS'], b['Qn'], b['dK'], T, HD, T, A.YES, A.NO, T, HD, HD,
            T * T, T * HD, T * HD))))
        for nm, ms in rows:
            print('  %-30s %8.2f ms' % (nm, ms))
        ngeb = sum(ms for nm, ms in rows if 'GEMM' in nm and nm.startswith('GEMM'))
        natt = sum(ms for nm, ms in rows if nm.startswith('bGEMM') or 'softmax' in nm)
        print('  小计: 大 GEMM %.1f ms/层 (前向 4 个) / attention %.1f ms/层（fwd+bwd 全算）'
              % (ngeb, natt))
        fs, bs, os_ = [], [], []
        for i in range(5):
            t0 = time.time(); model.forward(tokens); fs.append(time.time() - t0)
            t0 = time.time(); model.backward(tokens); bs.append(time.time() - t0)
            t0 = time.time(); model.opt_step(5e-6, 100 + i); os_.append(time.time() - t0)
        print('  整步: fwd=%.3f bwd=%.3f opt=%.3f total=%.3f s'
              % (np.median(fs), np.median(bs), np.median(os_),
                 np.median(fs) + np.median(bs) + np.median(os_)), flush=True)
    finally:
        print('resume rc=%d' % resume(h), flush=True)


if __name__ == '__main__':
    main()
