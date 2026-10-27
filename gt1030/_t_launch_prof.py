# -*- coding: utf-8 -*-
"""独占条件：测「小 kernel 的启动开销」与每层小 kernel 的总时间（判断 400 次 launch 是不是新瓶颈）。"""
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
    d = model.dev
    q, K = d.q, d.K
    b = model.blocks[0].b
    M, MD = MM.M, MM.MD

    h, _ = suspend(args.pid)
    try:
        time.sleep(3.0)
        # 1) 单次 launch 开销（用不存在的极小 grid 的 add_vec）
        small = d.buf(256 * 2)
        d.finish()

        def one():
            K['add_vec'](q, (128,), (128,), small, small, small, np.int32(128))

        one(); d.finish()
        t0 = time.time()
        for _ in range(500):
            one()
        d.finish()
        per = (time.time() - t0) / 500 * 1000
        print('单次 kernel launch + 执行（128 元素）: %.3f ms' % per)

        # 2) 每层前向的小 kernel 组合
        def fwd_small():
            K['rms_stats'](q, (M,), (64,), b['xin'], b['ss'], np.int32(M), np.int32(MD))
            K['rms_apply'](q, (M * MD,), (256,), b['xin'], model.blocks[0].ln1, b['ss'],
                           b['h1'], np.int32(M), np.int32(MD))
            K['split_q'](q, (MM.BH * MM.T * MM.HD,), (256,), b['qkv'], b['Qraw'],
                         np.int32(MM.T), np.int32(MM.HD), np.int32(MM.NH), np.int32(MM.NKV))
            K['merge_o'](q, (MM.BH * MM.T * MM.HD,), (256,), b['O'], b['o_tok'],
                         np.int32(MM.T), np.int32(MM.HD), np.int32(MM.NH), np.int32(MM.B))
            K['split_silu'](q, (M * MM.FF,), (256,), b['a2'], b['g'], np.int32(M), np.int32(MM.FF))

        def t(fn, n=10):
            fn(); d.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            d.finish()
            return (time.time() - t0) / n * 1000

        print('每层前向 5 个代表性小 kernel: %.2f ms（含 5 次 launch）' % t(fwd_small))
        print('  => 若 14 次 launch/层，前向小 kernel 约 %.0f ms/步'
              % (t(fwd_small) / 5 * 14 * 10))

        # 3) 整个 Block.forward / backward 里去掉 GEMM 的量级参考
        print()
        xh = np.zeros((MM.M, MM.MD), np.float16)
        print('参考：Block.forward 整层 %.2f ms' % t(lambda: model.blocks[0].forward(xh), 5))
    finally:
        print('resume rc=%d' % resume(h))


if __name__ == '__main__':
    main()
