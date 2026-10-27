# -*- coding: utf-8 -*-
"""剖析 opt_step 内部：device AdamW / 小向量传输 / host AdamW 各占多少（独占条件）。"""
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
    model.forward(tokens); model.backward(tokens); model.opt_step(5e-6, 1)

    h, _ = suspend(args.pid)
    try:
        time.sleep(3.0)
        d = model.dev

        def t(fn, n=5):
            fn(); d.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            d.finish()
            return (time.time() - t0) / n * 1000

        # 1) 纯 device AdamW（40 个权重）
        def dev_adamw():
            for blk in model.blocks:
                for k in ('Wqkv', 'Wo', 'W12', 'W3'):
                    blk.W[k].step(5e-6, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001)
            model.head.W.step(5e-6, 0.9, 0.999, 1e-8, 0.01, 0.1, 0.001)

        print('device AdamW (40 个参数)          %7.1f ms' % t(dev_adamw))

        # 2) 小向量：下载 + 上传
        def small_xfer():
            for blk in model.blocks:
                gs = {k: d.down(blk.dw[k], (v.size,), np.float32, sync=False)
                      for k, v in blk.host.items()}
                d.finish()
                for k in blk.host:
                    blk._upload_small(k, sync=False)
                d.finish()

        print('小向量下载+上传 (10 层 x 6)        %7.1f ms' % t(small_xfer))

        # 3) host 侧 AdamW 计算
        for blk in model.blocks:
            for k, v in blk.host.items():
                g = np.random.randn(v.size).astype(np.float32) * 1e-6
                blk.m[k] = 0.9 * blk.m[k] + 0.1 * g
                blk.v[k] = 0.999 * blk.v[k] + 0.001 * g * g

        def host_adamw():
            for blk in model.blocks:
                for k, v in blk.host.items():
                    blk.m[k] = 0.9 * blk.m[k] + 0.1 * blk.m[k]
            return None

        print('host 侧小向量更新（估计）          %7.3f ms' % t(host_adamw, 20))

        # 4) 整个 opt_step
        print('整个 opt_step                      %7.1f ms'
              % t(lambda: model.opt_step(5e-6, 5), 3))

        # 5) grad_norm（含 40 次下载！）
        print('grad_norm                          %7.1f ms' % t(model.grad_norm, 3))
    finally:
        print('resume rc=%d' % resume(h))


if __name__ == '__main__':
    main()
