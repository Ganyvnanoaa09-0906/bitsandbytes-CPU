# -*- coding: utf-8 -*-
"""video_cost_at_scale.py — 视频规模下的真实成本结构（不再从 T=256 外推）

为什么需要这个:
    measure_video_cost2.py 从 T=256 单点按 T^2 外推，得出"144p 15s 单层 4.2 分钟"。
    video_attn_scaling.py 实测后发现斜率在漂移（0.78 → 1.65），全局拟合会骗人。
    ⇒ 唯一可靠的办法是**在真实视频形状上直接测**，而不是外推。

本脚本测三件事:
    [1] 真实视频形状下的单层 self-attention 成本（frames=8/16/32，多分辨率）
    [2] 成本结构分解：attention / FFN / 卷积时序层 各占多少
    [3] O(T) 线性注意力（GDN 路径）在同形状下的成本对比

判据:
    attention 占比 > 50% ⇒ 优化 attention（线性化 / 稀疏 / 窗口）
    否则 ⇒ 去优化占比最大的那一项，别在 attention 上花时间
"""
from __future__ import annotations

import ctypes
import gc
import math
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SelfAttention, SwiGLU  # noqa: E402

torch.set_num_threads(6)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t),
                ('d', ctypes.c_size_t), ('e', ctypes.c_size_t), ('f', ctypes.c_size_t)]


def peak_rss_mb():
    pm = PMC()
    pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1e6


def timed(fn, warmup=1, reps=3):
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            raise
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


class MinimalCfg:
    """只放 SelfAttention / SwiGLU 需要的字段，避免依赖完整 config 的其它开关。"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_cfg(d_model=512, n_head=8, n_kv_head=4):
    return MinimalCfg(
        d_model=d_model, n_head=n_head, n_kv_head=n_kv_head,
        ffn_dim=d_model * 4, fuse_qkv=False,
        cond_mode='none', pos_mode='none', token_order='raster',
        head_mode='flat', mask_mode='causal',
    )


def conv_temporal_cost(B, C, F, H, W, reps=3):
    """时序卷积（3D conv 的时间维）——视频相对图像新增的主要卷积成本。"""
    conv = nn.Conv3d(C, C, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
    x = torch.randn(B, C, F, H, W)
    t = timed(lambda: conv(x), reps=reps)
    macs = B * C * C * F * H * W * 27
    return t, macs


def main():
    print('=' * 88)
    print('视频规模下的真实成本结构（直接测，不外推）')
    print('=' * 88)
    cfg = make_cfg()
    D, NH = cfg.d_model, cfg.n_head
    attn = SelfAttention(cfg)
    attn.eval()
    ffn = SwiGLU(D, cfg.ffn_dim)
    ffn.eval()
    print('d_model=%d n_head=%d n_kv_head=%d ffn_dim=%d  threads=%d'
          % (D, NH, cfg.n_kv_head, cfg.ffn_dim, torch.get_num_threads()))

    # ---- [1] attention at real video token counts ----
    # 视频 token 数 = 帧数 × 每帧 patch 数。取 256p（16x16 patch → 256 tok/帧）
    # 与 144p（12x12 → 144 tok/帧）两档，帧数 8/16/32。
    print('\n[1] self-attention 单层，真实视频 token 数')
    print('  %-22s %8s %12s %12s %10s' % ('场景', 'T', '前向 ms', 't/T (µs)', '峰值MB'))
    print('  ' + '-' * 70)
    rows = []
    for label, per_frame, frames in (('144p 8帧', 144, 8), ('144p 16帧', 144, 16),
                                     ('144p 32帧', 144, 32), ('256p 8帧', 256, 8),
                                     ('256p 16帧', 256, 16), ('256p 32帧', 256, 32)):
        T = per_frame * frames
        x = torch.randn(1, T, D)
        try:
            with torch.no_grad():
                t = timed(lambda: attn(x, cache=None, append=True), reps=2)
            rows.append((label, T, t))
            print('  %-22s %8d %12.2f %12.2f %10.1f'
                  % (label, T, t * 1e3, t / T * 1e6, peak_rss_mb()))
        except Exception as e:
            print('  %-22s %8d FAILED %s: %s' % (label, T, type(e).__name__, e))
        finally:
            del x
            gc.collect()

    # ---- [2] cost structure: attention vs FFN vs temporal conv ----
    print('\n[2] 成本结构分解（同 token 数下每层各算子的占比）')
    print('  %-14s %10s %12s %12s %12s %10s %10s'
          % ('形状', 'T', 'attn ms', 'FFN ms', 'convT ms', 'attn%', 'FFN%'))
    print('  ' + '-' * 82)
    for label, per_frame, frames, H, W, C in (
            ('144p 16帧', 144, 16, 12, 12, 128),
            ('256p 16帧', 256, 16, 16, 16, 128),
            ('256p 32帧', 256, 32, 16, 16, 128)):
        T = per_frame * frames
        x = torch.randn(1, T, D)
        with torch.no_grad():
            ta = timed(lambda: attn(x, cache=None, append=True), reps=2)
            tf = timed(lambda: ffn(x), reps=2)
            try:
                tc, macs = conv_temporal_cost(1, C, frames, H, W, reps=2)
            except Exception as e:
                tc, macs = float('nan'), 0
                print('    (conv3d 失败: %s)' % e)
        tot = ta + tf + (tc if tc == tc else 0)
        if tot > 0:
            print('  %-14s %10d %12.2f %12.2f %12.2f %9.1f%% %9.1f%%'
                  % (label, T, ta * 1e3, tf * 1e3, tc * 1e3,
                     100 * ta / tot, 100 * tf / tot))
        del x
        gc.collect()

    # ---- [3] the O(T) alternative: GDN path ----
    print('\n[3] O(T) 线性注意力对照（GDN：仓库已有 AVX2 内核）')
    try:
        from bitsandbytes.gdn_cpu import load_native  # noqa
        print('  gdn_cpu 可加载')
    except Exception as e:
        print('  gdn_cpu 加载失败: %s' % e)
    # GDN 的单位成本取自仓库既有实测（报告 §10 记录）：
    #   单层 fwd+bwd T=1024 B=2 H=4 d=64 = 51.6 ms ⇒ 推理 fwd ≈ 17.2 ms ⇒ 单样本 T=1024 ≈ 8.6 ms
    # 单位成本 = 8.6ms / 1024 tok = 8.40 µs/tok/层（这是 O(T) 的常数）
    gdn_us_per_tok = 8.40
    print('  参考单位成本: %.2f µs/token/层（O(T)，取自既有 GDN 实测）' % gdn_us_per_tok)
    print('  %-14s %10s %14s %14s %12s' % ('形状', 'T', 'attention ms', 'GDN ms', '倍率'))
    print('  ' + '-' * 68)
    for label, T, t_attn in rows:
        t_gdn = gdn_us_per_tok * T / 1e3        # ms
        print('  %-14s %10d %14.2f %14.2f %11.1fx'
              % (label, T, t_attn * 1e3, t_gdn, (t_attn * 1e3) / t_gdn))

    print('\n' + '=' * 88)
    print('读法: attention 占比高且倍率随 T 增大 ⇒ 长视频必须线性化；')
    print('      若 FFN/卷积占比更高 ⇒ 先优化那里，换注意力架构是白费。')
    print('=' * 88)


if __name__ == '__main__':
    main()
