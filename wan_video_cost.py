# -*- coding: utf-8 -*-
"""wan_video_cost.py — Wan2.1 出 720p/1080p、15 秒到底要什么（算 + 实测）

先算，再实测，避免"感觉不行"或"感觉能行"。

Wan2.1 的压缩（来自其模型卡与 config）:
    空间 8×、时间 4×、latent 通道 16
    ⇒ latent 形状 (16, F/4+1, H/8, W/8)，token 数 T = (F/4+1) × (H/8) × (W/8)

config（已从 hf-mirror 拉取并确认）:
    dim=1536  ffn_dim=8960  num_heads=12  num_layers=30  in/out_dim=16

本脚本:
    [1] 算 token 数与各部分内存（权重 / 激活 / attention 矩阵）
    [2] **实测** 单层 attention 在真实 T 下的内存与时间（这是决定性问题）
    [3] 给出结论：哪一档在本机可行，哪一档根本不可行

⚠️ 判据是**内存**不是时间：时间只是慢，内存不够是"做不了"。
"""
from __future__ import annotations

import math
import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
TOTAL_RAM_GB = 15.4
DIM = 1536
FFN = 8960
NHEAD = 12
NLAYER = 30
LATENT_CH = 16
SPATIAL = 8
TEMPORAL = 4


def latent_shape(frames, h, w):
    f = frames // TEMPORAL + 1
    return (LATENT_CH, f, h // SPATIAL, w // SPATIAL)


def tokens(ls):
    return ls[1] * ls[2] * ls[3]


def main():
    print("=" * 92)
    print("Wan2.1-T2V-1.3B 出 720p / 1080p / 15 秒：成本账")
    print("=" * 92)
    print("模型: dim=%d ffn=%d heads=%d layers=%d latent_ch=%d 压缩 空间%d×/时间%d×"
          % (DIM, FFN, NHEAD, NLAYER, LATENT_CH, SPATIAL, TEMPORAL))
    print("本机内存: %.1f GB" % TOTAL_RAM_GB)

    # Wan2.1-T2V 默认 16 fps
    FPS = 16
    secs = 15
    frames = secs * FPS
    print("\n15 秒 @ %d fps = %d 帧" % (FPS, frames))

    cases = [
        ("480p (832x480)", frames, 480, 832),
        ("720p (1280x720)", frames, 720, 1280),
        ("1080p (1920x1080)", frames, 1080, 1920),
    ]

    print("\n[1] latent 与 token 数")
    print("  %-20s %-26s %10s %14s" % ("分辨率", "latent 形状(16,F',h,w)", "T", "T² (attention)"))
    print("  " + "-" * 78)
    info = []
    for name, F, H, W in cases:
        ls = latent_shape(F, H, W)
        T = tokens(ls)
        info.append((name, F, H, W, ls, T))
        print("  %-20s %-26s %10d %14.3e"
              % (name, "(%d,%d,%d,%d)" % ls, T, float(T) * T))

    print("\n[2] 各部分内存（fp32）")
    print("  %-20s %12s %14s %16s %14s"
          % ("分辨率", "权重(4bit)", "注意力矩阵", "激活(粗估)", "合计(4bit主干)"))
    print("  " + "-" * 84)
    w_fp32 = (1419e6 * 4) / 1e9
    w_4bit = (1419e6 * 0.5) / 1e9
    te = 11.36
    vae = 0.51
    for name, F, H, W, ls, T in info:
        # 注意力矩阵：为了让"得分矩阵"可物化所需的字节（单头 fp32）
        attn_mat = T * T * 4 / 1e9
        # 激活粗估：主干中间张量。每层至少持有 h (T×dim) 与其若干派生
        # 用 T*dim*4 作为"一层一个张量"的量纲
        one_tensor = T * DIM * 4 / 1e9
        act = one_tensor * 8          # 保守：同时活着 8 个同量级张量
        tot = w_4bit + vae + act
        print("  %-20s %10.2f GB %12.3f GB %13.2f GB %12.2f GB"
              % (name, w_4bit, attn_mat, act, tot))
    print("  （另有文本编码器 %.2f GB，可离线预计算后卸载，不计入）" % te)
    print("  权重 fp32 = %.2f GB；4bit = %.2f GB" % (w_fp32, w_4bit))

    # ---- [3] 实测：真实 T 下的单层 attention ----
    print("\n[3] 实测：单层 self-attention（真实 T，fp32，本机 6 线程）")
    print("  %-20s %10s %14s %16s %14s"
          % ("分辨率", "T", "峰值RSS增量", "单层时间", "×30 层(仅attn)"))
    print("  " + "-" * 84)
    import ctypes
    import time

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                    ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]

    def rss():
        pm = PMC(); pm.cb = ctypes.sizeof(pm)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
        return pm.WorkingSetSize / 1e9

    hd = DIM // NHEAD
    for name, F, H, W, ls, T in info:
        r0 = rss()
        try:
            q = torch.randn(1, NHEAD, T, hd)
            k = torch.randn(1, NHEAD, T, hd)
            v = torch.randn(1, NHEAD, T, hd)
            t0 = time.perf_counter()
            with torch.no_grad():
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            dt = time.perf_counter() - t0
            r1 = rss()
            print("  %-20s %10d %12.2f GB %13.3f s %13.1f s"
                  % (name, T, r1 - r0, dt, dt * NLAYER))
            del q, k, v, y
            import gc; gc.collect()
        except Exception as e:
            print("  %-20s %10d  FAILED: %s: %s" % (name, T, type(e).__name__, str(e)[:50]))

    print("\n" + "=" * 92)
    print("结论读法:")
    print("  · 注意力矩阵那列若远超本机内存 ⇒ 该分辨率**不可行**，不是慢的问题")
    print("  · 实测单层时间 × 30 层只在 T 小的时候可外推；T 大时 O(T²) 会主导")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
