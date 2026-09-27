# -*- coding: utf-8 -*-
"""wan_feasible_envelope.py — 反解：这台机器上 Wan 到底能出多大的视频

上一轮已判定 720p/1080p 15 秒不可行（720p 需 37 TB 注意力矩阵、5.1 天计算）。
本脚本反过来解**可行包络**：在给定内存预算下，分辨率 × 帧数 的哪些组合能跑。

判据:
    attention 的 T×T 矩阵内存 ≤ 预算（默认 6 GB，留出权重与激活）
    且单次前向的注意力时间在可接受范围（默认 ≤ 300 s）

数据来源:
    斜率实测（wan_4way_stress.py）: dense p=2.16, SDPA p=1.95；
    绝对量在 T=8192 处标定（dense 4.279 s, SDPA 1.598 s，单层）。

输出:
    · 一张 分辨率 × 帧数 的可行性表（OK / 内存不足 / 太慢）
    · 明确给出"最大可行分辨率 @ 给定时长"和"最大可行时长 @ 给定分辨率"
"""
from __future__ import annotations

import math
import os

LAT_CH, SP, TP = 16, 8, 4
DIM, NHEAD, NLAYER = 1536, 12, 30
FPS = 16

# 实测标定点（wan_4way_stress.py，单层 attention，6 线程）
T_REF = 8192
T_DENSE_REF = 4.279
T_SDPA_REF = 1.598
P_DENSE = 2.16
P_SDPA = 1.95

BUDGET_GB = float(os.environ.get("BUDGET_GB", "6"))
TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", "300"))


def tokens(frames, h, w):
    f = frames // TP + 1
    return f * (h // SP) * (w // SP)


def attn_matrix_gb(T):
    return NHEAD * T * T * 4 / 1e9


def fwd_seconds(T, per_layer_ref, p):
    """30 层单次前向的注意力时间（用最大标定点外推，最保守）。"""
    if T <= T_REF:
        # 小 T 按实测斜率内插
        return per_layer_ref * (T / T_REF) ** p * NLAYER
    return per_layer_ref * (T / T_REF) ** p * NLAYER


def verdict(T):
    mem = attn_matrix_gb(T)
    t = fwd_seconds(T, T_SDPA_REF, P_SDPA)
    if mem > BUDGET_GB:
        return "内存不足", mem, t
    if t > TIME_LIMIT_S:
        return "太慢", mem, t
    return "OK", mem, t


def main():
    print("=" * 96)
    print("Wan2.1 在本机的**可行包络**（预算 %.1f GB，单次前向注意力 ≤ %.0f s）"
          % (BUDGET_GB, TIME_LIMIT_S))
    print("=" * 96)
    print("标定: T=%d 时单层 dense %.3f s / SDPA %.3f s；斜率 dense p=%.2f, SDPA p=%.2f"
          % (T_REF, T_DENSE_REF, T_SDPA_REF, P_DENSE, P_SDPA))

    resolutions = [(128, 128), (176, 176), (256, 256), (320, 320), (384, 384),
                   (480, 480), (512, 512), (640, 640), (720, 720), (832, 832)]
    frame_list = [16, 32, 48, 64, 81, 121, 161, 240]

    print("\n[1] 可行性表（行=分辨率，列=帧数；数值=T，标记=判定）")
    hdr = "  %-12s" % "分辨率\\帧数"
    for F in frame_list:
        hdr += " %9s" % ("%d(%.1fs)" % (F, F / FPS))
    print(hdr)
    print("  " + "-" * (12 + 10 * len(frame_list)))
    for (H, W) in resolutions:
        line = "  %-12s" % ("%dx%d" % (H, W))
        for F in frame_list:
            T = tokens(F, H, W)
            v, mem, t = verdict(T)
            mark = {"OK": "OK", "内存不足": "MEM", "太慢": "SLOW"}[v]
            line += " %9s" % mark
        print(line)

    print("\n[2] 逐档明细（只列判定 OK 的，按 T 排序）")
    print("  %-12s %8s %9s %12s %14s %12s"
          % ("分辨率", "帧数", "秒数", "T", "注意力矩阵", "前向(30层)"))
    print("  " + "-" * 78)
    ok = []
    for (H, W) in resolutions:
        for F in frame_list:
            T = tokens(F, H, W)
            v, mem, t = verdict(T)
            if v == "OK":
                ok.append((H, W, F, T, mem, t))
    for (H, W, F, T, mem, t) in sorted(ok, key=lambda r: -r[3]):
        print("  %-12s %8d %9.1f %12d %11.3f GB %11.1f s"
              % ("%dx%d" % (H, W), F, F / FPS, T, mem, t))

    # ---- 反解：最大分辨率 @ 目标时长；最大时长 @ 目标分辨率 ----
    print("\n[3] 反解")
    for secs in (2, 5, 10, 15):
        F = secs * FPS
        best = None
        for (H, W) in resolutions:
            v, mem, t = verdict(tokens(F, H, W))
            if v == "OK":
                best = (H, W)
        print("  %2d 秒: 最大可行分辨率 = %s"
              % (secs, ("%dx%d" % best) if best else "无（连 128x128 都不行）"))
    print()
    for (H, W) in ((256, 256), (384, 384), (480, 480), (512, 512)):
        Fmax = None
        for F in range(16, 401, 1):
            v, _, _ = verdict(tokens(F, H, W))
            if v != "OK":
                break
            Fmax = F
        print("  %-10s: 最大时长 = %s"
              % ("%dx%d" % (H, W),
                 ("%.1f 秒（%d 帧）" % (Fmax / FPS, Fmax)) if Fmax else "无"))

    print("\n" + "=" * 96)
    print("读法: MEM=注意力矩阵超预算；SLOW=矩阵放得下但单次前向注意力超过 %.0f s。" % TIME_LIMIT_S)
    print("      真正的采样还要乘步数（LCM 4 步 / 常规 20-50 步）。")
    print("      ⇒ 上表 OK 的档，实际生成时间要再乘 4~50。")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
