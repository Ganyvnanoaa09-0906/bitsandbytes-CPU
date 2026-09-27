# -*- coding: utf-8 -*-
"""sr_cost.py — 超分能不能救 720p/1080p？先量 SR 的成本

思路（用户的提议，而且是对的思路）:
    Wan 直接出 720p/1080p 已实测不可行（720p 需 37 TB 注意力矩阵、5.1 天计算）。
    但超分把「**生成分辨率**」与「**输出分辨率**」解耦：
        Wan 只出它能出的低分辨率  →  Real-ESRGAN 4× 放大  →  逐帧拼成高分辨率视频
    这样 Wan 的 O(T²) 只作用在低分辨率上，而 SR 是**逐帧、O(pixel)** 的卷积网络。

本脚本量的是**这条路唯一的未知数：SR 在 CPU 上要多久**。
    判据: 单帧 SR 耗时 × 帧数 与"可以接受"比。同时给出与 Wan 生成一步的对比，
          说明 SR 是否真的是更便宜的那一半。

资产（本地已确认）:
    RealESRGAN_x4plus.pth          67.0 MB  RRDBNet 23 块（通用）
    RealESRGAN_x4plus_anime_6B.pth 17.9 MB  RRDBNet  6 块（动漫，轻）

oracle:
    · 复用仓库已有的 RRDBNet 内联实现（basicsr 需编译，这里不装）；
    · 用 min-of-N 计时；
    · 同时测 6B 与 23 块两个模型，给出速度/体积差；
    · 输出必须 finite 且尺寸正确（512×4=2048）——先确认能跑对再谈快慢。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

torch.set_num_threads(int(os.environ.get("THREADS", "6")))

ANIME_6B = r"D:\work\textmodel\RealESRGAN_x4plus_anime_6B.pth"
X4PLUS = r"D:\work\textmodel\RealESRGAN_x4plus.pth"


# ---- 复用仓库里已有的内联 RRDBNet（BasicSR 布局）----
def make_layer(block, n, **kw):
    return nn.Sequential(*[block(**kw) for _ in range(n)])


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


def build_rrdb(num_block=6, num_feat=64, num_grow_ch=32, scale=4):
    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = make_layer(RRDB, num_block, num_feat=num_feat,
                                   num_grow_ch=num_grow_ch)
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def load(path, num_block, num_feat=64):
    m = build_rrdb(num_block=num_block, num_feat=num_feat)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in sd else ("params" if "params" in sd else None)
    if key:
        sd = sd[key]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    return m, n, len(missing), len(unexpected)


def timed(fn, reps=3):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    print("=" * 90)
    print("超分能不能救 720p/1080p？—— 先量 SR 成本")
    print("=" * 90)

    models = []
    for tag, path, nb in (("anime_6B (6块)", ANIME_6B, 6), ("x4plus (23块)", X4PLUS, 23)):
        if not os.path.isfile(path):
            print("  %-16s 权重缺失: %s" % (tag, path))
            continue
        try:
            m, npar, miss, unexp = load(path, nb)
            print("[load] %-16s 参数 %.2fM  missing=%d unexpected=%d"
                  % (tag, npar / 1e6, miss, unexp))
            models.append((tag, m))
        except Exception as e:
            print("[load] %-16s 失败: %s: %s" % (tag, type(e).__name__, e))

    if not models:
        print("没有可用 SR 模型")
        return 1

    print("\n[1] 单帧 SR 成本（4× 放大，输入尺寸 → 输出 4×）")
    print("  %-16s %10s %12s %14s %14s %10s"
          % ("模型", "输入", "输出", "单帧 ms", "每像素 ns", "像素/秒"))
    print("  " + "-" * 82)
    rows = []
    for tag, m in models:
        for S in (128, 256, 384, 480):
            x = torch.randn(1, 3, S, S)
            try:
                with torch.no_grad():
                    y = m(x)
                ok = (y.shape[-1] == S * 4) and bool(torch.isfinite(y).all())
                t = timed(lambda: m(x), reps=3)
                out_px = (S * 4) ** 2
                rows.append((tag, S, t, out_px))
                print("  %-16s %10s %12s %14.1f %14.2f %10.0f"
                      % (tag, "%dx%d" % (S, S), "%dx%d" % (S * 4, S * 4),
                         t * 1e3, t * 1e9 / out_px, out_px / t))
                if not ok:
                    print("      ⚠ 输出形状/数值异常!")
            except Exception as e:
                print("  %-16s %10s  FAILED %s" % (tag, "%dx%d" % (S, S), e))
            del x
    print("  注: '每像素 ns' 按**输出**像素算 ⇒ 可与生成的每像素成本直接比。")

    print("\n[2] 一次 720p / 1080p 视频的 SR 总成本（按最大帧数）")
    print("  %-14s %10s %14s %16s %16s"
          % ("目标", "帧数", "单帧(256源)", "6块 总时间", "23块 总时间"))
    print("  " + "-" * 76)
    # 从 256x256 源放大到目标：需要多少倍？4x 一次到 1024；720p 需再放大
    def per_frame(tag, S=256):
        for (t_, s_, tt, opx) in rows:
            if t_ == tag and s_ == S:
                return tt
        return float("nan")
    t6 = per_frame("anime_6B (6块)", 256)
    t23 = per_frame("x4plus (23块)", 256)
    for name, frames, note in (("480p 15s", 240, "256→1024"), ("720p 15s", 240, "256→1024+"),
                               ("1080p 15s", 240, "256→1024++")):
        def fmt(x):
            if x != x:
                return "n/a"
            if x < 60:
                return "%.0f 秒" % x
            if x < 3600:
                return "%.1f 分" % (x / 60)
            if x < 86400:
                return "%.1f 小时" % (x / 3600)
            return "%.1f 天" % (x / 86400)
        print("  %-14s %10d %14s %16s %16s"
              % (name, frames, "%.0f ms" % (t6 * 1e3) if t6 == t6 else "n/a",
                 fmt(t6 * frames), fmt(t23 * frames)))

    print("\n[3] 与 Wan 生成一步的对比（说明 SR 是不是更便宜的那一半）")
    print("  实测已知: Wan 在 T=8192（256x256x32帧量级）时**单层** attention SDPA 1.598 s")
    print("            ⇒ 30 层 ≈ 48 s/步")
    print("  对照: 256x256 单帧 SR = %.0f ms（6块）" % (t6 * 1e3))
    print("  ⇒ SR 比 Wan 一步便宜 %.0f 倍" % (48 / t6) if t6 == t6 else "  ⇒ n/a")
    print("\n" + "=" * 90)
    print("判据: 若 SR 总时间 << Wan 生成时间 ⇒ 用户思路成立：")
    print("      Wan 出低分辨率（可行包络内），SR 逐帧放大到 720p/1080p。")
    print("      若 SR 本身就要几小时/几天 ⇒ 这条路同样不通。")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
