# -*- coding: utf-8 -*-
"""sr_lab.py — 换超分：在同一张 896×512 源图上对比多种 RealESRGAN 模型，统一输出到 1920×1080。

模型按 state_dict 自动推断 num_block 与上采样倍数（2/4/8），无需手写配置。
输出：每种路线的时间、锐度、以及统一裁切的对照图。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock(nf, gc) for _ in range(3))

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_block=23, nf=64, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        ups = 1 if scale == 2 else (2 if scale == 4 else 3)
        self.ups = nn.ModuleList([nn.Conv2d(nf, nf, 3, 1, 1) for _ in range(ups)])
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        for up in self.ups:
            feat = self.lrelu(up(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def infer(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in sd else ("params" if "params" in sd else None)
    sd = sd[key] if key else sd
    ups = sum(1 for k in sd if k.startswith("conv_up") and k.endswith(".weight"))
    scale = 2 ** ups
    blocks = 0
    for k in sd:
        if k.startswith("body."):
            try:
                blocks = max(blocks, int(k.split(".")[1]) + 1)
            except Exception:
                pass
    return sd, blocks, scale


def load(path):
    sd, nb, scale = infer(path)
    m = RRDBNet(num_block=nb, scale=scale)
    # 权重用的是 conv_up1/2/3，本实现用 ups.0/1/2，做一次名字映射
    remap = {}
    for k, v in sd.items():
        if k.startswith('conv_up') and k.endswith('.weight'):
            idx = int(k[len('conv_up'):-len('.weight')]) - 1
            remap['ups.%d.weight' % idx] = v
        elif k.startswith('conv_up') and k.endswith('.bias'):
            idx = int(k[len('conv_up'):-len('.bias')]) - 1
            remap['ups.%d.bias' % idx] = v
    sd = {k: v for k, v in sd.items() if not k.startswith('conv_up')}
    sd.update(remap)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    print(f"  load {os.path.basename(path)}: blocks={nb} scale={scale} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return m, scale


def sharpen_metric(a):
    g = a.astype(np.float32).mean(2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\work\research\bench_results\qwen21_16x9_896x512.png")
    ap.add_argument("--outdir", default=r"D:\work\research\bench_results\qwen_sr_lab")
    ap.add_argument("--target-w", type=int, default=1920)
    ap.add_argument("--target-h", type=int, default=1080)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.set_num_threads(6)

    models = [
        ("x4plus", r"D:\work\textmodel\RealESRGAN_x4plus.pth"),
        ("anime6B", r"D:\work\textmodel\RealESRGAN_x4plus_anime_6B.pth"),
        ("ultrasharp", r"D:\work\textmodel\sr-extra\4x-UltraSharp.pth"),
        ("remacri", r"D:\work\textmodel\sr-extra\4x_foolhardy_Remacri.pth"),
        ("nmkd_siax", r"D:\work\textmodel\sr-extra\4x_NMKD-Siax_200k.pth"),
    ]
    src = Image.open(a.src).convert("RGB")
    x = torch.from_numpy(np.asarray(src, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    rows = []
    imgs = []
    for name, path in models:
        if not os.path.exists(path):
            print(f"  [skip] {name} 缺权重"); continue
        out_png = os.path.join(a.outdir, f"{name}.png")
        if os.path.exists(out_png):
            print(f"  [skip] {name} 已存在", flush=True)
            imgs.append((name, Image.open(out_png).convert("RGB")))
            continue
        try:
            m, scale = load(path)
        except Exception as e:
            print(f"  [fail-load] {name}: {repr(e)[:80]}"); continue
        t0 = time.time()
        try:
            with torch.no_grad():
                y = m(x).clamp(0, 1)
        except RuntimeError as e:
            print(f"  [fail-run] {name}: {repr(e)[:80]}"); continue
        dt = time.time() - t0
        im = Image.fromarray((y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        # 统一到目标分辨率：按宽缩放后居中裁高
        im = im.resize((a.target_w, max(a.target_h, round(a.target_w * im.height / im.width))), Image.LANCZOS)
        if im.height > a.target_h:
            top = (im.height - a.target_h) // 2
            im = im.crop((0, top, a.target_w, top + a.target_h))
        im.save(out_png)
        arr = np.asarray(im, np.float32)
        sh = sharpen_metric(arr)
        rows.append(dict(name=name, scale=scale, seconds=round(dt, 1), sharpness=round(sh, 1), out=out_png))
        imgs.append((name, im))
        print(f"  {name}: {scale}× | {dt:.1f}s | 锐度 {sh:.0f}", flush=True)
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 对照：同一区域裁切放大
    crops = []
    for name, im in imgs:
        w, h = im.size
        box = (int(w * 0.30), int(h * 0.20), int(w * 0.70), int(h * 0.60))
        crops.append((name, im.crop(box).resize((420, 420), Image.LANCZOS)))
    if crops:
        sheet = Image.new("RGB", (420 * len(crops), 430), "white")
        for i, (n, c) in enumerate(crops):
            sheet.paste(c, (i * 420, 0))
        sheet.save(os.path.join(a.outdir, "crops.png"))
    with open(os.path.join(a.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
