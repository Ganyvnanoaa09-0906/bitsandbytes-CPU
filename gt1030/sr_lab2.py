# -*- coding: utf-8 -*-
"""sr_lab2.py — 加载社区 upscaler（basicsr 键名：model.0 / model.1.sub.N.RDBk.convJ.0），
与 RealESRGAN_x4plus 在同一源图上对比，统一输出 1920×1080。

键名映射（basicsr 风格）：
  model.0  -> conv_first
  model.1.sub.N.RDBk.convJ.0 -> body.N.rdbK.convJ
  model.3  -> ups.0   model.6 -> ups.1   model.8 -> conv_hr   model.10 -> conv_last
注意：这类权重没有 conv_body，前向是 feat = feat + body(feat) 直接相加。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RDB(nn.Module):
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
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = RDB(nf, gc)
        self.rdb2 = RDB(nf, gc)
        self.rdb3 = RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class Net(nn.Module):
    def __init__(self, num_block=23, nf=64, gc=32, ups=2):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
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


def load_basicsr(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "params_ema" in sd:
        sd = sd["params_ema"]
    elif "params" in sd:
        sd = sd["params"]
    nb = 0
    for k in sd:
        m = re.match(r"model\.1\.sub\.(\d+)\.RDB", k)
        if m:
            nb = max(nb, int(m.group(1)) + 1)
    ups = 2 if "model.6.weight" in sd else 1
    net = Net(num_block=nb, ups=ups)
    new = {}
    for k, v in sd.items():
        if k.startswith("model.0."):
            new["conv_first." + k[len("model.0."):]] = v
        elif k.startswith("model.1.sub."):
            m = re.match(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)$", k)
            if m:
                new[f"body.{m.group(1)}.rdb{m.group(2)}.conv{m.group(3)}.{m.group(4)}"] = v
            else:
                # body 里的最后一项是 conv_body（sub.23.weight / bias 这种单卷积）
                m2 = re.match(r"model\.1\.sub\.(\d+)\.(weight|bias)$", k)
                if m2:
                    new["conv_body." + m2.group(2)] = v
        elif k.startswith("model.3."):
            new["ups.0." + k[len("model.3."):]] = v
        elif k.startswith("model.6."):
            new["ups.1." + k[len("model.6."):]] = v
        elif k.startswith("model.8."):
            new["conv_hr." + k[len("model.8."):]] = v
        elif k.startswith("model.10."):
            new["conv_last." + k[len("model.10."):]] = v
    missing, unexpected = net.load_state_dict(new, strict=False)
    net.eval()
    print(f"  {os.path.basename(path)}: blocks={nb} ups={ups} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return net, 2 ** ups


def sharp(a):
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
    src = Image.open(a.src).convert("RGB")
    x = torch.from_numpy(np.asarray(src, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    models = [
        ("ultrasharp", r"D:\work\textmodel\sr-extra\4x-UltraSharp.pth"),
        ("remacri", r"D:\work\textmodel\sr-extra\4x_foolhardy_Remacri.pth"),
        ("nmkd_siax", r"D:\work\textmodel\sr-extra\4x_NMKD-Siax_200k.pth"),
    ]
    # 双三次基线（同尺寸），给锐度一个参照
    bic = src.resize((a.target_w, a.target_h), Image.BICUBIC)
    rows = [dict(name="bicubic", scale=1, seconds=0.0, sharpness=round(sharp(np.asarray(bic, np.float32)), 1),
                 out=os.path.join(a.outdir, "bicubic.png"))]
    bic.save(rows[0]["out"])
    imgs = [("bicubic", bic)]

    for name, path in models:
        if not os.path.exists(path):
            print(f"  [skip] {name}"); continue
        out_png = os.path.join(a.outdir, f"{name}.png")
        net, scale = load_basicsr(path)
        t0 = time.time()
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        dt = time.time() - t0
        im = Image.fromarray((y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        im = im.resize((a.target_w, max(a.target_h, round(a.target_w * im.height / im.width))), Image.LANCZOS)
        if im.height > a.target_h:
            top = (im.height - a.target_h) // 2
            im = im.crop((0, top, a.target_w, top + a.target_h))
        im.save(out_png)
        sh = sharp(np.asarray(im, np.float32))
        rows.append(dict(name=name, scale=scale, seconds=round(dt, 1), sharpness=round(sh, 1), out=out_png))
        imgs.append((name, im))
        print(f"  {name}: {scale}× | {dt:.1f}s | 锐度 {sh:.0f}", flush=True)
        del net

    crops = []
    for name, im in imgs:
        w, h = im.size
        box = (int(w * 0.30), int(h * 0.20), int(w * 0.70), int(h * 0.60))
        crops.append((name, im.crop(box).resize((420, 420), Image.LANCZOS)))
    sheet = Image.new("RGB", (420 * len(crops), 430), "white")
    for i, (n, c) in enumerate(crops):
        sheet.paste(c, (i * 420, 0))
    sheet.save(os.path.join(a.outdir, "crops2.png"))
    with open(os.path.join(a.outdir, "report2.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
