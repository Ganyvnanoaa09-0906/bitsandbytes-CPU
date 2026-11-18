# -*- coding: utf-8 -*-
"""sr_models.py — 统一超分模型加载器（供 qwen_img.py 与实验脚本共用）。

支持两种权重布局：
  A) realesrgan 风格（x4plus / anime6B）：conv_first / body.N.rdbK.convJ / conv_body / conv_up1/2 / conv_hr / conv_last
  B) basicsr 风格（4x-UltraSharp / Remacri / NMKD）：model.0 / model.1.sub.N.RDBK.convJ.0 / model.3 / model.6 / model.8 / model.10
返回 (net, scale)。net 输入 0..1 的 NCHW float，输出同 range。

用法：
  from sr_models import load
  net, scale = load('ultrasharp')
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = {
    "x4plus": r"D:\work\textmodel\RealESRGAN_x4plus.pth",
    "anime6b": r"D:\work\textmodel\RealESRGAN_x4plus_anime_6B.pth",
    "ultrasharp": r"D:\work\textmodel\sr-extra\4x-UltraSharp.pth",
    "remacri": r"D:\work\textmodel\sr-extra\4x_foolhardy_Remacri.pth",
    "nmkd": r"D:\work\textmodel\sr-extra\4x_NMKD-Siax_200k.pth",
}
DEFAULT = "ultrasharp"


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
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
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


def _load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("params_ema", "params"):
        if k in sd:
            return sd[k]
    return sd


def _map_realesrgan(sd):
    """conv_upN -> ups.N，其余同名。"""
    out = {}
    for k, v in sd.items():
        if k.startswith("conv_up") and (k.endswith(".weight") or k.endswith(".bias")):
            m = re.match(r"conv_up(\d)\.(weight|bias)$", k)
            out["ups.%d.%s" % (int(m.group(1)) - 1, m.group(2))] = v
        else:
            out[k] = v
    return out


def _map_basicsr(sd):
    """model.* -> 本实现的字段名。"""
    out = {}
    nb = 0
    for k, v in sd.items():
        if k.startswith("model.0."):
            out["conv_first." + k[len("model.0."):]] = v
        elif k.startswith("model.1.sub."):
            m = re.match(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)$", k)
            if m:
                nb = max(nb, int(m.group(1)) + 1)
                out["body.%s.rdb%s.conv%s.%s" % (m.group(1), m.group(2), m.group(3), m.group(4))] = v
            else:
                m2 = re.match(r"model\.1\.sub\.(\d+)\.(weight|bias)$", k)
                if m2:
                    out["conv_body." + m2.group(2)] = v
        elif k.startswith("model.3."):
            out["ups.0." + k[len("model.3."):]] = v
        elif k.startswith("model.6."):
            out["ups.1." + k[len("model.6."):]] = v
        elif k.startswith("model.8."):
            out["conv_hr." + k[len("model.8."):]] = v
        elif k.startswith("model.10."):
            out["conv_last." + k[len("model.10."):]] = v
    return out, nb


def load(name=DEFAULT, verbose=False):
    path = WEIGHTS.get(name.lower())
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"超分权重缺失: {name} -> {path}")
    sd = _load_state(path)
    if any(k.startswith("model.") for k in sd):
        sd2, nb = _map_basicsr(sd)
        ups = 2 if "ups.1.weight" in sd2 else 1
        net = RRDBNet(num_block=nb, ups=ups)
        sd2 = _map_realesrgan(sd2) if "conv_up" in str(list(sd2)[:1]) else sd2
    else:
        nb = 0
        for k in sd:
            m = re.match(r"body\.(\d+)\.", k)
            if m:
                nb = max(nb, int(m.group(1)) + 1)
        ups = sum(1 for k in sd if k.startswith("conv_up") and k.endswith(".weight"))
        net = RRDBNet(num_block=nb, ups=ups)
        sd2 = _map_realesrgan(sd)
    missing, unexpected = net.load_state_dict(sd2, strict=False)
    net.eval()
    if verbose:
        print(f"  SR {name}: blocks={nb} ups={ups} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return net, 2 ** ups


def upscale_array(arr, name=DEFAULT, out_w=None, out_h=None, threads=6):
    """arr: HxWx3 uint8 -> 超分后（可选缩到 out_w×out_h）的 uint8 数组。"""
    import numpy as np
    from PIL import Image
    torch.set_num_threads(threads)
    net, scale = load(name)
    x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = net(x).clamp(0, 1)
    im = Image.fromarray((y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    if out_w and out_h:
        im = im.resize((out_w, max(out_h, round(out_w * im.height / im.width))), Image.LANCZOS)
        if im.height > out_h:
            top = (im.height - out_h) // 2
            im = im.crop((0, top, out_w, top + out_h))
    return np.asarray(im), scale
