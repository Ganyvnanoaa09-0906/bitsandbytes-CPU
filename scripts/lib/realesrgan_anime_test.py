# -*- coding: utf-8 -*-
"""Real-ESRGAN x4plus anime 6B test, vs the FSRCNN detail-restoration baseline.

basicsr/realesrgan are NOT installed here (basicsr needs a compile step), so the standard
RRDBNet architecture is implemented inline. Real-ESRGAN anime 6B is the "6 block" variant of
RRDBNet (num_block=6 instead of 23), which is why the checkpoint is only ~17.9 MB.

Both restorers are compared the same way: 4x upscale then bicubic back to the source size,
so the injected detail is comparable at identical output resolution.

Watch for: over-sharpening halos and whether hair strands look more natural than FSRCNN's.

Usage:
  python realesrgan_anime_test.py --src <frames dir> --picks 9,30 --out <dir>
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from detail_restore import load_sr, lapvar  # noqa: E402

ANIME_6B = r"D:\work\textmodel\RealESRGAN_x4plus_anime_6B.pth"


# ---------------------------------------------------------------- RRDBNet (BasicSR layout)
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


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6,
                 num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_realesrgan(path=ANIME_6B, num_block=6):
    sd = torch.load(path, map_location="cpu")
    for k in ("params_ema", "params", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    net = RRDBNet(num_block=num_block)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[load] {os.path.basename(path)}  blocks={num_block}  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:4]:
        print("       missing e.g.:", missing[:4])
    if unexpected[:4]:
        print("       unexpected e.g.:", unexpected[:4])
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--picks", default="9,30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--num-block", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    rr = load_realesrgan(num_block=a.num_block)
    fs = load_sr()
    picks = [int(x) for x in a.picks.split(",")]

    print(f"\n{'frame':>6} {'restorer':>18} {'Laplacian':>10} {'x src':>7} {'sec':>7}")
    print("-" * 55)
    for pi in picks:
        pil = Image.open(os.path.join(a.src, f"f{pi:03d}.png")).convert("RGB")
        sl = lapvar(pil)
        print(f"{pi:>6} {'source':>18} {sl:>10.1f} {1.0:>7.2f} {'-':>7}")
        t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        t0 = time.time()
        with torch.no_grad():
            up = rr(t)
        dn = F.interpolate(up, size=pil.size[::-1], mode="bicubic", align_corners=False)
        arr = (dn.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(a.out, f"f{pi:03d}_real_anime6B_down.png"))
        print(f"{pi:>6} {'RealESRGAN anime6B':>18} {lapvar(arr):>10.1f} "
              f"{lapvar(arr)/sl:>7.2f} {time.time()-t0:>7.1f}")

        t0 = time.time()
        arr2 = (up.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr2).save(os.path.join(a.out, f"f{pi:03d}_real_anime6B_2048.png"))
        print(f"{pi:>6} {'  (native 2048 out)':>18} {lapvar(arr2):>10.1f} "
              f"{'-':>7} {time.time()-t0:>7.1f}")

        t0 = time.time()
        with torch.no_grad():
            fup = fs(t)
        fdn = F.interpolate(fup, size=pil.size[::-1], mode="bicubic", align_corners=False)
        arr3 = (fdn.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr3).save(os.path.join(a.out, f"f{pi:03d}_fsrcnn_down.png"))
        print(f"{pi:>6} {'FSRCNN (baseline)':>18} {lapvar(arr3):>10.1f} "
              f"{lapvar(arr3)/sl:>7.2f} {time.time()-t0:>7.1f}")
        print()

    print(f"[save] {a.out}")


if __name__ == "__main__":
    main()
