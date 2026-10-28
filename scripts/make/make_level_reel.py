# -*- coding: utf-8 -*-
"""Stitch several clips into one review reel (GIF) + a contact-sheet PNG.

Built for the L1..L5 motion-boundary ladder: one row/segment per motion level, hard cut
between them (matching the production "independent chunks + hard cuts" edit), each frame
labelled so it is obvious which level is on screen.

Usage:
  python make_level_reel.py --spec "L1 微表情=D:\\...\\L1_micro\\frames" \
                            "L2 转头=D:\\...\\L2_headturn\\frames" ... \
      --out-gif D:\\work\\output\\reel\\L1toL5.gif --out-sheet D:\\work\\output\\reel\\L1toL5_sheet.png
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                   r"C:\Windows\Fonts\arial.ttf"]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_clip(d):
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
    return [Image.open(os.path.join(d, f)).convert("RGB") for f in files]


def label(img, text, sub=None):
    """Draw a label bar on the image (returns a new image)."""
    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    fs = max(14, H // 18)
    f = _font(fs)
    pad = max(6, H // 48)
    h = fs + 2 * pad + (fs * 7 // 10 if sub else 0)
    dr.rectangle([0, 0, W, h], fill=(0, 0, 0, 165))
    dr.text((pad, pad // 2), text, font=f, fill=(255, 255, 255, 255))
    if sub:
        dr.text((pad, pad // 2 + fs + 2), sub, font=_font(max(11, fs * 7 // 10)),
                fill=(255, 230, 150, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True,
                    help="items of the form '<label>=<frames dir>'")
    ap.add_argument("--sub", nargs="*", default=[], help="optional sub-label per item")
    ap.add_argument("--out-gif", default=None)
    ap.add_argument("--out-sheet", default=None)
    ap.add_argument("--duration", type=int, default=200, help="ms per frame")
    ap.add_argument("--sheet-cell", type=int, default=150)
    a = ap.parse_args()

    clips, names = [], []
    for i, item in enumerate(a.spec):
        if "=" not in item:
            raise SystemExit(f"[err] bad --spec item (need label=dir): {item}")
        name, d = item.split("=", 1)
        if not os.path.isdir(d):
            raise SystemExit(f"[err] missing dir: {d}")
        c = load_clip(d)
        print(f"[load] {name}: {len(c)} frames from {d}")
        clips.append(c)
        sub = a.sub[i] if i < len(a.sub) else None
        # PowerShell drops empty-string arguments, so "-" is the placeholder for "no sub".
        # It also BOM-prefixes the first argument; strip that before comparing.
        if sub is not None:
            sub = sub.lstrip("\ufeff").strip()
        if sub in ("", "-"):
            sub = None
        names.append((name.lstrip("\ufeff"), sub))

    frames = []
    for c, (name, sub) in zip(clips, names):
        for im in c:
            frames.append(label(im, name, sub))
    print(f"[reel] {len(frames)} frames total @ {1000 // a.duration}fps"
          f" = {len(frames) * a.duration / 1000:.1f}s")

    if a.out_gif:
        os.makedirs(os.path.dirname(a.out_gif), exist_ok=True)
        pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
        pal[0].save(a.out_gif, save_all=True, append_images=pal[1:],
                    duration=a.duration, loop=0, optimize=False, disposal=2)
        print(f"[save] {a.out_gif}  ({os.path.getsize(a.out_gif) / 1e6:.1f} MB)")

    if a.out_sheet:
        cell = a.sheet_cell
        ncol = max(len(c) for c in clips)
        fs = max(13, cell // 9)
        head = fs + 10
        sheet = Image.new("RGB", (ncol * cell, len(clips) * (cell + head)), (18, 18, 20))
        dr = ImageDraw.Draw(sheet)
        for r, (c, (name, sub)) in enumerate(zip(clips, names)):
            y0 = r * (cell + head)
            dr.text((6, y0 + 4), name + (f"  {sub}" if sub else ""),
                    font=_font(fs), fill=(255, 235, 160))
            for j, im in enumerate(c[:ncol]):
                sheet.paste(im.resize((cell, cell), Image.LANCZOS), (j * cell, y0 + head))
        os.makedirs(os.path.dirname(a.out_sheet), exist_ok=True)
        sheet.save(a.out_sheet)
        print(f"[save] {a.out_sheet}  {sheet.size}")


if __name__ == "__main__":
    main()
