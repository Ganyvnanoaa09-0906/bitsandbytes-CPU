# -*- coding: utf-8 -*-
"""probe_rejected.py - 把 fast_clean 的 rejected.jsonl 里某一类拒绝原因抽样拼成联系表，
用来人工确认「拒绝判定是不是误杀」。

用法:
  python probe_rejected.py --jsonl .autopilot\\clean_dl\\rejected.jsonl --reason blur --out sheet.png
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os, random
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--reason', default='', help='只看含此原因的条目，留空表示全部')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--cell', type=int, default=300)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    rows = []
    with open(a.jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rs = r.get('reasons') or []
            if isinstance(rs, str):
                rs = [rs]
            if a.reason and a.reason not in rs:
                continue
            p = r.get('src') or r.get('image')
            if p and os.path.exists(p):
                rows.append((p, r))
    print('匹配 %d 条 (原因=%s)' % (len(rows), a.reason or 'ALL'))
    if not rows:
        return

    random.seed(a.seed)
    samp = random.sample(rows, min(a.n, len(rows)))
    cols = a.cols
    C, strip = a.cell, 18
    rr = (len(samp) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * C, rr * (C + strip)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (p, rec) in enumerate(samp):
        r_, c_ = divmod(i, cols)
        y0 = r_ * (C + strip)
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                im.thumbnail((C, C))
                sheet.paste(im, (c_ * C + (C - im.width) // 2, y0 + (C - im.height) // 2))
        except Exception:
            d.text((c_ * C + 4, y0 + 4), 'ERR', fill=(255, 80, 80))
        b = rec.get('blur')
        lab = '%s b=%.4f' % (os.path.basename(p)[:22], b if isinstance(b, float) else -1)
        d.text((c_ * C + 3, y0 + C + 4), lab, fill=(200, 200, 200))
    sheet.save(a.out, quality=92)
    print('拼图 -> %s (%d 张)' % (a.out, len(samp)))


if __name__ == '__main__':
    main()
