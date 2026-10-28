# -*- coding: utf-8 -*-
"""dup_pairs_sheet.py - 跨池重复的左右对照图：上排=保留的(dl)，下排=被移除的(webp)。

被移走的文件在 _dups_final 里被重命名成 <集名>_<文件名>.jpg，这里做名字还原。
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, json, os
from PIL import Image, ImageDraw

BASE = r'D:\work\bitsandbytes-CPU\.autopilot'
DUPS = os.path.join(BASE, '_dups_final')


def moved_path(p):
    """把原始路径映射到 _dups_final 里的新位置。"""
    setname = os.path.basename(os.path.dirname(p))
    base = os.path.basename(p)
    for cand in (os.path.join(DUPS, '%s_%s' % (setname, base)),
                 os.path.join(DUPS, base),
                 os.path.join(DUPS, os.path.splitext(base)[0] + '.jpg')):
        if os.path.exists(cand):
            return cand
    return None


def pick_pairs(rows, kp_name, dp_name, pool_of, n):
    out = []
    for r in rows:
        if len(r['dup']) != 1:
            continue
        if pool_of(r['keep']) != kp_name or pool_of(r['dup'][0]) != dp_name:
            continue
        mp = moved_path(r['dup'][0])
        if mp is None:
            continue
        out.append((r['keep'], mp, r['dup'][0]))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(BASE, 'dup_pairs.png'))
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--cell', type=int, default=260)
    a = ap.parse_args()

    POOLS = [('char2', r'clean_char2_1024\images'), ('dl', r'clean_dl\images'),
             ('webp', r'clean_webp\images'), ('photobook', r'sujvji\photobook'),
             ('tu', r'clean_tu\images'), ('za', r'clean_za\images')]

    def pool_of(p):
        for name, frag in POOLS:
            if frag in p:
                return name
        return '?'

    rows = [json.loads(l) for l in open(os.path.join(BASE, 'final_dups.jsonl'), encoding='utf-8')
            if l.strip()]
    pairs = pick_pairs(rows, 'dl', 'webp', pool_of, a.n)
    print('找到 %d 组可显示的 dl->webp 重复对' % len(pairs))
    if not pairs:
        return

    C, strip = a.cell, 16
    cols = len(pairs)
    sheet = Image.new('RGB', (cols * C, 2 * (C + strip)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (kp, mp, orig) in enumerate(pairs):
        for row, (tag, path) in enumerate((('KEEP dl', kp), ('DUP webp', mp))):
            y0 = row * (C + strip)
            try:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    im.thumbnail((C, C))
                    sheet.paste(im, (i * C + (C - im.width) // 2, y0 + (C - im.height) // 2))
            except Exception:
                d.text((i * C + 4, y0 + 4), 'ERR', fill=(255, 80, 80))
            d.text((i * C + 3, y0 + C + 2), tag, fill=(150, 230, 150) if row == 0 else (230, 170, 120))
    sheet.save(a.out, quality=92)
    print('对照图 -> %s' % a.out)
    print()
    print('=== 本次展示的文件对 ===')
    for kp, mp, orig in pairs:
        print('  KEEP dl  : %s' % kp)
        print('  DUP webp : %s   (原名 %s)' % (mp, orig))
        print()


if __name__ == '__main__':
    main()
