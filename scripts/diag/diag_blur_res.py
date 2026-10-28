# -*- coding: utf-8 -*-
"""diag_blur_res.py - 验证「blur 拒绝是否由降采样倍率造成」。

对同一批图，分别在**原始分辨率**和 **res=512 缩放后**计算 Laplacian 方差，
看被拒图和保留图的差异到底来自「图本身糊」还是「缩得太狠」。
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
import json, os, random, sys
import numpy as np
from PIL import Image

BASE = r'D:\work\bitsandbytes-CPU\.autopilot'
RES = 512


def lapvar(a):
    g = a.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(2)
    out = np.zeros_like(g)
    out[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
                       - 4.0 * g[1:-1, 1:-1])
    return float(out[1:-1, 1:-1].var())


def load(p):
    with Image.open(p) as im:
        im = im.convert('RGB')
        nat = np.asarray(im)
        s = im.copy()
        s.thumbnail((RES, RES))
        return nat, np.asarray(s), im.size


def main():
    random.seed(0)
    rej_all = [json.loads(l) for l in open(os.path.join(BASE, 'clean_dl', 'rejected.jsonl'),
                                           encoding='utf-8') if l.strip()]
    blur_rej = [r for r in rej_all if 'blur' in (r.get('reasons') or [])]
    print('rejected.jsonl 共 %d 条, 其中 blur %d 条' % (len(rej_all), len(blur_rej)))

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    samp = random.sample(blur_rej, min(n, len(blur_rej)))

    print()
    print('=== 被判 blur 的图 ===')
    print('%-14s %8s %6s %11s %11s %8s' % ('file', 'native', 'scale', 'lap@native', 'lap@512', 'ratio'))
    nat_l, r512_l, scales = [], [], []
    for r in samp:
        try:
            nat, s512, (w, h) = load(r['src'])
            sc = min(w, h) / float(RES)
            a, b = lapvar(nat), lapvar(s512)
            nat_l.append(a); r512_l.append(b); scales.append(sc)
            print('%-14s %8s %6.2f %11.5f %11.5f %8.2f' % (
                os.path.basename(r['src'])[:14], '%dx%d' % (w, h), sc, a, b, a / max(b, 1e-9)))
        except Exception as e:
            print('  ERR', r['src'], e)

    print()
    print('  被判 blur：lap@native 中位 %.5f | lap@512 中位 %.5f | 降采样倍率中位 %.2fx'
          % (np.median(nat_l), np.median(r512_l), np.median(scales)))
    print('  -> 若 lap@native 明显高于阈值 0.003，说明这些图**本来不糊**，是缩放压出来的。')


if __name__ == '__main__':
    main()
