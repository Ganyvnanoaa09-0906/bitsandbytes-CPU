# -*- coding: utf-8 -*-
"""drop_nonperson.py - 按 find_nonperson.py 的清单，从清洗目录里删除无人图。

清单会保留（可回溯），只删图片。

用法:
  python drop_nonperson.py --jsonl <base>\\nonperson_nf.jsonl --clean <base>\\clean_nf --dry-run
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--clean', required=True, help='清洗输出目录（含 images/）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--keep', type=int, default=0, help='>0 时保留 n<=keep 的（默认删 n==0）')
    a = ap.parse_args()

    cdir = os.path.join(a.clean, 'images')
    thr = a.keep
    victims = []
    total = 0
    with open(a.jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            if r.get('nperson', 1) <= thr:
                victims.append(r['path'])

    print('清单总计 : %d' % total)
    print('待删(无人): %d' % len(victims))

    freed = 0
    gone = 0
    for p in victims:
        # 清单路径本身就是 clean 目录里的路径
        if not os.path.exists(p):
            continue
        try:
            freed += os.path.getsize(p)
        except OSError:
            pass
        if a.dry_run:
            if gone < 5:
                print('   [dry] %s' % os.path.basename(p))
        else:
            try:
                os.remove(p)
            except OSError as e:
                print('   删除失败 %s: %s' % (p, e))
                continue
        gone += 1

    if a.dry_run:
        print('\nDRY RUN: 将删除 %d 张，释放 %.1f MB' % (gone, freed / 1e6))
    else:
        left = sum(1 for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        print('\n已删除 %d 张，释放 %.1f MB' % (gone, freed / 1e6))
        print('clean_nf 现在: %d 张' % left)


if __name__ == '__main__':
    main()
