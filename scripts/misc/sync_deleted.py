# -*- coding: utf-8 -*-
"""sync_deleted.py - 对账：把 clean_* 里「源文件已被删除」的图清掉。

场景：清洗跑完之后，又回原始目录手工删了图（比如删码图），
clean_* 里就会留下孤儿。本工具按文件名一一对应地清掉它们。

用法:
  python sync_deleted.py --raw D:\\新建文件夹\\images --clean <base>\\clean_nf --dry-run
  python sync_deleted.py --raw D:\\新建文件夹\\images --clean <base>\\clean_nf
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
import argparse, os

IMG = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def stems(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(IMG):
                out.setdefault(os.path.splitext(f)[0], os.path.join(dp, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='原始目录（当前状态，已删过图）')
    ap.add_argument('--clean', required=True, help='清洗输出目录（含 images/ 子目录）')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    raw = stems(a.raw)
    cdir = os.path.join(a.clean, 'images')
    cln = stems(cdir)
    print('原始目录 : %d 张' % len(raw))
    print('清洗目录 : %d 张' % len(cln))

    orphan = {k: v for k, v in cln.items() if k not in raw}
    print('孤儿（源已删）: %d 张' % len(orphan))
    if not orphan:
        print('无需处理。')
        return

    freed = 0
    removed = 0
    for k, p in orphan.items():
        try:
            freed += os.path.getsize(p)
        except OSError:
            pass
        if a.dry_run:
            if removed < 10:
                print('   [dry] %s' % p)
        else:
            try:
                os.remove(p)
            except OSError as e:
                print('   删除失败 %s: %s' % (p, e))
                continue
        removed += 1

    print()
    if a.dry_run:
        print('DRY RUN: 将删除 %d 张，释放 %.1f MB（未实际删除）' % (removed, freed / 1e6))
    else:
        print('已删除 %d 张，释放 %.1f MB' % (removed, freed / 1e6))
        print('清洗目录现在: %d 张' % len(stems(cdir)))


if __name__ == '__main__':
    main()
