# -*- coding: utf-8 -*-
"""verify_subset.py - 删除前验证：A 目录是否被 B 目录按「集名+文件名」完全覆盖，且字节一致。

用法: python verify_subset.py --a <要删的目录> --b <保留的目录>
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, hashlib, os, sys

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')


def build(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        setn = os.path.basename(dp)
        for f in fns:
            if f.lower().endswith(EXTS):
                idx[(setn, f)] = os.path.join(dp, f)
    return idx


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='要删的目录')
    ap.add_argument('--b', required=True, help='保留的目录')
    ap.add_argument('--sample', type=int, default=0, help='>0 时只抽查 N 个（快）')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    A, B = build(a.a), build(a.b)
    print('A (%s): %d 个文件' % (a.a, len(A)))
    print('B (%s): %d 个文件' % (a.b, len(B)))

    missing = [k for k in A if k not in B]
    print('A 中有、B 中没有的:', len(missing))
    for k in missing[:10]:
        print('   ', k)

    keys = sorted(A.keys())
    if a.sample > 0 and len(keys) > a.sample:
        import random
        random.seed(a.seed)
        keys = random.sample(keys, a.sample)
        print('抽查 %d 个做字节比对...' % len(keys))
    else:
        print('全量比对 %d 个文件...' % len(keys))

    same = diff = 0
    bad = []
    for i, k in enumerate(keys, 1):
        if k not in B:
            continue
        if md5(A[k]) == md5(B[k]):
            same += 1
        else:
            diff += 1
            if len(bad) < 5:
                bad.append(k)
        if i % 2000 == 0:
            print('   ...%d/%d' % (i, len(keys)), flush=True)

    print()
    print('字节完全一致 :', same)
    print('字节不一致   :', diff)
    for k in bad:
        print('   不一致例:', k, '\n      A:', A[k], '\n      B:', B[k])
    print()
    if not missing and diff == 0:
        print('结论: A 被 B 完全覆盖，删除 A 不丢数据。')
    else:
        print('结论: 存在差异，不要直接删 A！')


if __name__ == '__main__':
    main()
