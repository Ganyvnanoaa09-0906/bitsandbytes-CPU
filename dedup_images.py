# -*- coding: utf-8 -*-
"""dedup_images.py - 感知哈希去重（dHash + 汉明距离）。

dHash: 缩到 9x8 灰度，比较水平相邻像素 -> 64 bit。对缩放/轻微压缩不敏感。
汉明距离 <= threshold 判为同一张（或几乎同一张）。

用法:
  python dedup_images.py --data <目录> [--out dup.jsonl] [--threshold 4] [--move-to <目录>]
默认只报告，不移动文件；加 --move-to 才把重复项移走（保留每组的代表图）。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def list_many(roots):
    """支持多个根目录（用 | 或 , 分隔），结果去重后排序。"""
    out = []
    for r in roots:
        out.extend(list_images(r))
    return sorted(set(out))


def dhash(path, hs=8, ws=8):
    im = Image.open(path).convert('L').resize((ws + 1, hs), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def ham(a, b):
    return int(np.count_nonzero(a != b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='图片目录；多个目录用 | 分隔（跨池去重）')
    ap.add_argument('--out', default='')
    ap.add_argument('--threshold', type=int, default=4)
    ap.add_argument('--move-to', default='', help='把重复项移到此目录（保留代表图）')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    roots = [x for x in a.data.split('|') if x.strip()]
    paths = list_many(roots)
    print('数据源 %d 个，图片合计 %d 张' % (len(roots), len(paths)))
    hashes, ok = [], []
    for i, p in enumerate(paths):
        try:
            hashes.append(dhash(p)); ok.append(p)
        except Exception:
            pass
        if (i + 1) % 2000 == 0:
            print('  %d/%d' % (i + 1, len(paths)), flush=True)

    n = len(ok)
    H = np.stack(hashes)
    print('哈希完成 (%d 张)，两两比较 %d 对 ...' % (n, n * (n - 1) // 2), flush=True)

    # 按文件大小降序，优先把「体积最大」的作为代表（保留分辨率最高的）
    order = sorted(range(n), key=lambda i: -os.path.getsize(ok[i]))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # 分块比较，控制内存
    CH = 512
    for s in range(0, n, CH):
        blk = H[s:s + CH]
        d = (blk[:, None, :] != H[None, :, :]).sum(axis=2)   # (CH, n)
        for i in range(d.shape[0]):
            js = np.nonzero(d[i] <= a.threshold)[0]
            for j in js:
                union(order.index(s + i) if False else (s + i), int(j))
        if (s // CH) % 5 == 0:
            print('  比较 %d/%d' % (min(s + CH, n), n), flush=True)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    total_dup = sum(len(v) - 1 for v in dups.values())
    print()
    print('重复组数      : %d' % len(dups))
    print('可移除的重复项: %d  (占 %.1f%%)' % (total_dup, total_dup / n * 100))
    print('去重后剩余    : %d' % (n - total_dup))

    reps = {}
    for k, v in dups.items():
        v.sort(key=lambda i: -os.path.getsize(ok[i]))
        reps[k] = ok[v[0]]
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            for k, v in dups.items():
                f.write(json.dumps(dict(keep=reps[k],
                                        dup=[ok[i] for i in v[1:]],
                                        n=len(v)), ensure_ascii=False) + '\n')

    sizes = [len(v) for v in dups.values()]
    import collections
    print('重复规模分布  :', dict(sorted(collections.Counter(sizes).items())))
    if sizes:
        print('最大一组      : %d 张' % max(sizes))

    if a.move_to:
        os.makedirs(a.move_to, exist_ok=True)
        import shutil
        moved = 0
        for k, v in dups.items():
            for i in v[1:]:
                dst = os.path.join(a.move_to, os.path.basename(ok[i]))
                if os.path.exists(dst):
                    dst = os.path.join(a.move_to, '%d_%s' % (moved, os.path.basename(ok[i])))
                shutil.move(ok[i], dst); moved += 1
        print('已移动 %d 张重复项 -> %s' % (moved, a.move_to))


if __name__ == '__main__':
    main()
