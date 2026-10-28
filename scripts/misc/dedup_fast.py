# -*- coding: utf-8 -*-
"""dedup_fast.py — 可扩展的感知哈希去重（banded LSH，替代 O(n²) 的 dedup_images.py）

## 为什么不用现成的 dedup_images.py

它做【全对比较】：`d = (blk[:,None,:] != H[None,:,:]).sum(2)`。
n=69,000 时是 24 亿对，每块还要 512×69000×64 的 bool 张量（2.3 GB）⇒ 要跑几天。
本项目数据从 29k 涨到 69k 后必须换算法。

## 算法：banded LSH（鸽巢原理，对阈值有严格保证）

64 位 dHash 切成 B 段。若两张图的汉明距离 ≤ T，则这 T 个差异位最多落在 T 段里，
⇒ 只要 **B > T**，就【必然至少有一段完全相同】。

取 B=5 段（每段 12~13 位）、T=4 ⇒ 保证不漏（B=5 > T=4）。
于是只需在"每段取值相同"的桶内两两比较 —— 平均桶大小 ≈ n/2^13 ≈ 8，
总比较量 O(n · B · 桶大小) 而不是 O(n²)。

## 保留策略

同一重复组内保留【优先级最低】的那张：
  priority 0 = 已在现有 token 集里的图（保住现有 token 文件的行序，只需追加新的）
  priority 1 = 新清洗出来的图
同优先级下保留文件最大的（通常分辨率最高）。
"""
from __future__ import annotations
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
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
from collections import defaultdict

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def dhash(path, hs=8, ws=8):
    with Image.open(path) as im:
        a = np.asarray(im.convert('L').resize((ws + 1, hs), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def pack_bits(b):
    """(64,) bool -> python int"""
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep', default='', help='优先保留的路径清单（json，如 --keep-paths）')
    ap.add_argument('--keep-paths', default='',
                    help='优先保留的图片路径清单 json（现有 token 集的 paths.json）')
    ap.add_argument('--new', default='', help='新图片目录；多个用 | 分隔')
    ap.add_argument('--out', default='', help='输出：新的 paths.json（去重后的完整清单）')
    ap.add_argument('--threshold', type=int, default=4)
    ap.add_argument('--bands', type=int, default=5)
    ap.add_argument('--dump-dup', default='', help='把重复组写到这个 jsonl（可选）')
    a = ap.parse_args()

    old = json.load(open(a.keep_paths, encoding='utf-8')) if a.keep_paths else []
    newp = []
    for r in [x for x in a.new.split('|') if x.strip()]:
        newp.extend(list_images(r))
    paths = list(old) + newp
    prio = np.array([0] * len(old) + [1] * len(newp), dtype=np.int8)
    print('现有 %d 张 + 新 %d 张 = %d 张' % (len(old), len(newp), len(paths)), flush=True)

    t0 = time.time()
    H, idx_ok = [], []
    for i, p in enumerate(paths):
        try:
            H.append(pack_bits(dhash(p)))
            idx_ok.append(i)
        except Exception:
            pass
        if (i + 1) % 5000 == 0:
            print('  哈希 %d/%d  %.0fs' % (i + 1, len(paths), time.time() - t0), flush=True)
    H = np.array(H, dtype=np.uint64)
    n = len(H)
    print('  哈希完成 %d 张  %.0fs' % (n, time.time() - t0), flush=True)

    # ---- banded LSH ----
    B = max(1, a.bands)
    bits = 64 // B
    mask = np.uint64((1 << bits) - 1)
    cand = defaultdict(list)
    for b in range(B):
        shift = np.uint64(b * bits)
        keys = (H >> shift) & mask
        buckets = defaultdict(list)
        for i, k in enumerate(keys.tolist()):
            buckets[k].append(i)
        for k, v in buckets.items():
            if len(v) > 1:
                cand[(b, k)] = v
    print('  候选桶 %d 个（每段）' % len(cand), flush=True)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    npair = 0
    for (b, k), v in cand.items():
        va = np.array(v)
        A = H[va]
        # 只比同一桶内（已按段相同）
        for ii in range(len(va)):
            d = np.bitwise_xor(A[ii], A).view(np.uint8).reshape(-1, 8)
            pop = np.unpackbits(d, axis=1).sum(1)      # 汉明距离
            for jj in np.nonzero(pop <= a.threshold)[0]:
                union(int(va[ii]), int(va[jj]))
                npair += 1
    print('  比较 %d 对（O(n²) 会是 %d 对，省了 %.4f%%）'
          % (npair, n * (n - 1) // 2, 100 * (1 - npair / max(n * (n - 1) // 2, 1))),
          flush=True)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    size = {p: (os.path.getsize(p) if os.path.exists(p) else 0) for p in paths}

    keep_flags = np.ones(n, dtype=bool)
    n_dup = 0
    n_dup_new = 0
    groups_multi = [v for v in groups.values() if len(v) > 1]
    for v in groups_multi:
        # ★ 组里只要有一张【旧图】，就保留【所有旧图】，只移除新图。
        # 为什么：旧图的行序必须与 tokens_16x16_full.pt 严格一致，
        # 否则训练时的 --val-range 27658:29113 会指向别的图，与 ar_flat256 不可比。
        # （第一版把 23 张旧图当彼此的重复删了，断言直接抓住。）
        old_in = [i for i in v if prio[idx_ok[i]] == 0]
        if old_in:
            for i in v:
                if prio[idx_ok[i]] == 1:
                    keep_flags[i] = False
                    n_dup += 1
                    n_dup_new += 1
        else:
            # 全是新图：保留文件最大的（通常分辨率最高）
            best = min(v, key=lambda i: -size[paths[idx_ok[i]]])
            for i in v:
                if i != best:
                    keep_flags[i] = False
                    n_dup += 1
                    n_dup_new += 1
    print()
    print('重复组 %d   移除 %d 张（其中新图 %d 张）' % (len(groups_multi), n_dup, n_dup_new))

    final = [paths[idx_ok[i]] for i in range(n) if keep_flags[i]]
    # 保序：现有在前（原序），新的在后
    # ⚠️ set(old) 必须提到循环外，否则是 O(n²)（69k 张会跑很久）
    oldset = set(old)
    final_old = [p for p in final if p in oldset]
    final_new = [p for p in final if p not in oldset]
    final = final_old + final_new
    print('去重后 %d 张 = 现有 %d + 新增 %d' % (len(final), len(final_old), len(final_new)))

    if a.out:
        json.dump(final, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
        print('=> %s' % a.out)
    if a.dump_dup:
        with open(a.dump_dup, 'w', encoding='utf-8') as f:
            for v in groups_multi:
                f.write(json.dumps([paths[idx_ok[i]] for i in v], ensure_ascii=False) + '\n')
    print('总耗时 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
