# -*- coding: utf-8 -*-
"""mix_datasets.py - 按配比从多个数据池抽样，输出统一训练清单。

设计原则
--------
1. **按池子分别抽样**，不混在一个大池里随机抽（否则配比会失真）
2. **按原生分辨率分桶**：分辨率差异大的池子混训时，小图被上采样会变虚。
   本工具会报告各桶占比，并支持 --min-side 直接排除过小的图。
3. **不复制文件、不移动文件**，只输出清单（manifest.jsonl），
   让后续 tokenize/训练自己读清单 —— 避免再多占一份磁盘。

用法
----
  python mix_datasets.py ^
    --pool "coser=D:\\work\\coser_1024:0.5" ^
    --pool "nsfw_real=D:\\work\\sujvji\\nsfw_dataset:0.3" ^
    --pool "gravure=D:\\work\\gravure_1024:0.2" ^
    --out mix_manifest.jsonl --min-side 256

  # 只报告，不生成清单
  python mix_datasets.py --pool "a=<dir>:1.0" --report-only
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, collections, json, os, random, sys

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
SCREEN_BUCKETS = [(0, 256, '<256 不可用'), (256, 512, '256~512'), (512, 1024, '512~1024'),
                  (1024, 2048, '1024~2048'), (2048, 10 ** 9, '>=2048')]


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def size_of(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def probe_size(path):
    """只读图片头拿宽高，不解码整张图。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def bucket(w, h):
    m = min(w, h)
    for lo, hi, name in SCREEN_BUCKETS:
        if lo <= m < hi:
            return name
    return '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', action='append', required=True,
                    help='"名字=目录:比例" 形式，可重复。比例之和会自动归一化。')
    ap.add_argument('--out', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-side', type=int, default=0, help='排除短边小于此值的图')
    ap.add_argument('--max-per-pool', type=int, default=0, help='每个池子上限（0=不限）')
    ap.add_argument('--total', type=int, default=0,
                    help='目标总张数（0=取配比允许的最大值，即最紧张的池子被抽干）')
    ap.add_argument('--probe', type=int, default=0,
                    help='每个池子抽样多少张来探测分辨率分布（0=不探测）')
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()

    random.seed(a.seed)
    pools = []
    for spec in a.pool:
        if '=' not in spec or ':' not in spec.rsplit('=', 1)[1]:
            raise SystemExit('格式错误: %s  （应为 "名字=目录:比例"）' % spec)
        name, rest = spec.split('=', 1)
        path, ratio = rest.rsplit(':', 1)
        pools.append((name.strip(), path.strip(), float(ratio)))

    rs = sum(r for _, _, r in pools)
    if rs <= 0:
        raise SystemExit('比例之和必须 > 0')

    print('=== 数据池 ===')
    infos = []
    for name, path, ratio in pools:
        if not os.path.isdir(path):
            print('  [!] %-14s 目录不存在: %s' % (name, path))
            infos.append((name, path, ratio, []))
            continue
        fs = list_images(path)
        infos.append((name, path, ratio, fs))
        print('  %-14s 比例 %5.1f%%   %6d 张   %s' % (name, ratio / rs * 100, len(fs), path))

    if a.probe:
        print('\n=== 分辨率分布（每池抽样 %d 张）===' % a.probe)
        for name, path, ratio, fs in infos:
            if not fs:
                continue
            sample = random.sample(fs, min(a.probe, len(fs)))
            c = collections.Counter()
            for p in sample:
                w, h = probe_size(p)
                c[bucket(w, h)] += 1
            tot = sum(c.values())
            desc = '  '.join('%s %d(%.0f%%)' % (k, v, v / tot * 100)
                             for k, v in sorted(c.items()))
            print('  %-14s %s' % (name, desc))

    # 抽样：要在"最终配比尽量贴近给定比例"和"不超出各池大小"之间取平衡。
    # 设目标总量 N，第 i 池需要 N*r_i 张，受 n_i 限制。
    # N 的最大可行值由最紧张的池子决定：N_max = min_i( n_i / r_i )
    live = [(name, ratio, fs) for name, path, ratio, fs in infos if fs]
    if not live:
        print('\n没有任何可用图片')
        return
    N_max = int(min(len(fs) / (r / rs) for _, r, fs in live if r > 0))
    N = a.total if a.total else N_max
    N = min(N, N_max)
    print('\n=== 抽样（目标总量 %d，可行上限 %d）===' % (N, N_max))
    chosen = []
    for name, ratio, fs in live:
        want = int(round(N * (ratio / rs)))
        want = min(want, len(fs))
        if a.max_per_pool:
            want = min(want, a.max_per_pool)
        sel = random.sample(fs, want) if want < len(fs) else list(fs)
        chosen.append((name, sel))
        print('  %-14s 池内 %6d 张  ->  抽 %6d 张' % (name, len(fs), len(sel)))

    # 过滤
    if a.min_side:
        before = sum(len(s) for _, s in chosen)
        out = []
        for name, sel in chosen:
            keep, drop = [], 0
            for p in sel:
                w, h = probe_size(p)
                if min(w, h) >= a.min_side:
                    keep.append(p)
                else:
                    drop += 1
            out.append((name, keep))
            print('  %-14s min-side 过滤: 保留 %d, 剔除 %d' % (name, len(keep), drop))
        chosen = out

    total = sum(len(s) for _, s in chosen)
    print('\n=== 最终配比 ===')
    for name, sel in chosen:
        print('  %-14s %6d 张   %5.1f%%' % (name, len(sel), len(sel) / max(1, total) * 100))
    print('  %-14s %6d 张' % ('合计', total))
    if total == 0:
        return
    if a.report_only:
        return
    if not a.out:
        print('\n(未指定 --out，只报告不产出清单)')
        return

    with open(a.out, 'w', encoding='utf-8') as f:
        for name, sel in chosen:
            for p in sel:
                f.write(json.dumps(dict(path=p, pool=name), ensure_ascii=False) + '\n')
    sz = sum(size_of(p) for _, sel in chosen for p in sel)
    print('\n-> %s   共 %d 条，原始体积 %.2f GB（不复制文件，只出清单）' % (
        a.out, total, sz / 1e9))


if __name__ == '__main__':
    main()
