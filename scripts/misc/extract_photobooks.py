# -*- coding: utf-8 -*-
"""extract_photobooks.py - 批量解压写真集 zip，按集名分目录，并统计。

设计：
  - 每个 zip 解压到 <out>/<zip 名去掉扩展名>/ ，天然保留"哪张属于哪个 coser"的分组
  - 已存在的目录跳过（可重复运行）
  - 输出每集的图片数、短边分布、总体积

用法:
  python extract_photobooks.py --root D:\\work\\sujvji --out D:\\work\\sujvji\\photobook
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
import argparse, collections, os, re, sys, zipfile

IMG = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def safe(name):
    """把 zip 名转成合法的目录名。"""
    name = os.path.splitext(name)[0]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:80] if len(name) > 80 else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    zips = []
    for dp, _, fns in os.walk(a.root):
        for fn in fns:
            if fn.lower().endswith('.zip'):
                # 跳过已经解压出来的目录里的
                if os.path.abspath(os.path.join(a.out)) in os.path.abspath(dp):
                    continue
                zips.append(os.path.join(dp, fn))
    zips.sort()
    print('找到 %d 个 zip' % len(zips))
    os.makedirs(a.out, exist_ok=True)

    total_n = total_b = 0
    rows = []
    for z in zips:
        name = safe(os.path.basename(z))
        dst = os.path.join(a.out, name)
        if os.path.isdir(dst) and not a.overwrite:
            n = sum(1 for f in os.listdir(dst) if f.lower().endswith(IMG))
            sz = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)
                     if f.lower().endswith(IMG))
            print('  [跳过] %-56s 已有 %3d 张' % (name[:56], n))
            rows.append((name, n, sz))
            total_n += n; total_b += sz
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                os.makedirs(dst, exist_ok=True)
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    fn = os.path.basename(info.filename)
                    if not fn:
                        continue
                    with zf.open(info) as src, open(os.path.join(dst, fn), 'wb') as out:
                        out.write(src.read())
            n = sum(1 for f in os.listdir(dst) if f.lower().endswith(IMG))
            sz = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)
                     if f.lower().endswith(IMG))
            rows.append((name, n, sz))
            total_n += n; total_b += sz
            print('  [OK]   %-56s %3d 张 %7.1f MB' % (name[:56], n, sz / 1e6))
        except Exception as e:
            print('  [!!]   %-56s %s' % (name[:56], e))

    print('\n合计: %d 集, %d 张, %.2f GB' % (len(rows), total_n, total_b / 1e9))

    # 分辨率抽样
    try:
        from PIL import Image
        import random
        random.seed(0)
        res = collections.Counter()
        for name, n_, sz_ in rows:
            d = os.path.join(a.out, name)
            if not os.path.isdir(d):
                continue
            fs = [f for f in os.listdir(d) if f.lower().endswith(IMG)]
            for f in random.sample(fs, min(8, len(fs))):
                try:
                    with Image.open(os.path.join(d, f)) as im:
                        res[im.size] += 1
                except Exception:
                    pass
        print('\n分辨率抽样（每集最多 8 张）:')
        for (w, h), c in res.most_common(10):
            print('   %-12s %3d 次  短边 %d' % ('%dx%d' % (w, h), c, min(w, h)))
        shorts = sorted(min(w, h) for (w, h) in res.elements())
        if shorts:
            print('   短边: min %d  中位 %d  max %d' % (shorts[0], shorts[len(shorts) // 2], shorts[-1]))
    except Exception as e:
        print('分辨率抽样失败:', e)


if __name__ == '__main__':
    main()
