# -*- coding: utf-8 -*-
"""downscale_archive.py - 把原始图降到指定短边，用于冷存归档。

要点：
  - 用 PIL 的 draft() 走 JPEG 的 DCT 域降采样：只解码需要的那一档分辨率，
    比「全解码再 LANCZOS 缩放」快很多，3000px 大图尤其明显。
  - 不放大：短边已经小于目标的图原样重存。
  - 断点续传：输出已存在的直接跳过，可反复运行。

用法:
  python downscale_archive.py --src D:\\新建文件夹\\images --dst C:\\_archive_1024 --short 1024
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
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def convert(src, dst, short, quality):
    """返回 (读入字节, 写出字节)。dst 已存在则返回 None 表示跳过。"""
    if os.path.exists(dst):
        return None
    with Image.open(src) as im:
        try:
            im.draft('RGB', (short, short))     # JPEG: 解码时就降到该档
        except Exception:
            pass
        im = im.convert('RGB')
        w, h = im.size
        s = short / float(min(w, h))
        if s < 1.0:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst, 'JPEG', quality=quality, optimize=True)
    return (os.path.getsize(src), os.path.getsize(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--short', type=int, default=1024)
    ap.add_argument('--quality', type=int, default=90)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    files = []
    for dp, _, fns in os.walk(a.src):
        for f in fns:
            if f.lower().endswith(EXTS):
                files.append(os.path.join(dp, f))
    files.sort()
    if a.limit:
        files = files[:a.limit]
    print('源: %d 张  目标短边: %dpx  quality: %d  线程: %d'
          % (len(files), a.short, a.quality, a.workers))

    t0 = time.time()
    done = skip = 0
    in_b = out_b = 0
    jobs = [(p, os.path.join(a.dst, os.path.splitext(os.path.relpath(p, a.src))[0] + '.jpg'))
            for p in files]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(convert, s, d, a.short, a.quality): (s, d) for s, d in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                print('  失败 %s: %s' % (futs[fu][0], e), flush=True)
                continue
            if r is None:
                skip += 1
            else:
                done += 1
                in_b += r[0]
                out_b += r[1]
            if i % 500 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = (done + skip) / max(el, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                print('  [%5d/%d] 转换 %d 跳过 %d  %.2f 张/s  输出 %.2f GB  剩约 %.0f 分'
                      % (i, len(jobs), done, skip, rate, out_b / 1e9, eta / 60), flush=True)

    el = time.time() - t0
    print()
    print('完成: 转换 %d, 跳过 %d, 用时 %.1f 分' % (done, skip, el / 60))
    if in_b:
        print('  输入 %.2f GB -> 输出 %.2f GB  (比值 %.4f, 省 %.1f%%)'
              % (in_b / 1e9, out_b / 1e9, out_b / in_b, (1 - out_b / in_b) * 100))
    print('  目标目录: %s' % a.dst)


if __name__ == '__main__':
    main()
