# -*- coding: utf-8 -*-
"""resume_list.py - 为 fast_clean.py 生成断点续跑清单。

只把「还没有产出结果」的源图写进清单。fast_clean 的输出保留相对目录结构
(<out>/images/<rel>.jpg)，所以直接按 rel 反查是否已完成。

用法:
  python resume_list.py --data D:\\a --out-images D:\\out\\images --out resume.jsonl
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

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='源图根目录')
    ap.add_argument('--out-images', required=True, help='<clean_out>/images')
    ap.add_argument('--out', required=True, help='写出的 jsonl')
    a = ap.parse_args()

    todo, done = [], 0
    for dp, _, fns in os.walk(a.data):
        for fn in fns:
            if not fn.lower().endswith(EXTS):
                continue
            p = os.path.join(dp, fn)
            rel = os.path.splitext(os.path.relpath(p, a.data))[0] + '.jpg'
            if os.path.exists(os.path.join(a.out_images, rel)):
                done += 1
            else:
                todo.append(p)

    todo.sort()
    with open(a.out, 'w', encoding='utf-8') as f:
        for p in todo:
            f.write(json.dumps({'src': p}, ensure_ascii=False) + '\n')

    print('已完成 %d, 待处理 %d -> %s' % (done, len(todo), a.out))


if __name__ == '__main__':
    main()
