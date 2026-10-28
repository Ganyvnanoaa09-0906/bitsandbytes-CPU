# -*- coding: utf-8 -*-
"""extract_pdfs.py — 把 D:\\work\\pdf 里的论文抽成纯文本，并按"与本项目病根的相关度"排序。

不盲读 18 篇 425 MB：我们的病根已经被实测钉死为
  **自由采样偏离数据流形（内容丢失 + 对比度/高频塌缩），teacher-forced 是好的**
所以真正相关的是【治采样漂移/误差累积】那一类：
  · GRN 2604.13030     输入腐化 + 迭代精化（我们已在做同族）
  · Infinity 2412.04431 bitwise 自校正，论文自己说是解 VAR 的误差累积
  · FARMER 2510.23588  AR over pixels，统一 flow+AR
  · Seedream 4.0       整体配方
  · DreamLite 2603.28713 端侧小模型
其余（LCM/ADP/ADM/APT/Flow-GRPO）是**扩散加速**，速度已解决 ⇒ 低优先。
"""
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
import io, os, sys, re

from pypdf import PdfReader

D = r'D:\work\pdf'
OUT = r'D:\work\pdf\_txt'
PRIORITY = [
    ('2604.13030', 'GRN 输入腐化+精化'),
    ('2412.04431', 'Infinity bitwise 自校正(治误差累积)'),
    ('2510.23588', 'FARMER AR over pixels'),
    ('2509.20427', 'Seedream 4.0'),
    ('2603.28713', 'DreamLite 端侧'),
    ('2503.07703', 'Seedream 2.0'),
    ('2504.11346', 'Seedream 3.0'),
    ('2506.09113', 'Seedance 1.0'),
]
os.makedirs(OUT, exist_ok=True)


def extract(path, maxpages=None):
    r = PdfReader(path)
    n = len(r.pages)
    txt = []
    for i, p in enumerate(r.pages):
        if maxpages and i >= maxpages:
            break
        try:
            txt.append(p.extract_text() or '')
        except Exception as e:
            txt.append('[page %d extract failed: %s]' % (i, str(e)[:60]))
    return n, '\n'.join(txt)


def main():
    files = sorted(os.listdir(D))
    files = [f for f in files if f.lower().endswith('.pdf')]
    print('PDF: %d 个' % len(files))
    done = {}
    for f in files:
        key = f.split('v')[0].replace('.pdf', '')
        outp = os.path.join(OUT, key + '.txt')
        if os.path.exists(outp) and os.path.getsize(outp) > 2000:
            done[key] = os.path.getsize(outp)
            continue
        try:
            n, t = extract(os.path.join(D, f))
            io.open(outp, 'w', encoding='utf-8').write(t)
            print('  %-22s %3d 页 -> %7d 字符' % (f, n, len(t)), flush=True)
            done[key] = len(t)
        except Exception as e:
            print('  %-22s FAILED: %s' % (f, str(e)[:70]), flush=True)
    print()
    print('=== 按相关度排序 ===')
    for k, why in PRIORITY:
        if k in done:
            print('  %s  %8d 字符   %s' % (k, done[k], why))
    print()
    print('=== 其余（低优先：扩散加速族）===')
    for k in done:
        if k not in [p[0] for p in PRIORITY]:
            print('  %s  %8d 字符' % (k, done[k]))


if __name__ == '__main__':
    main()
