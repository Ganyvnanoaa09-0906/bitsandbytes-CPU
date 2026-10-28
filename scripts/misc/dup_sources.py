# -*- coding: utf-8 -*-
"""dup_sources.py - 把 final_dups.jsonl 还原成【源目录】的可读路径，供人工判断来源。

clean_webp/images/<集>/<名>.jpg  ->  D:\work\sujvji\webp_sets\<集>\<名>.<ext>
clean_dl/images/<集>/<名>.jpg    ->  D:\work\sujvji\dl_sets\<集>\<名>.<ext>
clean_char2_1024/images/img_x.jpg->  D:\work\sujvji\<原 char2 源>
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
import collections, json, os

BASE = r'D:\work\bitsandbytes-CPU\.autopilot'
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp')

SRC = {
    'webp':      r'D:\work\sujvji\webp_sets',
    'dl':        r'D:\work\sujvji\dl_sets',
    'photobook': r'D:\work\sujvji\photobook',
    'char2':     r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\images',
    'tu':        r'D:\work\sujvji\图',
    'za':        r'D:\work\sujvji\杂图',
}

FRAG = [('char2', r'clean_char2_1024\images'), ('dl', r'clean_dl\images'),
        ('webp', r'clean_webp\images'), ('photobook', r'sujvji\photobook'),
        ('tu', r'clean_tu\images'), ('za', r'clean_za\images')]


def pool_of(p):
    for n, f in FRAG:
        if f in p:
            return n
    return '?'


def build_index(root):
    """(集目录名, 不含扩展名的文件名) -> 完整路径"""
    idx = {}
    if not os.path.isdir(root):
        return idx
    for dp, _, fns in os.walk(root):
        setn = os.path.basename(dp)
        for f in fns:
            if f.lower().endswith(EXTS):
                idx[(setn, os.path.splitext(f)[0])] = os.path.join(dp, f)
    return idx


def to_source(clean_path, idx_cache):
    """把 clean_* 的路径映射回源目录路径。"""
    pool = pool_of(clean_path)
    if pool == 'char2':
        return clean_path          # char2 已清洗，源就是它自己
    root = SRC.get(pool)
    if root is None:
        return clean_path
    if pool not in idx_cache:
        idx_cache[pool] = build_index(root)
    setn = os.path.basename(os.path.dirname(clean_path))
    stem = os.path.splitext(os.path.basename(clean_path))[0]
    return idx_cache[pool].get((setn, stem), '(源文件未找到: %s / %s)' % (setn, stem))


def main():
    rows = [json.loads(l) for l in open(os.path.join(BASE, 'final_dups.jsonl'), encoding='utf-8')
            if l.strip()]
    cache = {}
    pair_lines = []
    for r in rows:
        kp = to_source(r['keep'], cache)
        for d in r['dup']:
            pair_lines.append((pool_of(r['keep']), pool_of(d), kp, to_source(d, cache)))

    out = os.path.join(BASE, 'DUP_SOURCES.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('重复项源路径对照（共 %d 对）\n' % len(pair_lines))
        f.write('格式: [保留池] 保留的源路径\n')
        f.write('      [移除池] 被移除的源路径\n\n')
        for a, b, kp, dp in pair_lines:
            f.write('[%s] %s\n[%s] %s\n\n' % (a, kp, b, dp))
    print('完整清单 -> %s   (%d 对)' % (out, len(pair_lines)))

    print()
    print('==================== 集级对照：同一批内容在磁盘上的三个位置 ====================')
    for sid in ('1021933', '616288', '278837'):
        print()
        print('集 %s:' % sid)
        for label, d in (('webp_sets', os.path.join(SRC['webp'], sid)),
                         ('dl_sets  ', os.path.join(SRC['dl'], sid))):
            z = os.path.join(r'D:\work\sujvji\Download', sid + '.zip')
            n = len([x for x in os.listdir(d)]) if os.path.isdir(d) else -1
            print('  %s  %s   (%d 个文件)' % (label, d, n))
        print('  zip      %s   %s' % (os.path.join(r'D:\work\sujvji\Download', sid + '.zip'),
                                      '存在' if os.path.exists(os.path.join(r'D:\work\sujvji\Download', sid + '.zip')) else '不存在'))

    print()
    print('==================== photobook 与根目录 zip 的对照 ====================')
    for s in ('Coser – 第 1 页 _ Gallery Epic', '[半半子] [40P+12S - 186MB] 尼尔 - 寄叶2号B型 _ Gallery Epic'):
        print()
        print('  解压后  D:\\work\\sujvji\\photobook\\%s\\' % s)
        print('  池里    D:\\work\\sujvji\\dl_sets\\%s\\' % s)
        z = r'D:\work\sujvji\%s.zip' % s
        print('  压缩包  %s   %s' % (z, '存在' if os.path.exists(z) else '不存在'))

    print()
    print('==================== 前 12 对源路径 ====================')
    for a, b, kp, dp in pair_lines[:12]:
        print('  KEEP [%s] %s' % (a, kp))
        print('  DUP  [%s] %s' % (b, dp))
        print()


if __name__ == '__main__':
    main()
