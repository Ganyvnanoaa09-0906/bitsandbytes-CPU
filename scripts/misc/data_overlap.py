# -*- coding: utf-8 -*-
"""data_overlap.py — sujvji 原始图有多少已经进过清洗管线？（决定还有多少数据可用）

方法：原始文件名是内容哈希（如 0006885ad89f1c06...jpg）。若清洗时保留了 stem，
就能直接比对；若重命名了，就看数量关系 + 抽样的感知哈希近似。

先做最便宜的一步：数各集合的 stem 交集。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys
from collections import Counter

AP = r'D:\work\bitsandbytes-CPU\.autopilot'
SUJ = r'D:\work\sujvji'
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def stems(root):
    s = set()
    for dp, _dn, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(EXTS):
                s.add(os.path.splitext(f)[0].lower())
    return s


def names_sample(root, k=5):
    out = []
    for dp, _dn, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(EXTS):
                out.append(f)
                if len(out) >= k:
                    return out
    return out


print('=' * 88)
print('原始数据 vs 已清洗数据 的重叠检查')
print('=' * 88)

clean = {}
for d in ('clean_nf', 'clean_dl', 'clean_char2_1024', 'clean_webp', 'clean_za', 'clean_tu'):
    p = os.path.join(AP, d, 'images')
    if os.path.isdir(p):
        clean[d] = stems(p)
allclean = set().union(*clean.values()) if clean else set()
print()
print('已清洗各集合的 stem 数：')
for d, s in clean.items():
    print('  %-22s %6d' % (d, len(s)))
print('  %-22s %6d  (并集)' % ('合计', len(allclean)))
print('  文件名样例（clean_nf）:', names_sample(os.path.join(AP, 'clean_nf', 'images')))

raws = ['nsfw_dataset', 'dl_sets', 'civitai_nsfw', 'real_train_300',
        'flat_real_300', 'real_train_200', '图', '杂图']
print()
print('  %-22s %8s %8s %10s  %s' % ('原始目录', '张数', 'stem数', '与clean重叠', '样例名'))
print('  ' + '-' * 80)
grand_overlap = 0
grand_raw = 0
for r in raws:
    p = os.path.join(SUJ, r)
    if not os.path.isdir(p):
        continue
    s = stems(p)
    ov = len(s & allclean)
    grand_overlap += ov
    grand_raw += len(s)
    print('  %-22s %8d %8d %10d  %s' % (r, len(s), len(s), ov, names_sample(p, 1)))
print('  ' + '-' * 80)
print('  %-22s %8d %8d %10d  ⇒ 重叠率 %.1f%%'
      % ('合计', grand_raw, grand_raw, grand_overlap,
         100 * grand_overlap / max(grand_raw, 1)))
print()
print('判读：')
print('  * 重叠率很低 ⇒ sujvji 里大部分是【没进过管线】的原始图 ⇒ 数据是免费杠杆')
print('  * 重叠率很高 ⇒ 已经用过了，只是清洗后有损')
print('  * 注意：若清洗时【重命名】了，重叠率会假性为 0，需结合样例名判断')
print('DONE')
