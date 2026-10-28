# -*- coding: utf-8 -*-
"""data_overlap2.py — 修正版重叠检查：clean_* 的文件名可能是原始哈希的【前缀】

clean_nf 的样例名是 `img_00012d3a54cbb3cb.jpg`（img_ + 16 hex），
而 nsfw_dataset 是 64 hex 的完整哈希。若前者是后者的前 16 位，
上一版的"0% 重叠"就是纯命名假象。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, re
from collections import Counter

AP = r'D:\work\bitsandbytes-CPU\.autopilot'
SUJ = r'D:\work\sujvji'
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def collect(root):
    out = []
    for dp, _dn, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(EXTS):
                out.append(os.path.splitext(f)[0].lower())
    return out


def prefixes(stems, n):
    """返回每个 stem 的前 n 个 hex 字符（只取 hex 的）"""
    s = set()
    for x in stems:
        h = re.sub(r'^img_', '', x)
        h = re.sub(r'[^0-9a-f]', '', h)
        if len(h) >= n:
            s.add(h[:n])
    return s


print('=' * 92)
print('修正版重叠检查：clean_* 的 stem 是否为原始哈希前缀')
print('=' * 92)

clean = {}
for d in ('clean_nf', 'clean_dl', 'clean_char2_1024', 'clean_za', 'clean_tu', 'clean_webp'):
    p = os.path.join(AP, d, 'images')
    if os.path.isdir(p):
        clean[d] = collect(p)

# 先看命名模式
for d in ('clean_nf', 'clean_dl', 'clean_char2_1024'):
    if d in clean:
        s = clean[d][:3]
        print('  %-22s 样例 %s' % (d, s))

for npre in (8, 12, 16):
    cl = set()
    for d, s in clean.items():
        cl |= prefixes(s, npre)
    print()
    print('  --- 用 %d 位 hex 前缀比较（clean 侧共 %d 个前缀）---' % (npre, len(cl)))
    tot_ov = tot_raw = 0
    for r in ('nsfw_dataset', 'dl_sets', 'civitai_nsfw', 'real_train_300',
              'flat_real_300', 'real_train_200', '图', '杂图'):
        p = os.path.join(SUJ, r)
        if not os.path.isdir(p):
            continue
        rs = collect(p)
        rp = prefixes(rs, npre)
        ov = len(rp & cl)
        tot_ov += ov; tot_raw += len(rp)
        print('    %-20s %7d 张  重叠 %7d  (%.1f%%)'
              % (r, len(rs), ov, 100 * ov / max(len(rp), 1)))
    print('    %-20s %7d      %7d  (%.1f%%)'
          % ('合计', tot_raw, tot_ov, 100 * tot_ov / max(tot_raw, 1)))

print()
print('=' * 92)
print('判读：')
print('  * 若 16 位前缀下 nsfw_dataset 高度重叠 ⇒ 它已经用过了，上一版的 0% 是命名假象')
print('  * 若仍低 ⇒ sujvji 里确实有大量没进过管线的原始图')
print('DONE')
