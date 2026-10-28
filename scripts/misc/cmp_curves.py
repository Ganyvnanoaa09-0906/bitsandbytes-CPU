# -*- coding: utf-8 -*-
"""cmp_curves.py — 在【匹配步数】上比较 AR 实验，判断真正的瓶颈

单步 loss 噪声极大（实测同一实验相邻步 4.39 / 5.33），
所以每个比较点都取 ±window 步的均值，且只比【双方都覆盖】的步数区间。
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
import json, os, sys, math

D = r'D:\work\cloud_results\curves'
RUNS = [
    ('ar_flat256.jsonl',      '对照: 全量 29113 行', 256),
    ('ar_halfdata.jsonl',     '★半数据 14556 行', 256),
    ('ar_nbr256.jsonl',       '邻居融合', 256),
    ('ar_big200m_flat.jsonl', '200M', 256),
]
WINDOW = 120      # 窗口取小一点：新臂只有几百步


def load(p):
    if not os.path.exists(p):
        return None
    d = {}
    for l in open(p, encoding='utf-8'):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        d[r['step']] = r
    return d


data = []
print('=' * 104)
print('AR 实验曲线对比（单位 nats/token；均匀分布基线 = ln(1024) = %.4f）' % math.log(1024))
print('=' * 104)
for fn, name, T in RUNS:
    d = load(os.path.join(D, fn))
    if d is None:
        print('%-28s 缺失' % name); continue
    data.append((name, d, max(d), T))
    print('%-28s step=%-7d 点数=%d' % (name, max(d), len(d)))

if len(data) < 2:
    print('样本不足'); sys.exit(0)

hi = min(m for _n, _d, m, _T in data)
# 窗口自适应：至少在每臂平均有 ~3 个点，避免新臂只有一个点时被"均值"糊掉
avg_gap = 20.0
WIN = max(20, min(250, int(hi / 20) * 20 // 2 * 2) or 20)
print()
print('共同覆盖到 step %d；窗口 ±%d 步（自适应）' % (hi, WIN))
hdr = '%-8s' % 'step'
for n, _d, _m, _T in data:
    hdr += ' | %-24s' % n[:24]
print(hdr)
print('-' * len(hdr))
# 只在【所有臂都有覆盖】的步数上取点
pts = sorted({int(hi * f) for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)})
for s in pts:
    row = '%-8d' % s
    for n, d, m, T in data:
        win = [v['loss'] for k, v in d.items() if abs(k - s) <= WIN]
        row += ' | %-24s' % (('%.4f(n=%d)' % (sum(win) / len(win), len(win))) if win else '-')
    print(row)


print()
print('每个实验自己的最后窗口（loss / val / 相对均匀基线的解释率）')
for n, d, m, T in data:
    win = [v['loss'] for k, v in d.items() if k >= m - WIN]
    v = [v['val'] for k, v in d.items() if k >= m - WIN and v.get('val') is not None]
    ml = sum(win) / len(win)
    mv = sum(v) / len(v) if v else float('nan')
    U = math.log(1024)
    print('  %-28s step=%-7d loss=%.4f  val=%.4f  解释率 1-loss/U=%.1f%%'
          % (n, m, ml, mv, 100 * (1 - ml / U)))

print()
print('=' * 104)
print('判读：')
print('  ① 若 grn256(全双向) 与 flat256(causal) 在同步数下【接近】⇒ 上下文访问不是瓶颈，')
print('     瓶颈在容量/优化 ⇒ 别指望换 mask，该加容量或改训练')
print('  ② 若 200M 与 30M 在同步数下【接近】⇒ 参数量也不是瓶颈 ⇒ 瓶颈在数据或目标')
print('  ③ 已知参照：16x16 token 的 2D 因果下界 H(x|左+上) = 3.384 bits = 2.345 nats')
print('     8x8  token 的 2D 因果下界 H(x|左+上) = 1.406 bits = 0.975 nats')
print('     （而 8x8 的 AR 实测 loss = 1.0024，已打在下界上 ⇒ 那条路线饱和了）')
print('DONE')
