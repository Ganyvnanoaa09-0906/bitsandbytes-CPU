# -*- coding: utf-8 -*-
"""analyze_curves.py — 平滑对比 hier / loop1 / big200m 的 gen loss 曲线

关键：训练 loss 的单点波动有 ±0.55，必须【移动平均】后才能比。
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
import io, json, os

D = r'D:\work\cloud_results\arcurves'
RUNS = ['hier', 'loop1', 'big200m']


def load(name):
    p = os.path.join(D, name + '.jsonl')
    if not os.path.exists(p):
        return []
    rows = []
    for line in io.open(p, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'gen' in r and r['gen'] is not None:
            rows.append((int(r['step']), float(r['gen'])))
    rows.sort()
    return rows


data = {n: load(n) for n in RUNS}

print('=' * 80)
print('AR 训练曲线平滑对比（gen loss）')
print('=' * 80)
print()
print('  %-10s %8s %12s %12s %12s %12s' % ('实验', '记录数', '最新步', '最新 gen', '原始最低', '平滑最低'))
print('  ' + '-' * 70)
for n, rows in data.items():
    if not rows:
        print('  %-10s  （无数据）' % n)
        continue
    vs = [v for _, v in rows]
    w = 5
    sm = [sum(vs[max(0, i - w + 1):i + 1]) / len(vs[max(0, i - w + 1):i + 1])
          for i in range(len(vs))]
    i_best = sm.index(min(sm))
    i_raw = vs.index(min(vs))
    print('  %-10s %8d %12d %12.4f %12.4f %12.4f'
          % (n, len(rows), rows[-1][0], vs[-1], vs[i_raw], sm[i_best]))
    print('  %-10s %8s %12s %12s %12s %12s'
          % ('', '', '', '', 'step %d' % rows[i_raw][0], 'step %d' % rows[i_best][0]))

# 同步数对比（用平滑值）
print()
print('  同步数对比（窗口 5 移动平均，排除单点噪声）:')
steps = [2000, 5000, 10000, 15000, 20000, 26641, 40000, 60000]
avail = [n for n, r in data.items() if r]
print('  %-9s' % 'step' + ''.join('%13s' % n for n in avail))
print('  ' + '-' * (9 + 13 * len(avail)))


def smooth_at(rows, S, win=5):
    """取 step<=S 的最后 win 个点的平均"""
    sub = [v for s, v in rows if s <= S]
    if not sub:
        return None
    sub = sub[-win:]
    return sum(sub) / len(sub)


for S in steps:
    line = '  %-9d' % S
    any_ = False
    for n in avail:
        v = smooth_at(data[n], S)
        if v is None:
            line += '%13s' % '-'
        else:
            line += '%13.4f' % v
            any_ = True
    if any_:
        print(line)

# 原始曲线采样（看波动幅度）
print()
print('  原始单点波动（hier，看噪声有多大）:')
rows = data['hier']
for s, v in rows:
    if s % 2000 == 0 or s in (500, 1000):
        print('    step %-7d gen %.4f' % (s, v))

print()
print('=' * 80)
print('判读要点:')
print('  1. 单点波动 = 噪声；必须看移动平均')
print('  2. 循环版（loop1）相对基准（hier）的差异若 < 噪声幅度 ⇒ 不能下结论')
print('  3. 成本也要算：loop1 的步时是 hier 的 ~1.4x（平均 L=1.4）')
print('DONE')
