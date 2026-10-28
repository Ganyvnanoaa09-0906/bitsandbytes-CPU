# -*- coding: utf-8 -*-
"""training_report.py - 从训练 run 目录生成 loss 曲线 + 速度报告（markdown，无 matplotlib 依赖）。

用法:
  python training_report.py --run <run_dir> --out report.md
  python training_report.py --run <run_dir>                 # 直接打印到终端
"""
from __future__ import annotations
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
import argparse, json, os, statistics, sys


def load_jsonl(p):
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def ascii_curve(xs, ys, width=64, height=16, title='loss', logy=False, floor=1e-3):
    """把 (x, y) 序列画成 ASCII 折线图。loss 这种跨几个数量级的用 logy=True。"""
    if not ys:
        return '(no data)'
    if logy:
        import math
        vals = [math.log10(max(v, floor)) for v in ys]
        fmt = lambda v: f'{10 ** v:9.5f}'
    else:
        vals = list(ys)
        fmt = lambda v: f'{v:9.4f}'
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    n = len(ys)
    idx = [int(c * (n - 1) / max(1, width - 1)) for c in range(width)]
    grid = [[' '] * width for _ in range(height)]
    for c, i in enumerate(idx):
        r = int((hi - vals[i]) / (hi - lo) * (height - 1))
        grid[r][c] = '*'
    lines = []
    for r in range(height):
        v = hi - (hi - lo) * r / (height - 1)
        lines.append(f'{fmt(v)} |' + ''.join(grid[r]))
    lines.append(' ' * 9 + ' +' + '-' * width)
    x0, x1 = xs[0] if xs else 0, xs[-1] if xs else 0
    lines.append(' ' * 10 + f'{title}: step {x0} -> {x1}  ({n} points)')
    return '\n'.join(lines)


def stage_table(rows):
    """按 T 分组统计。"""
    groups = {}
    for r in rows:
        groups.setdefault(r.get('T'), []).append(r)
    out = []
    for T in sorted(groups, key=lambda x: (x is None, x)):
        g = groups[T]
        sp = [x['step_s'] for x in g if 'step_s' in x]
        ls = [x['loss'] for x in g if 'loss' in x]
        if not sp:
            continue
        sp_sorted = sorted(sp)
        out.append(dict(
            T=T, n_logged=len(g),
            first_loss=round(ls[0], 4) if ls else None,
            last_loss=round(ls[-1], 4) if ls else None,
            min_loss=round(min(ls), 4) if ls else None,
            mean_step_s=round(sum(sp) / len(sp), 3),
            median_step_s=round(statistics.median(sp), 3),
            p10_step_s=round(sp_sorted[len(sp_sorted) // 10], 3),
            p90_step_s=round(sp_sorted[min(len(sp_sorted) - 1, 9 * len(sp_sorted) // 10)], 3),
            min_step_s=round(min(sp), 3), max_step_s=round(max(sp), 3),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='训练输出目录（含 train.jsonl / summary.json）')
    ap.add_argument('--out', default='', help='输出 markdown 路径（默认打印到终端）')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=16)
    a = ap.parse_args()

    rows = load_jsonl(os.path.join(a.run, 'train.jsonl'))
    summ = {}
    sp = os.path.join(a.run, 'summary.json')
    if os.path.exists(sp):
        summ = json.load(open(sp, encoding='utf-8'))

    L = []
    L.append(f'# 训练报告 — `{os.path.basename(a.run.rstrip(os.sep))}`\n')
    if summ.get('params'):
        L.append(f"- 参数量：**{summ['params']/1e6:.2f} M**")
    if summ.get('peak_rss_gb'):
        L.append(f"- 峰值内存：**{summ['peak_rss_gb']:.2f} GB**")
    L.append(f"- 记录点数：{len(rows)}")
    L.append('')

    L.append('## 各阶段（按序列长度 T 分组）\n')
    L.append('| T | 记录点 | 首 loss | 末 loss | 最低 loss | 平均 s/step | 中位 | p10 | p90 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for s in stage_table(rows):
        L.append('| {T} | {n_logged} | {first_loss} | {last_loss} | {min_loss} | '
                 '**{mean_step_s}** | {median_step_s} | {p10_step_s} | {p90_step_s} |'.format(**s))
    L.append('')

    for T in sorted({r.get('T') for r in rows if r.get('T') is not None}):
        sub = [r for r in rows if r.get('T') == T and 'loss' in r]
        if not sub:
            continue
        L.append(f'## Loss 曲线 — T={T}（对数纵轴）\n')
        L.append('```')
        L.append(ascii_curve([r['step'] for r in sub], [r['loss'] for r in sub],
                             width=a.width, height=a.height, title=f'T={T} loss', logy=True))
        L.append('```')
        L.append('')
        L.append(f'## 单步耗时 — T={T}（线性纵轴）\n')
        L.append('```')
        L.append(ascii_curve([r['step'] for r in sub], [r['step_s'] for r in sub],
                             width=a.width, height=a.height, title=f'T={T} s/step'))
        L.append('```')
        L.append('')

    text = '\n'.join(L)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'[report] -> {a.out}')
    else:
        print(text)


if __name__ == '__main__':
    main()
