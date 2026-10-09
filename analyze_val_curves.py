"""analyze_val_curves.py -- where does validation loss actually flatten?

The target is 9 hours instead of 22 with quality unchanged. Every route through
raw throughput is closed by measurement: batch scaling is flat (304.9 tokens/s at
batch 4 and 8), the iGPU is unusable because CLBlast only works on cubic shapes
here -- its own sample shape fails with -1011 -- and the project's own OpenCL
kernels are unverified. So the only remaining lever is doing fewer steps at equal
quality, and the question that decides it is whether the loss has already
flattened before the budget runs out.

That question has been answerable all along from the training logs: several runs
record `val` per step. Read them rather than starting another run.

Reports, per run, the validation loss at increasing step counts and the improvement
per unit time, so the point of diminishing returns is visible.
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\work\bitsandbytes-CPU\i5build'


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('val') is not None:
                rows.append((int(r.get('step', 0)), float(r['val']),
                             float(r.get('step_s') or 0), int(r.get('T') or 0)))
    return rows


files = sorted(glob.glob(os.path.join(ROOT, '**', 'train.jsonl'), recursive=True))
best = []
for p in files:
    rows = load(p)
    if len(rows) < 20:
        continue
    best.append((len(rows), rows[-1][1], p, rows))

best.sort(key=lambda t: t[1])
print('%-42s %7s %9s %9s %8s' % ('run', '记录数', '末步', '末val', '最好val'))
print('-' * 80)
for n, lastval, p, rows in best[:8]:
    rel = p.replace(ROOT + '\\', '')
    print('%-42s %7d %9d %9.4f %9.4f'
          % (rel[:42], n, rows[-1][0], lastval, min(r[1] for r in rows)))
print()

# 对最长的几条做平台期分析
best.sort(key=lambda t: -t[0])
for n, lastval, p, rows in best[:4]:
    rel = p.replace(ROOT + '\\', '')
    print('=== %s（%d 个验证点，末步 %d，末 val %.4f）===' % (rel, n, rows[-1][0], lastval))
    tot_s = sum(r[2] for r in rows)
    marks = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    print('  %9s %10s %12s %14s %10s' % ('步', 'val', '相对最终', '累计小时', '每步增益'))
    prev = None
    for pct in marks:
        i = min(len(rows) - 1, max(0, int(len(rows) * pct / 100) - 1))
        step, v, ss, T = rows[i]
        cum = sum(r[2] for r in rows[:i + 1]) / 3600.0
        gain = '' if prev is None else '%+.5f' % (v - prev)
        print('  %9d %10.4f %11.2f%% %13.2f %10s'
              % (step, v, 100.0 * (v - lastval) / max(abs(lastval), 1e-9), cum, gain))
        prev = v
    print('  总训练时长约 %.1f 小时（按记录的 step_s 求和）' % (tot_s / 3600.0))
    print()
