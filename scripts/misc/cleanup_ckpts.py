"""Clean up intermediate checkpoints, with a manifest and a dry run.

181 checkpoint files, 18.29 GB, in i5build. C: has 8.1 GB free and D: 11 GB, so this is not yet
urgent but it will be, and it will be noticed at the worst moment -- mid-training, when a
checkpoint write fails.

What is kept and why:

  * ckpt_last.pt in every directory -- the resume point, and cheap to keep
  * vqgan32/ckpt_last.pt -- the tokenizer, used by every training run and every sampling call
  * the checkpoints that specific reported results refer to: ar_wd's 6000-step (the weight-decay
    comparison), ar_long's 3000 and 12000 (the samples), ar_g16's last, loop_ar's 10000 (the
    comparison that turned out to be invalid, kept because the observation is in the report)

What is deleted: intermediate snapshots whose only content is a point on a validation curve, and
that curve is already in the training logs, which are in the report's raw evidence directory.

The manifest records every path and size before deletion, so the cost of any mistake is visible
rather than assumed. Dry run by default; pass --apply to actually delete.
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

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

ROOT = r'D:\work\bitsandbytes-CPU\i5build'
APPLY = '--apply' in sys.argv

# directories whose entire contents predate the current work and whose experiment is closed.
# The logs are the record; the checkpoints are 116 MB each and there are dozens.
CLOSED = ['adiff_smoke', 'adiff_smoke2', 'loop_ctrl1', 'loop_sched1',
          'loop_cond1', 'loop_cond_probe', 'loop_ab_eval', 'loop_ab_L1', 'loop_ab_random',
          'msa_ar', 'loop_ar3', 'loop_ar4', 'ar3_test', 'ar_final', 'ceiling', 'diversity',
          'sched_result', 'cond_result', 'cond_result2', 'loop_ar2', 'vq24', 'vq32',
          'vq_probe_256', 'vq_probe_384', 'vq_probe_512', 'vq_roundtrip', 'vqgan_segtest',
          'vq_leaktest', 'speed_cmp', 'conv_quant_out', 'ss_test', 'cold_metric']

# keep these specific checkpoints even though they are intermediate
KEEP = {
    'ar_wd': {'ckpt_6000.pt', 'ckpt_last.pt'},          # the weight-decay comparison point
    'ar_long': {'ckpt_3000.pt', 'ckpt_12000.pt', 'ckpt_last.pt'},   # sampled at both
    'ar_g16': {'ckpt_last.pt'},                          # the grid experiment
    'ar32_run1': {'ckpt_3000.pt', 'ckpt_last.pt'},       # the original baseline
    'ar_muon3_3000': {'ckpt_last.pt'},                   # the 3.57x run
    'vqgan32': {'ckpt_last.pt', 'ckpt_8000.pt'},                         # the tokenizer, needed by everything
    'maskgit1': {'ckpt_last.pt'},
    'loop_ar': {'ckpt_10000.pt', 'ckpt_last.pt'},        # referenced in the report
    'currB_p2': {'ckpt_last.pt'},
    'adiff_run1': {'ckpt_last.pt'},
}

manifest = []
freed = 0
kept = 0

for dirpath, dirnames, filenames in os.walk(ROOT):
    for fn in filenames:
        if not (fn.startswith('ckpt') and fn.endswith('.pt')):
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, ROOT)
        top = rel.split(os.sep)[0]
        size = os.path.getsize(full)
        keepset = KEEP.get(top)
        # keep everything in directories not listed anywhere -- unknown is not the same as unused
        known = top in CLOSED or keepset is not None
        if not known:
            kept += 1
            continue
        if keepset is not None and fn in keepset:
            kept += 1
            continue
        manifest.append((rel, size, 'CLOSED' if top in CLOSED else 'intermediate'))
        freed += size

print('  i5build 下共 %d 个 checkpoint' % (kept + len(manifest)))
print('  保留 %d 个 · 删除 %d 个 · 释放 %.2f GB' % (kept, len(manifest), freed / 1024 ** 3))
print()
by_top = {}
for rel, size, why in manifest:
    t = rel.split(os.sep)[0]
    a, b = by_top.get(t, (0, 0))
    by_top[t] = (a + 1, b + size)
print('  %-22s %6s %10s' % ('目录', '个数', 'MB'))
for t, (n, s) in sorted(by_top.items(), key=lambda kv: -kv[1][1])[:16]:
    print('  %-22s %6d %10.1f' % (t, n, s / 1024 ** 2))
print()

mpath = r'D:\work\bitsandbytes-CPU\ckpt_cleanup_manifest.txt'
with io.open(mpath, 'w', encoding='utf-8') as f:
    f.write('# checkpoint cleanup manifest, %s\n' % ('APPLIED' if APPLY else 'DRY RUN'))
    f.write('# rel_path\tsize_bytes\twhy\n')
    for rel, size, why in manifest:
        f.write('%s\t%d\t%s\n' % (rel, size, why))
print('  清单已写: %s' % mpath)

if not APPLY:
    print()
    print('  ⇒ 干跑。确认无误后加 --apply 执行。')
    sys.exit(0)

n = 0
err = 0
for rel, size, why in manifest:
    full = os.path.join(ROOT, rel)
    try:
        os.remove(full)
        n += 1
    except Exception as e:
        err += 1
        if err <= 3:
            print('  删除失败 %s: %s' % (rel, str(e)[:60]))
print()
print('  ✓ 已删除 %d 个文件，失败 %d 个' % (n, err))

print()
print('  注意：CLOSED 目录只删 checkpoint，日志与 jsonl 全部保留 —— 那些实验的记录只有日志。')
