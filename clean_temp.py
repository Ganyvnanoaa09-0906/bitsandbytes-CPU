"""Clear the Temp directory. It is almost entirely Visual Studio installer cache.

11.2 GB in %TEMP%, of which:

  c1ljwm5j               6.7 GB   VS installer download cache -- VC LLVM Clang 1.58 GB, three
                                  Win11 SDK versions totalling 1.57 GB, the performance toolkit
                                  at 320 MB, Copilot at 196 MB
  ~140 dirs at 125.4 MB  ~4.5 GB  VS installer extraction output (Assets, cs, de, es, fr, ...)
  trae-cn-user-x64      454 MB   the Trae IDE installer executable

All of it is payload for software that is already installed. None of it is in use: no process has
its image under %TEMP%.

Only items older than one day are removed, so anything a currently-running program created today
survives. Files that are locked are skipped rather than forced.

$WinREAgent (1.65 GB) and SoftwareDistribution\Download (0.96 GB) would also be worth clearing but
both need administrator rights and this session does not have them, so they are recorded rather
than attempted.

A manifest is written first, same as the checkpoint cleanup, so the cost of any mistake is visible.
Dry run by default.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

TEMP = os.environ['TEMP']
APPLY = '--apply' in sys.argv
CUTOFF = time.time() - 86400

items = []
for name in os.listdir(TEMP):
    full = os.path.join(TEMP, name)
    try:
        st = os.stat(full)
    except OSError:
        continue
    if st.st_mtime >= CUTOFF:
        continue
    if os.path.isdir(full):
        size = 0
        for dirpath, _, filenames in os.walk(full):
            for fn in filenames:
                try:
                    size += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
    else:
        size = st.st_size
    items.append((name, full, size, os.path.isdir(full)))

total = sum(s for _, _, s, _ in items)
print('  TEMP 中超过 1 天的项: %d 个 · %.2f GB' % (len(items), total / 1024 ** 3))
print()
for name, full, size, isdir in sorted(items, key=lambda x: -x[2])[:8]:
    print('  %-40s %9.1f MB  %s' % (name[:40], size / 1024 ** 2, '目录' if isdir else '文件'))
if len(items) > 8:
    print('  ... 另外 %d 项' % (len(items) - 8))
print()

mpath = r'D:\work\bitsandbytes-CPU\temp_cleanup_manifest.txt'
with io.open(mpath, 'w', encoding='utf-8') as f:
    f.write('# %%TEMP%% cleanup, %s\n' % ('APPLIED' if APPLY else 'DRY RUN'))
    for name, full, size, isdir in sorted(items, key=lambda x: -x[2]):
        f.write('%s\t%d\t%s\n' % (name, size, 'dir' if isdir else 'file'))
print('  清单: %s' % mpath)

if not APPLY:
    print()
    print('  ⇒ 干跑。加 --apply 执行。')
    sys.exit(0)

ok = 0
fail = 0
freed = 0
for name, full, size, isdir in items:
    try:
        if isdir:
            shutil.rmtree(full, ignore_errors=False)
        else:
            os.remove(full)
        ok += 1
        freed += size
    except Exception:
        fail += 1

print()
print('  ✓ 已删 %d 项 · 跳过 %d 项（被占用）· 释放 %.2f GB' % (ok, fail, freed / 1024 ** 3))
