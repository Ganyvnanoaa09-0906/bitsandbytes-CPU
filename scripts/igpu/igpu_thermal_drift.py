"""Is the machine drifting? Measure the iGPU alone, idle, measure again.

The iGPU's solo figure today: 393.2, 392.7, 390.6, 386.5 GFLOPS in the morning, then 315.2 an hour
later. A 20% decline with no change in code or configuration. Within a single process the concurrent
measurement is stable to 2.1%, so the instability is the machine's, not the method's.

That matters beyond this experiment. If the solo rate drifts 20% over a session, then every iGPU figure
measured in the later part of the session is suspect, and so is any comparison between a figure taken
early and one taken late. The report has already recorded 7.62's cliff based on a single torch.mm size,
and 7.88 corrected it; this would be a third layer on the same quantity.

Temperature cannot be read: the BIOS exposes neither ACPI thermal zones nor RAPD, which 7.62 recorded
and today confirmed. So the test is behavioural rather than direct. Measure the solo rate, let the
machine sit idle for several minutes, measure again. If the rate recovers, the cause is thermal or a
sustained power budget that resets when load stops, and the drift is recoverable. If it does not
recover, something else changed and needs identifying.

Criterion, fixed in advance: recovery to within 5% of the morning's 393 GFLOPS means the drift is
thermal and recoverable, and that measurements must be taken from a rested machine. Recovery to
somewhere between means partial. No recovery means the morning's numbers and today's are not comparable
at all.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl
from scripts.lib.measure_guard import assert_idle, stamp

exec(open(r'D:\work\bitsandbytes-CPU\igpu_blocking_optimum.py', encoding='utf-8').read()
     .split('assert_idle()')[0])

MORNING = 393.2          # measured at 19:00 today, same kernel, same shapes

assert_idle()
stamp('机器热漂移:核显单独吞吐在不在恢复')
print()

idle = gpu_steady(reps=10)
now = GPU_FLOP / idle / 1e9
print('  现在: %.3f ms ⇒ 【%.1f GFLOPS】' % (idle * 1e3, now))
print('  今早基准: %.1f GFLOPS ⇒ 现在只有它的 %.0f%%' % (MORNING, now / MORNING * 100))
print()
print('  静置 6 分钟（不跑任何负载）……')
for m in range(6):
    time.sleep(60)
    sys.stdout.write('    第 %d 分钟\n' % (m + 1))
    sys.stdout.flush()
print()

idle2 = gpu_steady(reps=10)
after = GPU_FLOP / idle2 / 1e9
print('  静置后: %.3f ms ⇒ 【%.1f GFLOPS】' % (idle2 * 1e3, after))
print('  相对静置前: %.2f× · 相对今早: %.0f%%' % (after / now, after / MORNING * 100))
print()
print('  判据（事先固定）: 恢复到今早的 95%% 以内 ⇒ 热漂移,可恢复 ⇒ 测量必须从【歇过的机器】上取')
if after >= MORNING * 0.95:
    print('  ⇒ ★【可恢复的热漂移】✓ —— 恢复后 %.1f 对今早 %.1f' % (after, MORNING))
    print('  ⇒ ★【所以今天下午之后的核显数字,要与【当时的单独基准】一起读,不能跨时段比】')
elif after > now * 1.10:
    print('  ⇒ 【部分恢复】%.1f → %.1f,而仍未回到今早的 %.1f' % (now, after, MORNING))
else:
    print('  ⇒ ✗【没有恢复】⇒ 不是热,是别的状态变了,需要另行定位')
print()
print('  而这条的意义:本项目已经因为【跨时段比较】栽过两次（7.62 的悬崖、7.88 的两次跑分歧 2.9×）,')
print('  而这一次给出了一个可直接执行的规矩:【任何核显数字,必须同时报它的单独基准】。')
