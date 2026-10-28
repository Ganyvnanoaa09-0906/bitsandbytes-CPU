"""measure_guard.py -- refuse to benchmark a busy machine, and stamp every result.

The user found this one, and it is worth more than a note to self: three measurements
taken today -- memory bandwidth, FMA peak, and matmul shapes -- were all taken while the
Muon A/B was using 4.79 of 6 cores. Every one of them is suspect, and one of them (the
"FMA is already at 79%" figure) fed a conclusion. Memory of the rule did not prevent it;
a mechanism will.

Two functions, meant to be used together:

    guard.assert_idle()      raise before measuring if the machine is busy
    guard.stamp(label)       print the load alongside the numbers

assert_idle samples system-wide CPU several times and refuses if any sample exceeds a
threshold, naming the processes responsible so the cause is obvious rather than mysterious.
It also refuses if this process itself has already spun up threads, which catches the case
of a benchmark that starts its own load and then measures -- which cpu_fma_headroom.py did.

stamp() exists because a number without its conditions is not a measurement. Today's
failures were not wrong arithmetic so much as results whose context was lost by the time
they were used.

Usage:

    from measure_guard import assert_idle, stamp
    assert_idle()
    ... measure ...
    stamp('bandwidth')
"""
from __future__ import annotations
import os, sys
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

import statistics
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None

# The criterion is not total CPU. The harness that hosts the agent session holds most of
# a core continuously, so a global threshold refuses to measure on an idle machine. What
# matters is whether some *other* substantial job is running.
ALLOWLIST = ('deepseek harness', 'dsh', 'node', 'pwsh', 'powershell', 'conhost',
             'windowsterminal', 'explorer', 'code', 'taskmgr', 'systemsettings',
             'snippingtool', 'screenclip', 'searchhost', 'startmenuexperiencehost',
             # System Idle Process reports the *idle* percentage, so an idle machine
             # shows it at 400%; pid 0 must never count as load
             'system idle process', 'idle')
# taskmgr and friends sit at a few percent just by being open; they are not load
DEFAULT_THRESHOLD = 50.0     # percent of one core, for the busiest non-allowlisted job
DEFAULT_SAMPLES = 6
DEFAULT_INTERVAL = 0.25


def _top_processes(n=5, interval=0.4):
    """Busiest processes. cpu_percent is 0.0 on a process's first read, so every
    process is primed once and then sampled -- without this the list comes back empty
    and the guard says "busy" without saying what is busy."""
    if psutil is None:
        return []
    procs = list(psutil.process_iter(['pid', 'name']))
    for p in procs:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(interval)
    out = []
    for p in procs:
        try:
            c = p.cpu_percent(None)
            if c and c > 1.0:
                out.append((c, p.info['pid'], p.info['name']))
        except Exception:
            continue
    out.sort(reverse=True)
    return out[:n]


def load_samples(n=DEFAULT_SAMPLES, interval=DEFAULT_INTERVAL):
    """System-wide CPU percentages, sampled n times."""
    if psutil is None:
        return []
    psutil.cpu_percent(interval=None)      # prime the counters
    time.sleep(interval)
    return [psutil.cpu_percent(interval=interval) for _ in range(n)]


def _busiest_other():
    """The busiest process that is neither allowlisted nor this one."""
    import os
    me = os.getpid()
    for c, pid, name in _top_processes(10, interval=0.5):
        if pid == me or pid == 0:
            continue
        low = (name or '').lower()
        if any(a in low for a in ALLOWLIST):
            continue
        return c, pid, name
    return None


def assert_idle(threshold=DEFAULT_THRESHOLD, n=DEFAULT_SAMPLES, interval=DEFAULT_INTERVAL,
                allow=False):
    """Abort if a substantial job other than the harness is running.

    Reports the full process list either way: knowing what is running is the point, and
    "the machine is busy" without naming the culprit is what made today's pollution hard
    to notice after the fact.
    """
    if psutil is None:
        print('[guard] psutil 不可用，跳过空闲检查（结果请自行判断）')
        return None
    samples = load_samples(n, interval)
    total = statistics.median(samples)
    others = _top_processes(5)
    busy = _busiest_other()
    if busy is None or busy[0] <= threshold:
        print('[guard] 无其他大任务（整机 %.1f%%，最大他人进程 %s）'
              % (total, ('%s %.0f%%' % (busy[2], busy[0])) if busy else '无'))
        return total
    c, pid, name = busy
    msg = ['', '=' * 74,
           '[guard] 拒绝测量：有别的任务在跑 —— %s (pid %d) 占 %.1f%%（阈值 %.1f%%）'
           % (name, pid, c, threshold),
           '[guard] 当前进程列表：']
    for cc, pp, nn in others:
        msg.append('          %-24s pid %-8d %.1f%%' % (nn, pp, cc))
    msg.append('[guard] 今天有三组测量就是在 A/B 占 4.79 核时做的，全部作废。')
    msg.append('[guard] 等它结束，或 assert_idle(allow=True) 明确接受污染'
               '（仅当各臂条件相同时才正当）。')
    msg.append('=' * 74)
    text = '\n'.join(msg)
    if allow:
        print(text)
        print('[guard] allow=True ⇒ 继续。')
        return total
    raise SystemExit(text)


def stamp(label, note=''):
    """Print a measurement's conditions so the number carries its own context."""
    if psutil is None:
        print('[条件] %s：psutil 不可用，无法记录负载%s' % (label, note))
        return
    cpu = psutil.cpu_percent(interval=0.3)
    vm = psutil.virtual_memory()
    n_py = sum(1 for p in psutil.process_iter(['name'])
               if (p.info.get('name') or '').lower().startswith('python'))
    print('[条件] %-18s CPU %5.1f%%  内存 %.1f/%.1f GB  python 进程 %d  %s'
          % (label, cpu, (vm.total - vm.available) / 2**30, vm.total / 2**30, n_py, note))


if __name__ == '__main__':
    print('当前负载样本:', ['%.0f' % v for v in load_samples()])
    stamp('自检')
    assert_idle()
    print('机器空闲 ✓ 可以测量')
