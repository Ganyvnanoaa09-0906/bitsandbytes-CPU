"""Process memory watchdog: available RAM / Windows commit guard.

Use in heavy scripts before loading models and every N steps::

    from memwatch import check_or_exit
    check_or_exit('before-load')

Raises SystemExit(3) when physical available < min_available_gb or commit usage > limit_pct.
"""
from __future__ import annotations

import ctypes
import os

import psutil

MIN_AVAILABLE_GB = float(os.environ.get('MEM_MIN_AVAILABLE_GB', '2.5'))
COMMIT_LIMIT_PCT = float(os.environ.get('MEM_COMMIT_LIMIT_PCT', '85'))


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]


def commit_gb():
    """(used_commit_gb, total_commit_gb, pct) on Windows; (0,0,0) elsewhere."""
    try:
        st = _MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        total = st.ullTotalPageFile / 1e9
        used = (st.ullTotalPageFile - st.ullAvailPageFile) / 1e9
        return used, total, (100.0 * used / total if total else 0.0)
    except Exception:
        return 0.0, 0.0, 0.0


def stats():
    vm = psutil.virtual_memory()
    used, total, pct = commit_gb()
    return {'available_gb': vm.available / 1e9, 'percent': vm.percent,
            'commit_used_gb': used, 'commit_total_gb': total, 'commit_pct': pct}


def check_or_exit(tag='', min_available_gb=None, commit_limit_pct=None):
    s = stats()
    a = MIN_AVAILABLE_GB if min_available_gb is None else min_available_gb
    c = COMMIT_LIMIT_PCT if commit_limit_pct is None else commit_limit_pct
    bad = s['available_gb'] < a or (s['commit_pct'] and s['commit_pct'] > c)
    msg = ('[memwatch %s] avail=%.2fGB commit=%.1f/%.1fGB (%.1f%%)'
           % (tag, s['available_gb'], s['commit_used_gb'], s['commit_total_gb'], s['commit_pct']))
    print(msg, flush=True)
    if bad:
        print('MEMORY_GUARD_EXIT available<%.2fGB or commit>%.0f%%' % (a, c), flush=True)
        raise SystemExit(3)
    return s


if __name__ == '__main__':
    import json
    print(json.dumps(stats(), indent=1))
