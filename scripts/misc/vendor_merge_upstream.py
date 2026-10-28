"""Vendor-merge upstream bitsandbytes main into the fork's vendored tree (bitsandbytes/).

Our fork's vendored tree is based on upstream commit 2b6cfb79 ("Bump dev version", 2026-08-13,
__version__ = 0.50.2.dev0). Upstream main is now 83364904 (2026-09-03), 11 commits ahead, including the
0.50.2 release. Three of the changed paths are dev/CI directories the fork deliberately dropped
(.github); the other eight exist in the fork and are all files the fork itself has modified, so every one
of them needs a real 3-way merge:

    ours   = bitsandbytes/<path> at HEAD
    base   = 2b6cfb79:<path>
    theirs = upstream/main:<path>

Uses `git merge-file` (same algorithm as git merge) per file, writes the result into the worktree with
conflict markers, and reports which files conflicted. Nothing is committed here.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
BASE = '2b6cfb79'
THEIRS = 'upstream/main'
PATHS = [
    'CMakeLists.txt',
    'README.md',
    'bitsandbytes/__init__.py',
    'bitsandbytes/backends/triton/kernels_4bit.py',
    'csrc/common.cuh',
    'csrc/gemm_4bit.cu',
    'docs/source/installation.mdx',
    'setup.py',
]
SKIP = ['.github/scripts/build-rocm.sh', '.github/workflows/python-package.yml',
        '.github/workflows/test-runner.yml']


def git(*args, check=False):
    r = subprocess.run(['git', '-C', REPO, *args], capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if check and r.returncode != 0:
        raise SystemExit('git %s failed: %s' % (' '.join(args), r.stderr))
    return r.stdout


def show(rev_path):
    return git('show', rev_path)


print('=== vendor merge: %s .. %s ===' % (BASE, THEIRS))
print('skipping dev/CI paths absent from the fork: %s' % ', '.join(SKIP))
print()

tmpdir = os.path.join(REPO, '_vendor_merge_tmp')
os.makedirs(tmpdir, exist_ok=True)

report = []
for path in PATHS:
    ours_path = os.path.join(REPO, 'bitsandbytes', path)
    ours = open(ours_path, encoding='utf-8', errors='replace').read()
    base = show('%s:%s' % (BASE, path))
    theirs = show('%s:%s' % (THEIRS, path))

    if ours == base:
        status = 'ours==base -> take theirs'
        merged = theirs
        conflicts = 0
    elif theirs == base:
        status = 'theirs==base -> keep ours'
        merged = ours
        conflicts = 0
    else:
        f_ours = os.path.join(tmpdir, 'ours.txt')
        f_base = os.path.join(tmpdir, 'base.txt')
        f_theirs = os.path.join(tmpdir, 'theirs.txt')
        open(f_ours, 'w', encoding='utf-8').write(ours)
        open(f_base, 'w', encoding='utf-8').write(base)
        open(f_theirs, 'w', encoding='utf-8').write(theirs)
        r = subprocess.run(['git', '-C', REPO, 'merge-file', '-p',
                            '-L', 'fork', '-L', 'upstream-base', '-L', 'upstream-main',
                            f_ours, f_base, f_theirs],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        merged = r.stdout
        conflicts = merged.count('<<<<<<<')
        status = '3-way merge: %d conflict(s)' % conflicts

    if conflicts == 0:
        open(ours_path, 'w', encoding='utf-8', newline='').write(merged)
    else:
        conflict_path = os.path.join(tmpdir, path.replace('/', '__') + '.conflict')
        open(conflict_path, 'w', encoding='utf-8', newline='').write(merged)
        status += ' -> written to %s' % os.path.basename(conflict_path)

    report.append((path, status, len(ours), len(theirs), len(merged)))

print('  %-46s %s' % ('path', 'result'))
for path, status, lo, lt, lm in report:
    print('  %-46s %s' % (path, status))
print()
print('unconflicted files were written to the worktree; conflicted ones were NOT (see _vendor_merge_tmp).')
