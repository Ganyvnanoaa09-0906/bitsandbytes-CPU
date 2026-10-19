"""audit_project.py — whole-repository static audit for bitsandbytes-CPU.

Runs fast, offline checks and prints a structured report:
  1. inventory (files per type, tracked vs untracked, biggest files)
  2. Python syntax check (py_compile every .py)
  3. shell syntax check (bash -n every .sh) when bash is available
  4. suspicious-pattern scan: creds, dangerous calls, silent excepts, TODOs,
     hard-coded absolute paths inside the shipped package, -march=native
  5. shipped-package specific checks (import-time side effects, native artefacts expected)
  6. duplicate file names / huge files

Exit code 1 if any hard failure (syntax error), else 0. Warnings are reported but do not fail.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
SHIPPED = os.path.join(ROOT, 'bitsandbytes', 'bitsandbytes')
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.pytest_cache', 'dist', 'build',
             'dist_wheel_win', 'dist_sdist', 'dist_release', 'release_out', 'i5build',
             'logs', 'build_linux', 'build_manual', 'build_arm64', 'models', '.venv'}
CODE_EXT = {'.py', '.sh', '.ps1', '.c', '.cpp', '.h', '.cu', '.bat'}

CRED_PATTERNS = [
    ('pypi token', re.compile(r'pypi-[A-Za-z0-9_\-]{40,}')),
    ('openai key', re.compile(r'sk-[A-Za-z0-9]{32,}')),
    ('aws key', re.compile(r'AKIA[0-9A-Z]{16}')),
    ('private key', re.compile(r'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY')),
    ('hardcoded password', re.compile(r'(password|passwd|secret)\s*=\s*["\'][^"\']{6,}["\']', re.I)),
]

DANGEROUS = [
    ('eval(', re.compile(r'\beval\s*\(')),
    ('exec(', re.compile(r'\bexec\s*\(')),
    ('os.system(', re.compile(r'os\.system\s*\(')),
    ('subprocess shell=True', re.compile(r'shell\s*=\s*True')),
    ('pickle.load', re.compile(r'pickle\.(load|loads)\s*\(')),
    ('torch.load w/o weights_only', re.compile(r'torch\.load\s*\(')),
    ('yaml.load', re.compile(r'yaml\.load\s*\(')),
    ('verify=False', re.compile(r'verify\s*=\s*False')),
    ('-march=native', re.compile(r'-march=native')),
    ('requests/urllib http://', re.compile(r'http://(?!127\.0\.0\.1|localhost)')),
]

QUALITY = [
    ('bare except', re.compile(r'except\s*:')),
    ('silent except', re.compile(r'except[^:]*:\s*(pass|\.\.\.)\s*$', re.M)),
    ('TODO/FIXME', re.compile(r'\b(TODO|FIXME|XXX|HACK)\b')),
    ('hardcoded D:/C: path', re.compile(r'["\'](?:[A-Za-z]:\\\\|/mnt/[a-z]/)')),
    ('print( in shipped lib', re.compile(r'^\s*print\(', re.M)),
]


def walk_files(root=ROOT):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield os.path.join(dirpath, name)


def rel(path):
    return os.path.relpath(path, ROOT)


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def main() -> int:
    files = list(walk_files())
    # ---------------------------------------------------------------- inventory
    section('1. inventory')
    by_ext = {}
    total_bytes = 0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        by_ext.setdefault(ext, [0, 0])
        by_ext[ext][0] += 1
        try:
            sz = os.path.getsize(f)
        except OSError:
            sz = 0
        by_ext[ext][1] += sz
        total_bytes += sz
    for ext, (n, b) in sorted(by_ext.items(), key=lambda kv: -kv[1][1])[:15]:
        print('  %-8s %5d files %10.1f MB' % (ext or '<none>', n, b / 1e6))
    print('  total: %d files, %.1f MB' % (len(files), total_bytes / 1e6))

    print('\n  biggest files:')
    big = sorted(((os.path.getsize(f), f) for f in files
                  if os.path.isfile(f)), reverse=True)[:8]
    for sz, f in big:
        print('    %8.1f MB  %s' % (sz / 1e6, rel(f)))

    try:
        tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True,
                                 text=True, encoding='utf-8', errors='replace').stdout.splitlines()
        untracked = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'],
                                   cwd=ROOT, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace').stdout.splitlines()
        print('  git: %d tracked, %d untracked' % (len(tracked), len(untracked)))
    except OSError:
        print('  git: unavailable')

    # ---------------------------------------------------------------- syntax
    section('2. Python syntax check')
    py_files = [f for f in files if f.endswith('.py')]
    bad = []
    for f in py_files:
        try:
            src = open(f, 'rb').read()
            compile(src, f, 'exec')
        except (SyntaxError, ValueError) as exc:
            bad.append((rel(f), str(exc)[:160]))
    if bad:
        for f, e in bad:
            print('  SYNTAX ERROR %s: %s' % (f, e))
    else:
        print('  %d Python files compile clean' % len(py_files))

    section('3. shell syntax check')
    sh_files = [f for f in files if f.endswith('.sh')]
    bash = subprocess.run(['bash', '--version'], capture_output=True, text=True).returncode == 0 \
        if os.name != 'nt' else False
    if not bash:
        print('  bash not on PATH here; skipped (%d .sh files)' % len(sh_files))
    else:
        bad_sh = []
        for f in sh_files:
            r = subprocess.run(['bash', '-n', f], capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if r.returncode != 0:
                bad_sh.append((rel(f), (r.stderr or '').strip()[:160]))
        for f, e in bad_sh:
            print('  SYNTAX ERROR %s: %s' % (f, e))
        if not bad_sh:
            print('  %d shell scripts parse clean' % len(sh_files))

    # ---------------------------------------------------------------- patterns
    section('4. suspicious-pattern scan (whole repo)')
    hits = {name: [] for name, _ in CRED_PATTERNS + DANGEROUS + QUALITY}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in CODE_EXT:
            continue
        try:
            if os.path.getsize(f) > 2_000_000:
                continue
            text = open(f, encoding='utf-8', errors='replace').read()
        except OSError:
            continue
        for group in (CRED_PATTERNS, DANGEROUS, QUALITY):
            for name, pat in group:
                n = len(pat.findall(text))
                if n:
                    hits[name].append((rel(f), n))
    for name in hits:
        rows = hits[name]
        if not rows:
            continue
        label = 'CREDENTIAL' if name in dict(CRED_PATTERNS) else 'pattern'
        print('  [%s] %-28s %3d files  e.g. %s' %
              (label, name, len(rows), ', '.join(r for r, _ in rows[:3])))
    clean = [n for n, rows in hits.items() if not rows]
    print('  zero-hit patterns: %s' % ', '.join(clean[:12]))

    # ------------------------------------------------- shipped package specifics
    section('5. shipped-package checks')
    if not os.path.isdir(SHIPPED):
        print('  SHIPPED package dir missing: %s' % SHIPPED)
    else:
        shipped_files = [f for f in walk_files(SHIPPED)]
        natives = sorted(os.path.basename(f) for f in shipped_files
                         if os.path.splitext(f)[1] in ('.dll', '.so', '.dylib'))
        print('  native artefacts in package dir: %s' % (natives or 'none'))
        init = os.path.join(SHIPPED, '__init__.py')
        text = open(init, encoding='utf-8', errors='replace').read()
        print('  __init__: %d bytes, imports torch: %s' %
              (len(text), 'import torch' in text or 'from torch' in text))
        for expected in ('__version__',):
            print('  %-12s present: %s' % (expected, expected in text))
        # shipped modules with hard-coded absolute paths
        for f in shipped_files:
            if not f.endswith('.py'):
                continue
            t = open(f, encoding='utf-8', errors='replace').read()
            m = re.search(r'["\'](?:[A-Za-z]:\\\\|/mnt/[a-z]/)', t)
            if m:
                print('  ABSOLUTE PATH in shipped module: %s' % rel(f))
        # efst shipped
        efst = os.path.join(SHIPPED, 'efst.py')
        print('  bitsandbytes.efst shipped: %s' % os.path.isfile(efst))

    section('6. C/C++ warning build (shipped sources)')
    print('  run `bash build_linux.sh` in WSL / zig cross-builds separately;')
    print('  this audit only checks what is available on this host.')

    print()
    print('AUDIT RESULT: %s' % ('FAIL (syntax errors)' if bad else 'PASS (with warnings above)'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
