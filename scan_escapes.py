r"""scan_escapes.py -- find every module-level docstring/string with an illegal
backslash escape in the repo, because Python 3.14 turns those into hard errors.

Found originally on Android/ARM64: disk_balancer.py line 48 emitted
    SyntaxWarning: "\c" is an invalid escape sequence.
The local Python 3.11 only warns, so it went unnoticed for a long time; 3.14
(which Termux has) states it will stop working, and then the import fails
outright on every platform.

(This file is raw itself: the line above names the sequence `\c`, and an
unescaped backslash-c in a non-raw docstring is exactly the defect it looks for.)

This scans by compiling each file with SyntaxWarning promoted to an error, which
catches the real thing rather than pattern-matching backslashes by eye.
"""
import os
import pathlib
import py_compile
import sys
import tempfile
import warnings

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SKIP_DIRS = {"__pycache__", ".git", ".autopilot", "node_modules"}

files = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if fn.endswith(".py"):
            files.append(os.path.join(dirpath, fn))

print(f"scanning {len(files)} .py file(s) under {os.path.abspath(ROOT)}")

tmpdir = tempfile.mkdtemp(prefix="escscan_")
bad_compile = []
bad_escape = []

for p in files:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            py_compile.compile(p, cfile=os.path.join(tmpdir, "out.pyc"), doraise=True)
        except Exception as e:  # noqa: BLE001
            bad_compile.append((p, type(e).__name__, str(e).splitlines()[-1][:100]))
        for w in rec:
            if "invalid escape" in str(w.message):
                bad_escape.append((p, w.lineno, str(w.message)[:110]))

print()
print(f"compile failures      : {len(bad_compile)}")
for p, kind, msg in bad_compile[:10]:
    print(f"  {kind}: {p}")
    print(f"      {msg}")

print()
print(f"invalid-escape warnings: {len(bad_escape)}")
for p, ln, msg in bad_escape[:20]:
    print(f"  {p}:{ln}")
    print(f"      {msg}")

sys.exit(1 if (bad_compile or bad_escape) else 0)
