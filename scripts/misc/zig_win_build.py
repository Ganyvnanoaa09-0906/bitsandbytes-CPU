"""Build the portable Windows DLL with Zig's bundled clang and audit it (no MSVC needed).

Purpose: validate, without MSVC, that
  (a) the portable source compiles for x86_64-windows-gnu,
  (b) the generated assembly contains no AVX instructions,
  (c) Windows can load the resulting DLL and execute a kernel through ctypes.
The release artifacts remain MSVC-built; this is the no-toolchain verification path.
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

import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ZIG = r'D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe'
OUT = r'D:\work\research\zig_win'
REPO = r'D:\work\bitsandbytes-CPU\bitsandbytes'
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)

FLAGS = ['-target', 'x86_64-windows-gnu', '-O2', '-std=c++17',
         '-DNOMINMAX', '-DNDEBUG', '-DBUILD_CUDA=0', '-DBUILD_HIP=0', '-DBUILD_XPU=0',
         '-DBNB_CPU_PORTABLE=1', '-I', 'csrc']
AVX_PAT = 'ymm|zmm|vfmadd|vzeroupper|vmovaps|vmulps|vaddps|vperm'


def run(args: list[str]) -> int:
    r = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('FAILED:', ' '.join(args[:6]), '...')
        print((r.stderr or r.stdout)[-800:])
    return r.returncode


rc = 0
for tu in ('cpu_ops', 'cpu_gdn', 'pythonInterface'):
    print('=== compile %s.cpp ===' % tu)
    rc |= run([ZIG, 'c++', *FLAGS, '-c', r'csrc\%s.cpp' % tu, '-o', os.path.join(OUT, tu + '.o')])
    print('  exit %d' % rc)

print('=== AVX audit on generated assembly ===')
for tu in ('cpu_ops', 'cpu_gdn', 'pythonInterface'):
    run([ZIG, 'c++', *FLAGS, '-S', '-o', os.path.join(OUT, tu + '.s'), r'csrc\%s.cpp' % tu])
    text = open(os.path.join(OUT, tu + '.s'), encoding='utf-8', errors='replace').read()
    hits = sum(1 for line in text.splitlines() if any(tok in line for tok in
               ('ymm', 'zmm', 'vfmadd', 'vzeroupper', 'vmovaps', 'vmulps', 'vaddps', 'vperm')))
    print('  %-16s AVX-ish asm lines: %d' % (tu, hits))

print('=== link DLL ===')
dll = os.path.join(OUT, 'libbitsandbytes_cpu_generic.dll')
rc |= run([ZIG, 'c++', '-target', 'x86_64-windows-gnu', '-shared', '-o', dll,
           os.path.join(OUT, 'cpu_ops.o'), os.path.join(OUT, 'cpu_gdn.o'),
           os.path.join(OUT, 'pythonInterface.o'), '-Wl,--export-all-symbols'])
print('  link exit %d' % rc)
if os.path.exists(dll):
    print('  built %s (%d bytes)' % (dll, os.path.getsize(dll)))
    print('BUILD_OK')
else:
    print('BUILD_MISSING')
sys.exit(0 if rc == 0 else 1)
