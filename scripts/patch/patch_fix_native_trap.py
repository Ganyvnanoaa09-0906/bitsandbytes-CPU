"""Fix the -march=native trap that the real AVX-512 host exposed, and harden the E2E script.

On the AVX-512 container the "AVX2 baseline" library came out with 1204 zmm instruction
lines: build_linux.sh used `-march=native` for code generation, and on an AVX-512 build
host native means AVX-512, so the artifact meant for AVX2 machines would SIGILL there.
This is exactly the trap the CMake option was built to avoid, and the Linux manual build
had the same hole.

Fixes:
  build_linux.sh   the AVX2 artifact is compiled with explicit -mavx2 -mfma -mf16c (ISA
                   contract, not build-host detection); the /proc/cpuinfo check now only
                   decides whether to enable AVX2 at all. A post-build assertion counts
                   zmm lines and fails the build if the AVX2 artifact contains any.
  avx512/dual_lib_e2e.sh  same assertion at the E2E level: zmm lines of the AVX2 artifact
                   must be 0, and of the AVX-512 artifact > 0; a violation is reported as
                   PORTABLE_ARTIFACT_FAILED but the run continues so the rest stays visible.
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

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
BUILD = REPO + r'\bitsandbytes\build_linux.sh'
E2E = REPO + r'\avx512\dual_lib_e2e.sh'

OLD_FLAGS = '    ARCH_FLAGS="-march=native -D__AVX2__"\n'
NEW_FLAGS = (
    '    # ★ 这里必须是【显式 ISA】而不是 -march=native：\n'
    '    #   构建机若本身支持 AVX-512（如 EPYC 9Y24 容器），-march=native 会让这份\n'
    '    #   "AVX2 基线"产物里出现 zmm/EVEX 指令，它在 AVX2 机器上一调用就 SIGILL。\n'
    '    #   实测（2026-10-18，AVX-512 容器）：-march=native 版 AVX2 产物有 1204 条 zmm。\n'
    '    #   所以代码生成只用显式 -mavx2 -mfma -mf16c；下面的 /proc/cpuinfo 检测只用来\n'
    '    #   决定"要不要开 AVX2"，不参与指令集选择。\n'
    '    ARCH_FLAGS="-mavx2 -mfma -mf16c -D__AVX2__"\n'
)

OLD_POST = 'echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"\n'
NEW_POST = (
    'echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"\n'
    '\n'
    '# 可移植性断言：AVX2 产物里不允许出现 EVEX(zmm) 指令 —— 一旦出现，这份 .so 在\n'
    '# AVX2 机器上会 SIGILL，而它正是要发给 AVX2 用户的那份。\n'
    'if command -v objdump >/dev/null 2>&1; then\n'
    '  ZMM_LINES=$(objdump -d "$OUT" 2>/dev/null | grep -c zmm || true)\n'
    '  echo "  可移植性检查：zmm 指令 $ZMM_LINES 条（AVX2 产物必须为 0）"\n'
    '  if [ "$ZMM_LINES" -ne 0 ]; then\n'
    '    echo "ERROR: AVX2 产物包含 zmm 指令，它不能在 AVX2 机器上运行。"\n'
    '    echo "       检查编译 flags：代码生成必须是显式 -mavx2 -mfma -mf16c，不能用 -march=native。"\n'
    '    exit 3\n'
    '  fi\n'
    'fi\n'
)

# ---- E2E: replace the size/zmm print loop with assertions ----
E2E_OLD = (
    'for f in bitsandbytes/libbitsandbytes_cpu.so bitsandbytes/libbitsandbytes_cpu_avx512.so; do\n'
    '    if [ ! -f "$PKG/$f" ]; then echo "MISSING $f"; exit 2; fi\n'
    '    ls -la "$PKG/$f"\n'
    '    echo "    zmm instruction lines: $(objdump -d "$PKG/$f" 2>/dev/null | grep -c zmm || true)"\n'
    'done\n'
)
E2E_NEW = (
    'ZMM_AVX2=$(objdump -d "$PKG/bitsandbytes/libbitsandbytes_cpu.so" 2>/dev/null | grep -c zmm || true)\n'
    'ZMM_AVX512=$(objdump -d "$PKG/bitsandbytes/libbitsandbytes_cpu_avx512.so" 2>/dev/null | grep -c zmm || true)\n'
    'for f in bitsandbytes/libbitsandbytes_cpu.so bitsandbytes/libbitsandbytes_cpu_avx512.so; do\n'
    '    if [ ! -f "$PKG/$f" ]; then echo "MISSING $f"; exit 2; fi\n'
    '    ls -la "$PKG/$f"\n'
    'done\n'
    'echo "zmm lines: AVX2 artifact = $ZMM_AVX2 (must be 0) | AVX-512 artifact = $ZMM_AVX512 (must be > 0)"\n'
    'if [ "$ZMM_AVX2" -ne 0 ]; then\n'
    '    echo "PORTABLE_ARTIFACT_FAILED: the AVX2 artifact contains zmm/EVEX instructions"\n'
    '    echo "  it would SIGILL on an AVX2-only CPU; check build_linux.sh flags (-march=native?)"\n'
    'fi\n'
    'if [ "$ZMM_AVX512" -le 0 ]; then\n'
    '    echo "AVX512_ARTIFACT_WITHOUT_EVEX: the AVX-512 artifact contains no zmm code -- flags not applied?"\n'
    'fi\n'
)


def main() -> int:
    text = io.open(BUILD, encoding='utf-8').read()
    for old, new, label in ((OLD_FLAGS, NEW_FLAGS, 'flags'), (OLD_POST, NEW_POST, 'post-build assert')):
        if text.count(old) != 1:
            print('ABORT build_linux.sh %s anchor=%d' % (label, text.count(old)))
            return 1
        text = text.replace(old, new, 1)
    io.open(BUILD, 'w', encoding='utf-8', newline='').write(text)
    print('ok    build_linux.sh (explicit ISA + zmm assertion)')

    e2e = io.open(E2E, encoding='utf-8').read()
    if e2e.count(E2E_OLD) != 1:
        print('ABORT e2e anchor=%d' % e2e.count(E2E_OLD))
        return 1
    e2e = e2e.replace(E2E_OLD, E2E_NEW, 1)
    io.open(E2E, 'w', encoding='utf-8', newline='').write(e2e)
    print('ok    avx512/dual_lib_e2e.sh (artifact assertions)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
