"""Build a portable x86-64 artifact (no AVX2/FMA) for pre-Haswell machines.

The runtime half already exists: has_avx2_cpu()/gdn_has_avx2() guard every SIMD path and every
kernel has a scalar twin. What is missing is a BINARY that does not contain AVX2 instructions at
all, because the shipped artifacts are compiled with -mavx2 -mfma and the compiler may emit AVX2
anywhere (verified: the generic -march=x86-64 build does not even compile, since the intrinsics are
guarded by `__GNUC__ && x86` / GDN_X86 rather than `__AVX2__`).

This patch adds the compile-time switch BNB_CPU_PORTABLE:
  cpu_ops.cpp   every `__AVX2__ || x86` guarded region additionally requires BNB_AVX2_CODE, and
                has_avx2_cpu() returns false, so the scalar reference path is always taken;
  cpu_gdn.cpp   VO<true> and the <true> dispatch instantiations are excluded; the scalar twin runs;
  build_linux.sh  --generic builds libbitsandbytes_cpu_generic.so with -march=x86-64 and asserts
                that the artifact contains no ymm/zmm/FMA instructions.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
CSRC = REPO + r'\bitsandbytes\csrc'
BUILD = REPO + r'\bitsandbytes\build_linux.sh'

OPS = CSRC + r'\cpu_ops.cpp'
GDN = CSRC + r'\cpu_gdn.cpp'

GUARD = '#if defined(__AVX2__) || (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))'
GUARD_NEW = ('#if BNB_AVX2_CODE && (defined(__AVX2__) || '
             '(defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))))')

MACRO_ANCHOR = '// Env override for A/B benchmarking: BNB_CPU_NO_AVX2=1 forces the scalar path.\n'
MACRO_NEW = (
    '// ---------------------------------------------------------------------------\n'
    '// Portability switch for the generic artifact (BNB_CPU_PORTABLE).\n'
    '//\n'
    '// The regular build compiles this file with -mavx2 -mfma, so the compiler is free to emit\n'
    '// AVX2/FMA anywhere -- including in the scalar fallback twins -- which is why the artifact\n'
    '// cannot run on a pre-Haswell x86 CPU even though every kernel has a runtime CPUID guard.\n'
    '// With BNB_CPU_PORTABLE the AVX2/FMA code is removed at COMPILE time: the guards below\n'
    '// become false, has_avx2_cpu() reports false, and only baseline x86-64 (SSE2) code is\n'
    '// emitted, so the generic library runs anywhere and simply takes the scalar path.\n'
    '// ---------------------------------------------------------------------------\n'
    '#if defined(BNB_CPU_PORTABLE)\n'
    '#define BNB_AVX2_CODE 0\n'
    '#else\n'
    '#define BNB_AVX2_CODE 1\n'
    '#endif\n'
    '\n'
) + MACRO_ANCHOR

HAS_OLD = ('static inline bool has_avx2_cpu() {\n'
           '#if defined(__x86_64__) || defined(_M_X64)\n')
HAS_NEW = ('static inline bool has_avx2_cpu() {\n'
           '#if defined(BNB_CPU_PORTABLE)\n'
           '    // 通用产物里没有 AVX2/FMA 代码可调，直接走标量路径。\n'
           '    return false;\n'
           '#elif defined(__x86_64__) || defined(_M_X64)\n')

GDN_VO_OLD = '#if defined(GDN_X86)\ntemplate <> struct VO<true> {'
GDN_VO_NEW = '#if defined(GDN_X86) && !defined(BNB_CPU_PORTABLE)\ntemplate <> struct VO<true> {'
GDN_DISP_OLD = '#if defined(GDN_X86)\n    if (gdn_has_avx2())'
GDN_DISP_NEW = '#if defined(GDN_X86) && !defined(BNB_CPU_PORTABLE)\n    if (gdn_has_avx2())'

BUILD_OLD_ARG = (
    'RUN_SELFTEST=0\n'
    'BUILD_AVX512=0\n'
    'for arg in "$@"; do\n'
    '  case "$arg" in\n'
    '    --clang) COMPILER="clang++" ;;\n'
    '    --no-avx2) NO_AVX2=1 ;;\n'
    '    --selftest) RUN_SELFTEST=1 ;;\n'
    '    --avx512) BUILD_AVX512=1 ;;\n'
    '  esac\n'
    'done\n'
)
BUILD_NEW_ARG = (
    'RUN_SELFTEST=0\n'
    'BUILD_AVX512=0\n'
    'BUILD_GENERIC=0\n'
    'for arg in "$@"; do\n'
    '  case "$arg" in\n'
    '    --clang) COMPILER="clang++" ;;\n'
    '    --no-avx2) NO_AVX2=1 ;;\n'
    '    --selftest) RUN_SELFTEST=1 ;;\n'
    '    --avx512) BUILD_AVX512=1 ;;\n'
    '    --generic) BUILD_GENERIC=1 ;;\n'
    '  esac\n'
    'done\n'
)

GENERIC_ANCHOR = 'echo "============================================================"\necho "DONE."\n'
GENERIC_BLOCK = (
    '# 可选的通用 x86-64 变体：编译期去掉全部 AVX2/FMA（BNB_CPU_PORTABLE），只保留 SSE2 基线，\n'
    '# 可在 2013 年之前的 x86（Haswell 之前）上运行；载入端按 CPUID 选择，见 _cpu_caps.py。\n'
    'if [ "$BUILD_GENERIC" = "1" ]; then\n'
    '  if [ "$ARCH" != "x86_64" ]; then\n'
    '    echo "  [GENERIC] 跳过：架构 $ARCH 不是 x86_64"\n'
    '  else\n'
    '    echo "[2c/4] 编译通用 x86-64 版（libbitsandbytes_cpu_generic.so，无 AVX2/FMA）..."\n'
    '    OUTGEN="bitsandbytes/libbitsandbytes_cpu_generic.so"\n'
    '    rm -f "$OUTGEN"\n'
    '    "$COMPILER" -O2 -std=c++17 -fno-math-errno -fopenmp -march=x86-64 -DBNB_CPU_PORTABLE=1 \\\n'
    '      -fPIC -shared -DNOMINMAX -DNDEBUG \\\n'
    '      -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \\\n'
    '      -I csrc \\\n'
    '      csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \\\n'
    '      -o "$OUTGEN" \\\n'
    '      -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu_generic.so\n'
    '    echo "  -> $OUTGEN ($(du -h "$OUTGEN" | cut -f1))"\n'
    '    if command -v objdump >/dev/null 2>&1; then\n'
    '      AVX_INSNS=$(objdump -d "$OUTGEN" 2>/dev/null | grep -cE \'%[yz]mm|vfmadd|vzeroupper|vmovaps|vmulps|vaddps\' || true)\n'
    '      echo "  可移植性检查：AVX 指令 $AVX_INSNS 条（generic 产物必须为 0）"\n'
    '      if [ "$AVX_INSNS" -ne 0 ]; then\n'
    '        echo "ERROR: generic 产物包含 AVX 指令，它不能在非 AVX2 机器上运行。"\n'
    '        echo "       检查是否有 AVX2 代码没被 BNB_CPU_PORTABLE 排除。"\n'
    '        exit 4\n'
    '      fi\n'
    '    fi\n'
    '  fi\n'
    'fi\n'
    '\n'
) + GENERIC_ANCHOR


def main() -> int:
    ops = io.open(OPS, encoding='utf-8').read()
    n = ops.count(GUARD)
    if n != 31:
        print('ABORT cpu_ops guard count=%d (expected 31)' % n)
        return 1
    ops = ops.replace(GUARD, GUARD_NEW)
    if ops.count(MACRO_ANCHOR) != 1:
        print('ABORT macro anchor')
        return 1
    ops = ops.replace(MACRO_ANCHOR, MACRO_NEW, 1)
    if ops.count(HAS_OLD) != 1:
        print('ABORT has_avx2_cpu anchor=%d' % ops.count(HAS_OLD))
        return 1
    ops = ops.replace(HAS_OLD, HAS_NEW, 1)
    io.open(OPS, 'w', encoding='utf-8', newline='').write(ops)
    print('ok    cpu_ops.cpp: BNB_AVX2_CODE on %d guards + portable has_avx2_cpu' % n)

    gdn = io.open(GDN, encoding='utf-8').read()
    if gdn.count(GDN_VO_OLD) != 1:
        print('ABORT gdn VO anchor=%d' % gdn.count(GDN_VO_OLD))
        return 1
    gdn = gdn.replace(GDN_VO_OLD, GDN_VO_NEW, 1)
    n2 = gdn.count(GDN_DISP_OLD)
    if n2 != 2:
        print('ABORT gdn dispatch anchor=%d (expected 2)' % n2)
        return 1
    gdn = gdn.replace(GDN_DISP_OLD, GDN_DISP_NEW)
    io.open(GDN, 'w', encoding='utf-8', newline='').write(gdn)
    print('ok    cpu_gdn.cpp: VO<true> + 2 dispatch guards excluded in portable build')

    build = io.open(BUILD, encoding='utf-8').read()
    if build.count(BUILD_OLD_ARG) != 1:
        print('ABORT build args anchor=%d' % build.count(BUILD_OLD_ARG))
        return 1
    build = build.replace(BUILD_OLD_ARG, BUILD_NEW_ARG, 1)
    if build.count(GENERIC_ANCHOR) != 1:
        print('ABORT build generic anchor=%d' % build.count(GENERIC_ANCHOR))
        return 1
    build = build.replace(GENERIC_ANCHOR, GENERIC_BLOCK, 1)
    io.open(BUILD, 'w', encoding='utf-8', newline='').write(build)
    print('ok    build_linux.sh: --generic + AVX-instruction assertion')
    return 0


if __name__ == '__main__':
    sys.exit(main())
