"""The last AVX code in the portable build came from a scalar-protection pragma.

The region around build_quantize_lut/LUTCache/quantize_cpu_impl is wrapped in
`#pragma GCC target("avx2,fma,no-avx512f")` whose purpose is to stop GCC from emitting EVEX in
plain scalar code of the AVX-512 build. In a generic x86-64 build the same pragma is what forces
ymm instructions into otherwise scalar functions (53 AVX instructions attributed to
quantize_cpu_impl/build_quantize_lut/LUTCache). Under BNB_CPU_PORTABLE the pragma must not be
applied at all.
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

PATH = r'D:\work\bitsandbytes-CPU\bitsandbytes\csrc\cpu_ops.cpp'

PUSH_OLD = (
    '// Prevent GCC/Clang from emitting EVEX-encoded AVX512 instructions in plain scalar code.\n'
    '// The global -mavx512vl flag can cause GCC to fold broadcasts into EVEX encoding (e.g. vmulps {1to4})\n'
    '// which would SIGILL on non-AVX512 CPUs like Zen3. These functions are scalar C++ and don\'t need AVX512.\n'
    '#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\n'
    '#pragma GCC push_options\n'
    '#pragma GCC target("avx2,fma,no-avx512f")\n'
    '#endif\n'
)
PUSH_NEW = (
    '// Prevent GCC/Clang from emitting EVEX-encoded AVX512 instructions in plain scalar code.\n'
    '// The global -mavx512vl flag can cause GCC to fold broadcasts into EVEX encoding (e.g. vmulps {1to4})\n'
    '// which would SIGILL on non-AVX512 CPUs like Zen3. These functions are scalar C++ and don\'t need AVX512.\n'
    '//\n'
    '// BNB_CPU_PORTABLE: a generic x86-64 build must not apply this target at all -- it would force\n'
    '// ymm instructions into functions that are supposed to stay baseline, defeating the artifact.\n'
    '#if !defined(BNB_CPU_PORTABLE) && defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\n'
    '#pragma GCC push_options\n'
    '#pragma GCC target("avx2,fma,no-avx512f")\n'
    '#endif\n'
)

POP_OLD = (
    'void quantize_cpu_fp16(float* code, fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n) {\n'
    '    quantize_cpu_impl<fp16_t>(code, A, absmax, out, blocksize, n);\n'
    '}\n'
    '\n'
    '#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\n'
    '#pragma GCC pop_options\n'
    '#endif\n'
)
POP_NEW = (
    'void quantize_cpu_fp16(float* code, fp16_t* A, float* absmax, unsigned char* out, long long blocksize, long long n) {\n'
    '    quantize_cpu_impl<fp16_t>(code, A, absmax, out, blocksize, n);\n'
    '}\n'
    '\n'
    '#if !defined(BNB_CPU_PORTABLE) && defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))\n'
    '#pragma GCC pop_options\n'
    '#endif\n'
)


def main() -> int:
    text = io.open(PATH, encoding='utf-8').read()
    for old, new, label in ((PUSH_OLD, PUSH_NEW, 'push'), (POP_OLD, POP_NEW, 'pop')):
        if text.count(old) != 1:
            print('ABORT %s anchor=%d' % (label, text.count(old)))
            return 1
        text = text.replace(old, new, 1)
    io.open(PATH, 'w', encoding='utf-8', newline='').write(text)
    print('ok    scalar-protection pragma gated on BNB_CPU_PORTABLE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
