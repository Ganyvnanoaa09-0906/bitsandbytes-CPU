"""Windows: build the generic (no-AVX2) DLL too.

Same contract as Linux: `BNB_BUILD_GENERIC=1` compiles a second DLL with BNB_CPU_PORTABLE and NO
/arch:AVX2 (baseline x86-64), exported through export_generic.def. The release script packages it
when present. The loader already prefers libbitsandbytes_cpu_generic.dll when the CPU lacks AVX2
(_cpu_caps.avx2_supported() false).

This dev machine has no MSVC, so the compile is verified by CI (windows-latest) rather than here.
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

PKG = r'D:\work\bitsandbytes-CPU\bitsandbytes'
BAT = PKG + r'\build_manual\build_manual.bat'
REL = PKG + r'\build_release_windows.ps1'
DEF = PKG + r'\build_manual\export_generic.def'

BAT_ANCHOR = (
    '    if errorlevel 1 (\n'
    '        echo [ERROR] AVX-512 build failed\n'
    '        exit /b 1\n'
    '    )\n'
    ')\n'
)
BAT_NEW = BAT_ANCHOR + (
    '\n'
    'REM ---- optional generic x86-64 variant (no AVX2/FMA) ---------------------------\n'
    'REM  For machines that cannot run the AVX2 DLL. Compiled WITHOUT /arch:AVX2 and with\n'
    'REM  /DBNB_CPU_PORTABLE=1, which removes all AVX2/FMA code at compile time; the loader\n'
    'REM  in bitsandbytes/_cpu_caps.py selects it only when the CPU lacks AVX2. Needs VS2017+.\n'
    'REM  Off by default: set BNB_BUILD_GENERIC=1.\n'
    'if /i "%BNB_BUILD_GENERIC%"=="1" (\n'
    '    echo [1c/2] compiling generic x86-64 variant (no AVX2/FMA) ...\n'
    '    cl /nologo /O2 /Ob2 /fp:fast /openmp:experimental %FAVOR% ^\n'
    '       /std:c++17 /EHsc /utf-8 ^\n'
    '       /DBNB_CPU_PORTABLE=1 /DNOMINMAX /DNDEBUG /DWIN32 /D_WINDOWS /I csrc /LD ^\n'
    '       csrc\\cpu_ops.cpp csrc\\cpu_gdn.cpp csrc\\pythonInterface.cpp ^\n'
    '       /Fe:bitsandbytes\\libbitsandbytes_cpu_generic.dll /link /DEF:build_manual\\export_generic.def\n'
    '    if errorlevel 1 (\n'
    '        echo [ERROR] generic x86-64 build failed\n'
    '        exit /b 1\n'
    '    )\n'
    ')\n'
)

REL_ANCHOR = (
    '} else {\n'
    '    Write-Host "[..] no AVX-512 DLL (build_manual.bat with BNB_BUILD_AVX512=1 to add one)" -ForegroundColor DarkGray\n'
    '}\n'
)
REL_NEW = REL_ANCHOR + (
    '# Optional generic x86-64 variant (no AVX2/FMA), for CPUs the AVX2 DLL cannot run on.\n'
    '$generic = "bitsandbytes\\libbitsandbytes_cpu_generic.dll"\n'
    'if (Test-Path $generic) {\n'
    '    Copy-Item -Path $generic -Destination (Join-Path $stagePkg "libbitsandbytes_cpu_generic.dll") -Force\n'
    '    Write-Host "[..] packaged libbitsandbytes_cpu_generic.dll" -ForegroundColor Cyan\n'
    '} else {\n'
    '    Write-Host "[..] no generic DLL (build_manual.bat with BNB_BUILD_GENERIC=1 to add one)" -ForegroundColor DarkGray\n'
    '}\n'
)

DEF_TEXT = '''LIBRARY libbitsandbytes_cpu_generic
EXPORTS
    gdn_fwd_cpu
    gdn_bwd_cpu
    cquantize_blockwise_cpu_fp32
    cquantize_blockwise_cpu_bf16
    cquantize_blockwise_cpu_fp16
    cdequantize_blockwise_cpu_fp32
    cdequantize_blockwise_cpu_bf16
    cdequantize_blockwise_cpu_fp16
    cdequantize_blockwise_cpu_fp4_fp32
    cdequantize_blockwise_cpu_fp4_bf16
    cdequantize_blockwise_cpu_fp4_fp16
    cdequantize_blockwise_cpu_nf4_fp32
    cdequantize_blockwise_cpu_nf4_bf16
    cdequantize_blockwise_cpu_nf4_fp16
    cquantize_blockwise_cpu_4bit_fp32
    cquantize_blockwise_cpu_4bit_bf16
    cquantize_blockwise_cpu_4bit_fp16
    cint8_vector_quant_cpu_fp32
    cint8_vector_quant_cpu_bf16
    cint8_vector_quant_cpu_fp16
    cgemv_4bit_inference_cpu_fp32
    cgemv_4bit_inference_cpu_bf16
    cgemv_4bit_inference_cpu_fp16
    cgemv_fp16w_inference_cpu_fp32
    cgemv_fp16w_inference_cpu_bf16
    cgemv_fp16w_inference_cpu_fp16
    cgemm_8bit_inference_cpu_fp32
    coptimizer_update_8bit_blockwise_cpu
    cfused_rmsnorm_fwd_cpu
    cfused_rmsnorm_bwd_cpu
    cfused_swiglu_fwd_cpu
    cfused_swiglu_bwd_cpu
    cfused_add_scale_cpu
    cfused_swiglu_strided_fwd_cpu
    cfused_swiglu_strided_bwd_cpu
    cfused_add_scale_inplace_cpu
    cfused_add_scale_rmsnorm_cpu
'''


def main() -> int:
    bat = io.open(BAT, encoding='utf-8').read()
    if bat.count(BAT_ANCHOR) != 1:
        print('ABORT bat anchor=%d' % bat.count(BAT_ANCHOR))
        return 1
    io.open(BAT, 'w', encoding='utf-8', newline='').write(bat.replace(BAT_ANCHOR, BAT_NEW, 1))
    print('ok    build_manual.bat: BNB_BUILD_GENERIC=1 block')
    rel = io.open(REL, encoding='utf-8').read()
    if rel.count(REL_ANCHOR) != 1:
        print('ABORT release anchor=%d' % rel.count(REL_ANCHOR))
        return 1
    io.open(REL, 'w', encoding='utf-8', newline='').write(rel.replace(REL_ANCHOR, REL_NEW, 1))
    print('ok    build_release_windows.ps1: package generic DLL')
    io.open(DEF, 'w', encoding='ascii', newline='').write(DEF_TEXT)
    print('ok    export_generic.def written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
