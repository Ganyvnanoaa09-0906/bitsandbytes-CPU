"""Doctor reports the AVX2/AVX-512 decision; Windows build scripts can produce the AVX-512 DLL.

The loader change (cextension + gdn_cpu + _cpu_caps) makes the AVX-512 library opt-in by CPU
capability. Users need to be able to see which artifact would load, and the Windows build needs a
way to produce the second DLL (the dev machine has no MSVC, so this path is untested here and is
written the same way as the existing AVX2 command).

  bitsandbytes_cpu_cli.py  doctor: print capability, whether the AVX-512 DLL exists, and which
                           one will load; check symbols on the chosen artifact
  build_manual/build_manual.bat  set BNB_BUILD_AVX512=1 -> second cl pass with /arch:AVX512
  build_release_windows.ps1      copy the AVX-512 DLL when present (or compile it when
                                 $env:BNB_BUILD_AVX512=1)
"""
from __future__ import annotations
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

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
PKG = REPO + r'\bitsandbytes'

EDITS = []

# ---------------------------------------------------------------- CLI doctor
EDITS.append((
    REPO + r'\bitsandbytes_cpu_cli.py',
    '    print(f"  this platform wants  {os.path.basename(target)}")\n'
    '    if not os.path.isfile(target):\n',
    '    # The loader prefers the AVX-512 artifact when the CPU supports it (see\n'
    '    # bitsandbytes/_cpu_caps.py); doctor has to report the same decision, not just "the\n'
    '    # file for this platform exists".\n'
    '    try:\n'
    '        caps = _load_from_package("_cpu_caps")\n'
    '        avx512_ok = bool(caps.avx512_supported())\n'
    '    except Exception:  # pragma: no cover - doctor must still work if the helper is absent\n'
    '        avx512_ok = False\n'
    '    avx512 = os.path.join(pkg, f"libbitsandbytes_cpu_avx512{suffix}")\n'
    '    has_avx512 = os.path.isfile(avx512)\n'
    '    print(f"  cpu supports AVX-512  {\'yes\' if avx512_ok else \'no\'}")\n'
    '    print(f"  AVX-512 library       {os.path.basename(avx512)} "\n'
    '          f"{\'(present)\' if has_avx512 else \'(absent)\'}")\n'
    '    if avx512_ok and has_avx512:\n'
    '        target = avx512\n'
    '    print(f"  this process will load  {os.path.basename(target)}")\n'
    '    if not os.path.isfile(target):\n',
))

# ---------------------------------------------------------------- Windows manual build
EDITS.append((
    PKG + r'\build_manual\build_manual.bat',
    'echo [2/2] checking artifact ...\n',
    'REM ---- optional AVX-512 variant -------------------------------------------------\n'
    'REM  The AVX-512 library is a separate artifact because it must only be loaded on\n'
    'REM  CPUs that support AVX-512 (whole-TU -mavx512f codegen SIGILLs elsewhere); the\n'
    'REM  loader in bitsandbytes/_cpu_caps.py picks it automatically when it exists.\n'
    'REM  Off by default: set BNB_BUILD_AVX512=1. Needs VS2017+ for /arch:AVX512.\n'
    'REM  Not exercised on this dev machine (no MSVC installed there); same shape as the\n'
    'REM  AVX2 command above.\n'
    'if /i "%BNB_BUILD_AVX512%"=="1" (\n'
    '    echo [1b/2] compiling AVX-512 variant ...\n'
    '    cl /nologo /O2 /Ob2 /arch:AVX512 /fp:fast /openmp:experimental /GL /Qpar %FAVOR% ^\n'
    '       /std:c++17 /EHsc /utf-8 ^\n'
    '       /DNOMINMAX /DNDEBUG /DWIN32 /D_WINDOWS /I csrc /LD ^\n'
    '       csrc\\cpu_ops.cpp csrc\\cpu_gdn.cpp csrc\\pythonInterface.cpp ^\n'
    '       /Fe:bitsandbytes\\libbitsandbytes_cpu_avx512.dll /link /LTCG /DEF:build_manual\\export_avx512.def\n'
    '    if errorlevel 1 (\n'
    '        echo [ERROR] AVX-512 build failed\n'
    '        exit /b 1\n'
    '    )\n'
    ')\n'
    '\n'
    'echo [2/2] checking artifact ...\n',
))

# ---------------------------------------------------------------- Windows release packaging
EDITS.append((
    PKG + r'\build_release_windows.ps1',
    'Copy-Item -Path "$tmpBuild\\libbitsandbytes_cpu.dll" -Destination (Join-Path $stagePkg "libbitsandbytes_cpu.dll") -Force\n',
    'Copy-Item -Path "$tmpBuild\\libbitsandbytes_cpu.dll" -Destination (Join-Path $stagePkg "libbitsandbytes_cpu.dll") -Force\n'
    '# Optional AVX-512 variant: built by build_manual.bat with BNB_BUILD_AVX512=1. It is only\n'
    '# loaded on AVX-512 CPUs (see bitsandbytes/_cpu_caps.py); absent is a normal, complete build.\n'
    '$avx512 = "bitsandbytes\\libbitsandbytes_cpu_avx512.dll"\n'
    'if (Test-Path $avx512) {\n'
    '    Copy-Item -Path $avx512 -Destination (Join-Path $stagePkg "libbitsandbytes_cpu_avx512.dll") -Force\n'
    '    Write-Host "[..] packaged libbitsandbytes_cpu_avx512.dll" -ForegroundColor Cyan\n'
    '} else {\n'
    '    Write-Host "[..] no AVX-512 DLL (build_manual.bat with BNB_BUILD_AVX512=1 to add one)" -ForegroundColor DarkGray\n'
    '}\n',
))


def main() -> int:
    for path, old, new in EDITS:
        text = open(path, encoding='utf-8').read()
        count = text.count(old)
        if count != 1:
            print('ABORT %-64s anchor=%d' % (path.replace(REPO + '\\', ''), count))
            return 1
        open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %s' % path.replace(REPO + '\\', ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
