@echo off
setlocal
REM ============================================================
REM  bitsandbytes CPU DLL - manual cl build (no CMake), Windows x86/x64
REM  Run inside "x64 Native Tools Command Prompt for VS" or after
REM  calling vcvars64.bat. This file lives in <repo>\build_manual\.
REM
REM    build_manual.bat            auto-detect CPU vendor
REM    build_manual.bat amd        force /favor:AMD64
REM    build_manual.bat intel      force /favor:INTEL64 (compare)
REM
REM  Output: bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll
REM
REM  Why not CMake: local CMake hangs during the MSVC ABI probe and a
REM  stale CMakeCache from another environment breaks configure. This
REM  script compiles the three cpp files with cl directly plus .def
REM  exports. NOTE: keep this file pure ASCII - cmd parses batch files
REM  with the local codepage (GBK on zh-CN), non-ASCII bytes corrupt it.
REM ============================================================

REM locate the bitsandbytes repo root (this bat lives in <repo>\build_manual\)
REM %~dp0 = build_manual\  ->  parent = repo root (contains csrc\ + bitsandbytes\ package)
cd /d "%~dp0.."

REM Error if run from a wrong dir
if not exist "csrc\cpu_ops.cpp" (
    echo [ERROR] csrc\cpu_ops.cpp not found. Run from the bitsandbytes repo root.
    exit /b 1
)
if not exist "build_manual\export.def" (
    echo [ERROR] missing build_manual\export.def
    exit /b 1
)

echo [1/2] compiling cpu_ops.cpp / cpu_gdn.cpp / pythonInterface.cpp ...
REM /GL /LTCG: whole-program optimization + LTCG link (measured +18% train speed)
REM /favor: pick by CPU vendor (AMD64 scheduling for AMD, INTEL64 for Intel)
REM /Qpar: auto-parallelization
set FAVOR=/favor:INTEL64
if /i "%~1"=="amd" (
    set FAVOR=/favor:AMD64
) else if /i "%~1"=="intel" (
    set FAVOR=/favor:INTEL64
) else (
    for /f "tokens=1* delims==" %%a in ('wmic cpu get Manufacturer /value 2^>nul') do (
        echo %%b | find /i "AMD" >nul && set FAVOR=/favor:AMD64
    )
)
echo [build] CPU vendor favor: %FAVOR%
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /openmp:experimental /GL /Qpar %FAVOR% ^
   /std:c++17 /EHsc /utf-8 ^
   /DNOMINMAX /DNDEBUG /DWIN32 /D_WINDOWS /I csrc /LD ^
   csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp ^
   /Fe:bitsandbytes\libbitsandbytes_cpu.dll /link /LTCG /DEF:build_manual\export.def
if errorlevel 1 (
    echo [ERROR] build failed
    exit /b 1
)

echo [2/2] checking artifact ...
if exist "bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll" (
    echo [OK] bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll
) else if exist "bitsandbytes\libbitsandbytes_cpu.dll" (
    echo [OK] bitsandbytes\libbitsandbytes_cpu.dll
) else (
    echo [WARN] dll not found in expected paths, searching...
    dir /s /b libbitsandbytes_cpu.dll 2>nul
)

REM ============================================================
REM  [3/3] disaster tools: CLI sector_mirror.exe + GUI sector_mirror_gui.exe
REM  1) sector_mirror.exe (standalone exe, no bnb kernel dep).
REM     Built with the most basic cl (no /arch:AVX2 /fp:fast --- keep max CPU
REM     compatibility). It reads raw sectors (\.\PhysicalDriveN) to mirror a whole
REM     drive, as a safety net when disk_balancer corrupts data
REM     (see docs_cpu/DISASTER_RECOVERY.md).
REM  2) sector_mirror_gui.exe (pure Win32 GUI: double-click to run, pick source
REM     drive, Browse target image, progress bar --- no command line, no args;
REM     usable when cmd/powershell are broken). Deps (shell32/comdlg32/comctl32/
REM     advapi32) are declared via #pragma, no manual /link.
REM ============================================================
echo [3/3] building sector_mirror.exe (CLI disaster-recovery raw-sector mirror) ...
if exist "tools\sector_mirror.c" (
    cl /nologo /O2 /W4 /utf-8 /DNOMINMAX /DNDEBUG %FAVOR% ^
       tools\sector_mirror.c ^
       /Fe:tools\sector_mirror.exe /link advapi32.lib
    if errorlevel 1 (
        echo [WARN] sector_mirror.exe build failed (non-fatal; recovery tool optional)
    ) else (
        echo [OK] tools\sector_mirror.exe
    )
) else (
    echo [WARN] tools\sector_mirror.c not found, skipping sector_mirror.exe
)

REM ============================================================
REM  [3/3 continued] disaster tool GUI: sector_mirror_gui.exe
REM  pure Win32 GUI: double-click, pick source drive + Browse target image
REM  + progress bar; no command line needed (works when cmd/powershell break).
REM  Deps declared via #pragma; no manual /link. No /arch:AVX2 (max compat).
REM ============================================================
echo [3/3] building sector_mirror_gui.exe (GUI, no-cmd disaster mirror) ...
if exist "tools\sector_mirror_gui.c" (
    cl /nologo /O2 /utf-8 /DNOMINMAX /DNDEBUG %FAVOR% ^
       tools\sector_mirror_gui.c ^
       /Fe:tools\sector_mirror_gui.exe
    if errorlevel 1 (
        echo [WARN] sector_mirror_gui.exe build failed (non-fatal; recovery tool optional)
    ) else (
        echo [OK] tools\sector_mirror_gui.exe
    )
) else (
    echo [WARN] tools\sector_mirror_gui.c not found, skipping sector_mirror_gui.exe
)

REM ============================================================
REM  [3/3 continued] signature carve: sector_carve.exe (CLI) + sector_carve_gui.exe (GUI)
REM  Carve still-intact files from a raw disk/image by magic bytes, bypassing a broken
REM  MFT (PNG/JPEG/GIF/ZIP/PDF/MP4/docx/xlsx/pptx). GUI double-clicks, no cmd needed;
REM  both /MT static, only depend on system DLLs.
REM ============================================================
if exist "tools\sector_carve.c" (
    echo [3/3] building sector_carve.exe (signature carve, CLI) ...
    cl /nologo /O2 /W3 /utf-8 /DNOMINMAX /DNDEBUG %FAVOR% ^
       tools\sector_carve.c ^
       /Fe:tools\sector_carve.exe /link advapi32.lib
    if errorlevel 1 (
        echo [WARN] sector_carve.exe build failed (non-fatal; recovery tool optional)
    ) else (
        echo [OK] tools\sector_carve.exe
    )
) else (
    echo [WARN] tools\sector_carve.c not found, skipping sector_carve.exe
)
if exist "tools\sector_carve_gui.c" (
    echo [3/3] building sector_carve_gui.exe (signature carve, GUI) ...
    cl /nologo /O2 /utf-8 /DNOMINMAX /DNDEBUG %FAVOR% ^
       tools\sector_carve_gui.c ^
       /Fe:tools\sector_carve_gui.exe
    if errorlevel 1 (
        echo [WARN] sector_carve_gui.exe build failed (non-fatal; recovery tool optional)
    ) else (
        echo [OK] tools\sector_carve_gui.exe
    )
) else (
    echo [WARN] tools\sector_carve_gui.c not found, skipping sector_carve_gui.exe
)

exit /b 0
