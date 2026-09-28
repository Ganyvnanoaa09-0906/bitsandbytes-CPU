@echo off
REM ============================================================
REM selftest_i5.cmd -- build and run the torch-free kernel selftest on the i5
REM ------------------------------------------------------------
REM Uses D:\vs2022bt (the BuildTools install that HAS a Windows SDK). The
REM pre-existing D:\vs on this box has no SDK and cannot compile at all.
REM
REM /TP is required, not cosmetic: selftest_cpu.c wraps its kernel declarations
REM in `#ifdef __cplusplus / extern "C" {`, so compiling it as plain C skips
REM that block, the symbols take C++ linkage, and every kernel symbol fails to
REM link. Same reasoning as build_manual/selftest_win.bat in the repo.
REM
REM /arch:AVX2 matters here: without it the AVX2 dispatch is never taken and the
REM vector-vs-scalar cross-check in the selftest would compare scalar to scalar,
REM passing for the wrong reason.
REM ============================================================
setlocal
set VCVARS=D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat
set REPO=C:\Users\GanYv\bnb_repo

if not exist "%VCVARS%" ( echo [ERROR] vcvars not found & exit /b 2 )
if not exist "%REPO%\csrc\cpu_ops.cpp" ( echo [ERROR] repo missing & exit /b 2 )

call "%VCVARS%" >nul 2>&1
echo [env] VCToolsInstallDir=%VCToolsInstallDir%
cd /d "%REPO%"

echo.
echo === build selftest (/arch:AVX2, /TP) ===
cl /nologo /O2 /W4 /we4556 /TP /std:c++17 /EHsc /utf-8 /arch:AVX2 /openmp:experimental ^
   /DNOMINMAX /DNDEBUG /DBUILD_CUDA=0 /DBUILD_HIP=0 /DBUILD_XPU=0 /I csrc ^
   selftest_cpu.c csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp ^
   /Fe:build_manual\selftest_i5.exe /link /SUBSYSTEM:CONSOLE
if errorlevel 1 (
    echo [ERROR] selftest build failed
    exit /b 1
)

echo.
echo === run ===
build_manual\selftest_i5.exe
set RC=%errorlevel%
echo.
echo [selftest exit code = %RC%]
exit /b %RC%
