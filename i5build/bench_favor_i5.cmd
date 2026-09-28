@echo off
REM ============================================================
REM bench_favor_i5.cmd -- build bench_kernel.c on the i5 with a chosen /favor
REM ------------------------------------------------------------
REM ARGUMENT: intel | amd
REM
REM WHY /TP: bench_kernel.c declares the kernel entry points inside
REM `extern "C" { }`. Compiled as plain C that is invalid and the symbols take
REM C++ linkage, so the link fails. Same reasoning as selftest_cpu.c.
REM
REM WHY /arch:AVX2: without it has_avx2_cpu()'s compile-time branch is never
REM taken, the scalar fallback gets measured, and the /favor comparison becomes a
REM comparison of scalar code -- i.e. it would measure nothing.
REM
REM Paths are hardcoded per machine on purpose. An earlier version tried to fall
REM back from one machine's paths to the other's and mis-detected: `if not exist`
REM on a path handed in as an empty string evaluated true, so it silently pointed
REM at the wrong repo.
REM ============================================================
setlocal
set VCVARS=D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat
set REPO=C:\Users\GanYv\bnb_repo
set FLAVOR=%~1
if "%FLAVOR%"=="" set FLAVOR=intel
set FAVOR=
if /i "%FLAVOR%"=="intel" set FAVOR=/favor:INTEL64
if /i "%FLAVOR%"=="amd"   set FAVOR=/favor:AMD64
if not defined FAVOR ( echo [ERROR] arg must be intel or amd & exit /b 2 )

if not exist "%VCVARS%" ( echo [ERROR] vcvars missing: %VCVARS% & exit /b 2 )
if not exist "%REPO%\csrc\cpu_ops.cpp" ( echo [ERROR] repo missing: %REPO% & exit /b 2 )
if not exist "%REPO%\bench_kernel.c" ( echo [ERROR] bench_kernel.c missing in %REPO% & exit /b 2 )

call "%VCVARS%" >nul 2>&1
cd /d "%REPO%"

echo [build] flavor=%FLAVOR%  favor=%FAVOR%
REM Argument order matters here (measured): putting `/favor:X ... /Fe:out.exe
REM /link /SUBSYSTEM:CONSOLE` all at the end made cl exit after the OpenMP info
REM messages with NO error and NO exe -- it left a 14-byte stub. Moving /Fe: up
REM beside the sources and /favor before /link, matching the order that works in
REM build_manual.bat, is the fix. Also delete any stale output first, so
REM "did it build" cannot be answered by a leftover file.
if exist "build_manual\bench_%FLAVOR%.exe" del /q "build_manual\bench_%FLAVOR%.exe"
cl /nologo /O2 /W4 /we4556 /TP /std:c++17 /EHsc /utf-8 /arch:AVX2 /openmp:experimental ^
   /DNOMINMAX /DNDEBUG /DBUILD_CUDA=0 /DBUILD_HIP=0 /DBUILD_XPU=0 /I csrc ^
   %FAVOR% ^
   bench_kernel.c csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp ^
   /Fe:build_manual\bench_%FLAVOR%.exe /link /SUBSYSTEM:CONSOLE
echo [build] cl exit code = %errorlevel%
if not exist "build_manual\bench_%FLAVOR%.exe" (
    echo [ERROR] no exe produced
    exit /b 1
)
for %%G in ("build_manual\bench_%FLAVOR%.exe") do echo [build] OK -> %%~nxG  %%~zG bytes
exit /b 0
