@echo off
REM build_bw.bat -- same toolchain as the project's other probes. PURE ASCII.
REM /arch:AVX2 matters here: without it the compiler may emit SSE2 for the
REM streaming loop and understate the achievable bandwidth.
setlocal
set VCVARS=D:\vs\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" echo [ERROR] vcvars64.bat missing & exit /b 1
call "%VCVARS%" >nul 2>&1
if errorlevel 1 echo [ERROR] vcvars64 failed & exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /GL /utf-8 /openmp cpu_bw.c /Fe:cpu_bw.exe /link /LTCG
if errorlevel 1 exit /b 1
echo [OK] built cpu_bw.exe
