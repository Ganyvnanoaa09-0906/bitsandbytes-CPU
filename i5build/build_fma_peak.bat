@echo off
REM build_fma_peak.bat -- pure ASCII only. Same toolchain as the project's other probes.
setlocal
set VCVARS=D:\vs\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" echo [ERROR] vcvars64.bat missing & exit /b 1
call "%VCVARS%" >nul 2>&1
if errorlevel 1 echo [ERROR] vcvars64 failed & exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /GL /utf-8 cpu_fma_peak.c /Fe:cpu_fma_peak.exe /link /LTCG
if errorlevel 1 echo [ERROR] build failed & exit /b 1
echo [OK] cpu_fma_peak.exe
exit /b 0
