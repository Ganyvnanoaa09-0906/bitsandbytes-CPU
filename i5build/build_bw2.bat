@echo off
REM build_bw2.bat -- corrected bandwidth probe. PURE ASCII.
setlocal
set VCVARS=D:\vs\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" echo [ERROR] vcvars64.bat missing & exit /b 1
call "%VCVARS%" >nul 2>&1
if errorlevel 1 echo [ERROR] vcvars64 failed & exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /GL /utf-8 /openmp cpu_bw2.c /Fe:cpu_bw2.exe /link /LTCG
if errorlevel 1 exit /b 1
echo [OK] built cpu_bw2.exe
