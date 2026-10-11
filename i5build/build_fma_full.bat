@echo off
setlocal
set VCVARS=D:\vs\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul 2>&1
cd /d "%~dp0"
cl /nologo /O2 /Ob2 /arch:AVX2 /fp:fast /GL /utf-8 /openmp cpu_fma_full.c /Fe:cpu_fma_full.exe /link /LTCG
if errorlevel 1 exit /b 1
echo [OK]
