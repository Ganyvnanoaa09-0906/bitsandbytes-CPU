@echo off
REM Build and run a kernel benchmark with MSVC. Usage: build.cmd flops_peak [threads]
setlocal
set SRC=%~1
if "%SRC%"=="" set SRC=flops_peak
set THREADS=%~2
set VCVARS=D:\vs\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" set VCVARS=D:\vs2022bt\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul 2>&1
if errorlevel 1 (echo vcvars failed & exit /b 1)
cd /d "%~dp0"
echo === compiling %SRC%.c ===
REM /TP compiles the file as C++, and that is required rather than cosmetic: in C mode
REM MSVC's OpenMP front end rejects `for (int i = 0; ...)` inside `#pragma omp for` with
REM "the OpenMP for statement has an incorrect initialization format" (C3015). /std:c11
REM and /openmp:llvm both still reject it; only C++ mode accepts the loop variable being
REM declared in the for-init. Verified against kernels/omp_probe.c, which compiles all
REM four loop forms only under /TP.
cl /nologo /O2 /TP /arch:AVX2 /openmp /W3 "%SRC%.c" /Fe:"%SRC%.exe" /Fo:"%SRC%.obj"
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo === running %SRC%.exe ===
"%SRC%.exe" %THREADS%
