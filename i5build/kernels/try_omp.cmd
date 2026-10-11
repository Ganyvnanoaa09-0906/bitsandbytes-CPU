@echo off
setlocal
call D:\vs\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1
cd /d "%~dp0"
echo --- /openmp:llvm ---
cl /nologo /O2 /std:c11 /arch:AVX2 /openmp:llvm /W3 omp_probe.c /Fe:omp_llvm.exe /Fo:omp_llvm.obj
echo llvm_exit=%ERRORLEVEL%
if exist omp_llvm.exe (echo RUN: & omp_llvm.exe)
echo --- /TP (compile as C++) ---
cl /nologo /O2 /TP /arch:AVX2 /openmp /W3 omp_probe.c /Fe:omp_cpp.exe /Fo:omp_cpp.obj
echo cpp_exit=%ERRORLEVEL%
if exist omp_cpp.exe (echo RUN: & omp_cpp.exe)
