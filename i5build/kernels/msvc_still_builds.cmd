@echo off
call D:\vs\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1
cd /d D:\work\bitsandbytes-CPU\i5build\kernels
cl /nologo /O2 /std:c++17 /utf-8 /arch:AVX2 /openmp:experimental /EHsc /DNOMINMAX /DNDEBUG /DHAS_OPENMP ^
   /DBUILD_CUDA=0 /DBUILD_HIP=0 /DBUILD_XPU=0 ^
   /I D:\work\bitsandbytes-CPU\bitsandbytes\csrc ^
   /c D:\work\bitsandbytes-CPU\bitsandbytes\csrc\cpu_ops.cpp ^
   /Fo:msvc_check.obj 2>&1 | findstr /i "error" | findstr /v "C4819"
if exist msvc_check.obj (echo => MSVC BUILD OK with the cpu_cache.h change) else (echo => MSVC BUILD FAILED)
del msvc_check.obj 2>nul
