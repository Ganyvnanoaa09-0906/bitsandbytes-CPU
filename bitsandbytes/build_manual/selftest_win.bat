@echo off
setlocal
REM  selftest_win.bat -- Windows counterpart of `bash build_linux.sh --selftest`.
REM  MSVC has no `clang++ a.c` form: cl infers language from the extension, so we
REM  pass /TP to compile selftest_cpu.c as C++ (the documented Linux line hands the
REM  file to g++, which does the same thing). Without /TP the extern "C" block in
REM  selftest_cpu.c is skipped and every kernel symbol fails to link.
REM
REM  /we4556 promotes "intrinsic immediate out of range" from warning to error,
REM  which is what GCC/Clang do unconditionally -- so a clean build here is real
REM  evidence the kernels are portable to those toolchains, not a false pass.
REM
REM  Run inside "x64 Native Tools Command Prompt for VS" or after vcvars64.bat.

cd /d "%~dp0.."
if not exist "csrc\cpu_ops.cpp" (
    echo [ERROR] csrc\cpu_ops.cpp not found. Run from the bitsandbytes repo root.
    exit /b 1
)

if not exist "build_manual" mkdir "build_manual"

echo [1/2] compiling torch-free selftest (MSVC) ...
cl /nologo /O2 /W4 /we4556 /TP /std:c++17 /EHsc /utf-8 /arch:AVX2 /openmp:experimental ^
   /DNOMINMAX /DNDEBUG /DBUILD_CUDA=0 /DBUILD_HIP=0 /DBUILD_XPU=0 /I csrc ^
   selftest_cpu.c csrc\cpu_ops.cpp csrc\cpu_gdn.cpp csrc\pythonInterface.cpp ^
   /Fe:build_manual\selftest_cpu.exe /link /SUBSYSTEM:CONSOLE
if errorlevel 1 (
    echo [ERROR] selftest build failed
    exit /b 1
)

echo [2/2] running selftest ...
"build_manual\selftest_cpu.exe"
set RC=%errorlevel%
echo [selftest exit code: %RC%]
exit /b %RC%
