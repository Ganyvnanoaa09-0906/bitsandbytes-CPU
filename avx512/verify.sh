#!/usr/bin/env bash
# Build the fork's CPU backend twice -- an AVX2 baseline (AVX-512 codegen explicitly off) and an
# AVX-512 variant -- then cross-check them on identical inputs and time the kernels.
#
# Runs on any Ubuntu/Debian x86-64 box with g++ and python3; no torch, no numpy, no apt installs.
# The AVX-512 library is only *exercised* when the CPU really has AVX-512 (the harness checks
# /proc/cpuinfo first), because a non-AVX-512 CPU dies with SIGILL the moment the compiler-
# auto-vectorised ordinary functions execute zmm code.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
SRC="$ROOT/bitsandbytes"
OUT="${TMPDIR:-/tmp}/bnb_avx512_verify"
mkdir -p "$OUT"

echo "=== environment ==="
echo "utc         : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "kernel      : $(uname -srm)"
if [ -f /etc/os-release ]; then . /etc/os-release; echo "distro      : ${PRETTY_NAME:-unknown}"; fi
echo "cpu model   : $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
echo "nproc       : $(nproc)"
echo "avx512 flags: $(grep -o 'avx512[a-z_]*' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
echo "g++         : $(g++ --version 2>&1 | head -1)"
echo "python3     : $(python3 --version 2>&1)"
echo

COMMON="-O2 -std=c++17 -fno-math-errno -fopenmp -fPIC -shared -DNOMINMAX -DNDEBUG \
-DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I $SRC/csrc"
SRCS="$SRC/csrc/cpu_ops.cpp $SRC/csrc/cpu_gdn.cpp $SRC/csrc/pythonInterface.cpp"
AVX2_FLAGS="-mavx2 -mfma -mf16c"
AVX512_FLAGS="-mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl"

# -mavx512bf16 exists on GCC >= 9; probe instead of assuming.
if echo 'int main(){return 0;}' | g++ -mavx512bf16 -x c++ - -o /dev/null 2>/dev/null; then
    AVX512_FLAGS="$AVX512_FLAGS -mavx512bf16"
    BF16_FLAG=yes
else
    BF16_FLAG=no
fi
echo "AVX512-BF16 flag: $BF16_FLAG"
echo

echo "=== [1/3] build AVX2 baseline (AVX-512 codegen disabled) ==="
time g++ $COMMON $AVX2_FLAGS $SRCS -o "$OUT/libbnb_avx2.so" \
    -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbnb_avx2.so
if [ $? -ne 0 ] || [ ! -f "$OUT/libbnb_avx2.so" ]; then
    echo "BUILD_AVX2_FAILED"
    exit 2
fi
ls -la "$OUT/libbnb_avx2.so"
echo

echo "=== [2/3] build AVX-512 variant ==="
if g++ $COMMON $AVX512_FLAGS $SRCS -o "$OUT/libbnb_avx512.so" \
       -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbnb_avx512.so; then
    ls -la "$OUT/libbnb_avx512.so"
else
    echo "BUILD_AVX512_FAILED"
fi
echo

echo "=== [3/3] correctness + timing (ctypes, no torch) ==="
ARGS="--avx2 $OUT/libbnb_avx2.so"
[ -f "$OUT/libbnb_avx512.so" ] && ARGS="$ARGS --avx512 $OUT/libbnb_avx512.so"
python3 "$HERE/bench_ctypes.py" $ARGS
rc=$?
echo
echo "verify exit=$rc"
exit $rc
