#!/usr/bin/env bash
# Verify the fork's AVX-512 backend on a machine that can actually execute AVX-512.
#
# Builds the same sources twice -- an AVX2 baseline and an AVX-512 variant -- then
#   1) cross-checks 5 real kernels between the two builds on identical inputs,
#   2) times them (median, same machine, same inputs),
#   3) measures the *hardware* FP32/BF16 ceiling with the same compiler on both flag sets,
# so the kernel numbers can be read against the peak instead of in a vacuum.
#
# No torch, no numpy, no apt installs: g++ and python3 are enough.
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

echo "=== [1/4] build AVX2 baseline (AVX-512 codegen disabled) ==="
time g++ $COMMON $AVX2_FLAGS $SRCS -o "$OUT/libbnb_avx2.so" \
    -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbnb_avx2.so || {
    echo "BUILD_AVX2_FAILED"; exit 2; }
ls -la "$OUT/libbnb_avx2.so"
echo

echo "=== [2/4] build AVX-512 variant ==="
AVX512_SO_OK=no
if g++ $COMMON $AVX512_FLAGS $SRCS -o "$OUT/libbnb_avx512.so" \
       -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbnb_avx512.so; then
    ls -la "$OUT/libbnb_avx512.so"
    AVX512_SO_OK=yes
else
    echo "BUILD_AVX512_FAILED"
fi
echo

echo "=== [3/4] hardware ceiling: same compiler, AVX2 flags vs AVX-512 flags ==="
g++ -O2 $AVX2_FLAGS "$HERE/peak_probe.c" -o "$OUT/peak_avx2" && echo "peak_avx2 built"
if [ "$AVX512_SO_OK" = "yes" ]; then
    g++ -O2 $AVX512_FLAGS "$HERE/peak_probe.c" -o "$OUT/peak_avx512" && echo "peak_avx512 built"
else
    echo "peak_avx512 skipped (AVX-512 build failed)"
fi
if [ -x "$OUT/peak_avx2" ]; then echo "--- peak_avx2 ---"; "$OUT/peak_avx2"; fi
if [ -x "$OUT/peak_avx512" ]; then
    if grep -q ' avx512f' /proc/cpuinfo; then
        echo "--- peak_avx512 ---"; "$OUT/peak_avx512"
    else
        echo "peak_avx512 built but not run: this CPU cannot execute AVX-512"
    fi
fi
echo

echo "=== [4/4] correctness + timing of the real kernels (ctypes, no torch) ==="
ARGS="--avx2 $OUT/libbnb_avx2.so"
[ "$AVX512_SO_OK" = "yes" ] && ARGS="$ARGS --avx512 $OUT/libbnb_avx512.so"
python3 "$HERE/bench_ctypes.py" $ARGS
rc=$?
echo
echo "verify exit=$rc"
exit $rc
