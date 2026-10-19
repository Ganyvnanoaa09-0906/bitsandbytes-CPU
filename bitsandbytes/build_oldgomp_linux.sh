#!/usr/bin/env bash
# build_oldgomp_linux.sh — build the manylinux_2_17 wheel WITH OpenMP (glibc 2.17 compatible).
#
# This closes the old-glibc gap: the plain zig cross-build (build_manylinux_wheel.sh) is
# serial because zig's clang cannot emit libgomp ABI calls, so Ubuntu 18.04-22.04 /
# Debian 10-12 / RHEL 7-9 users had no parallel CPU kernels. Here we compile with CentOS 7's
# devtoolset-8 (GCC 8.3, glibc 2.17 sysroot), link its libgomp, static-link the C++ runtime,
# and let auditwheel bundle libgomp into the wheel (tag manylinux_2_17).
#
# Prerequisites (one-time, no root; see docs_cpu/WHEEL_PLATFORM_NOTES.md):
#   BNB_DTS_ROOT (default: $HOME/dts8/root) containing an extracted CentOS7 root with
#   devtoolset-8 g++ plus glibc-devel/glibc-headers/kernel-headers/libgomp/libstdc++ RPMs.
#   Required helpers: python -m build, auditwheel (pip install auditwheel).
#
# Usage: bash build_oldgomp_linux.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
ROOT=${BNB_DTS_ROOT:-$HOME/dts8/root}
DTS=$ROOT/opt/rh/devtoolset-8/root
BIN=$DTS/usr/bin
STAGE=${BNB_AVX512_SO:-/mnt/d/work/research/manylinux_so/libbitsandbytes_cpu_avx512.so}
OUT=$REPO/dist_release
export LD_LIBRARY_PATH="$DTS/usr/lib64:$DTS/usr/lib:$ROOT/usr/lib64:$ROOT/usr/lib"

[ -x "$BIN/g++" ] || { echo "missing devtoolset g++ at $BIN (set BNB_DTS_ROOT)"; exit 1; }
[ -f "$STAGE" ] || { echo "missing avx512 .so at $STAGE (set BNB_AVX512_SO)"; exit 1; }
mkdir -p "$OUT" "$ROOT/../build" 2>/dev/null || true
BUILD=${BNB_DTS_BUILD:-$HOME/dts8/build}
mkdir -p "$BUILD"

COMMON=(
  --sysroot="$ROOT"
  -B"$ROOT/usr/lib64/" -B"$DTS/usr/lib/gcc/x86_64-redhat-linux/8/"
  -O2 -std=c++17 -fno-math-errno -fopenmp -fPIC -shared
  -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0
  -static-libgcc -static-libstdc++
  -I "$REPO/csrc"
  -L"$ROOT/usr/lib64" -L"$ROOT/lib64"
  -Wl,-rpath-link,"$ROOT/usr/lib64:$ROOT/lib64" -Wl,-rpath,"$ROOT/usr/lib64"
  -Wl,--no-undefined -Wl,--export-dynamic
)

echo "=== generic (OpenMP, no AVX) ==="
"$BIN/g++" "${COMMON[@]}" -DBNB_CPU_PORTABLE \
  "$REPO/csrc/cpu_ops.cpp" "$REPO/csrc/cpu_gdn.cpp" "$REPO/csrc/pythonInterface.cpp" \
  -o "$BUILD/libbitsandbytes_cpu_generic.so" -Wl,-soname,libbitsandbytes_cpu_generic.so || exit 1

echo "=== avx2 (OpenMP) ==="
"$BIN/g++" "${COMMON[@]}" -mavx2 -mfma -mf16c -D__AVX2__ \
  "$REPO/csrc/cpu_ops.cpp" "$REPO/csrc/cpu_gdn.cpp" "$REPO/csrc/pythonInterface.cpp" \
  -o "$BUILD/libbitsandbytes_cpu.so" -Wl,-soname,libbitsandbytes_cpu.so || exit 1

for f in libbitsandbytes_cpu.so libbitsandbytes_cpu_generic.so; do
  echo -n "  $f: max GLIBC "; objdump -T "$BUILD/$f" | grep -oE 'GLIBC_[0-9.]+' | sort -u -V | tail -1
done

echo "=== stage .so files into the package ==="
cp -f "$BUILD/libbitsandbytes_cpu.so" "$REPO/bitsandbytes/libbitsandbytes_cpu.so"
cp -f "$BUILD/libbitsandbytes_cpu_generic.so" "$REPO/bitsandbytes/libbitsandbytes_cpu_generic.so"
cp -f "$STAGE" "$REPO/bitsandbytes/libbitsandbytes_cpu_avx512.so"

echo "=== build wheel + auditwheel ==="
cd "$REPO" || exit 1
rm -rf build bitsandbytes_cpu_fork.egg-info
python -m build --wheel --no-isolation --outdir "$OUT" 2>&1 | tail -2
W=$(ls -t "$OUT"/*linux_x86_64.whl 2>/dev/null | head -1)
[ -n "$W" ] || { echo NO_WHEEL; exit 1; }
rm -rf "$OUT/repaired_oldgomp"
auditwheel repair -w "$OUT/repaired_oldgomp" "$W" 2>&1 | tail -4
ls -la "$OUT/repaired_oldgomp"
