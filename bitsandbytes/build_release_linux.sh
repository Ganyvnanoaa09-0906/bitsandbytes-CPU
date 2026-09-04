#!/usr/bin/env bash
# ============================================================
# build_release_linux.sh — 一键产出 bitsandbytes CPU fork 的 Linux 预编译分发包
#
# 在 Linux（x86_64 / aarch64）上：编译 libbitsandbytes_cpu.so + 无 torch 的 C 层自检，
# 组装一个「解压即用」的发布目录，并打包 tar.gz。逻辑参照 Windows 版
# build_release_windows.ps1。
#
# 用法（在 Linux / WSL 里）：
#   bash build_release_linux.sh                 # 本机架构（x86_64 用 AVX2, aarch64 用 NEON）
#   bash build_release_linux.sh --clang         # 用 clang++ 编译
#   bash build_release_linux.sh --no-avx2       # 低配 x86 机器关闭 AVX2
#   bash build_release_linux.sh --out dist_linux # 指定输出目录
#
# 产物：
#   dist_linux/bitsandbytes-cpu-linux_<ver>/      解压即用目录
#   dist_linux/bitsandbytes-cpu-linux_<ver>.tar.gz 可直接分发
#
# 依赖：bash、g++/clang++、libomp（OpenMP）。完整安装：
#   Debian/Ubuntu: sudo apt install g++ libomp-dev
# ============================================================
set -e
cd "$(dirname "$0")"

COMPILER="g++"
NO_AVX2=0
OUT_DIR="dist_linux"
for arg in "$@"; do
  case "$arg" in
    --clang) COMPILER="clang++" ;;
    --no-avx2) NO_AVX2=1 ;;
    --out) OUT_DIR="$2"; shift ;;
  esac
done

# ---- 版本号（读 bitsandbytes/__init__.py） ----
VER=$(grep -oP '__version__\s*=\s*"\K[^"]+' bitsandbytes/__init__.py | head -1)
[ -z "$VER" ] && VER="dev"
PKG="bitsandbytes-cpu-linux_${VER}"
echo "[1/5] 版本：$VER"

# ---- 编译 .so + selftest（复用 build_linux.sh 的核心逻辑，最简：直接调用） ----
echo "[2/5] 编译 libbitsandbytes_cpu.so ..."
SELFTEST_ARG=""
if [ "$NO_AVX2" = "1" ]; then SELFTEST_ARG="$SELFTEST_ARG --no-avx2"; fi
if [ "$COMPILER" = "clang++" ]; then SELFTEST_ARG="$SELFTEST_ARG --clang"; fi
bash build_linux.sh $SELFTEST_ARG          # 编译 .so（不含 selftest 参数，避免重复）

# 明确编译 selftest（无 torch 的 C 层自检）
echo "[3/5] 编译无 torch 的 C 层自检 selftest_cpu ..."
ARCH_FLAGS=""
if [ "$(uname -m)" = "x86_64" ] && [ "$NO_AVX2" = "0" ]; then ARCH_FLAGS="-march=native -D__AVX2__"; fi
mkdir -p build_linux
$COMPILER -O2 -std=c++17 -fopenmp $ARCH_FLAGS -DNOMINMAX -DNDEBUG \
  -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc \
  selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
  -o build_linux/selftest_cpu
echo "  selftest_cpu 编译完成"

# ---- 组装发布目录 ----
echo "[4/5] 组装发布目录 ..."
STAGE="$OUT_DIR/$PKG"
rm -rf "$STAGE"
mkdir -p "$STAGE/bitsandbytes" "$STAGE/examples/cpu" "$STAGE/docs_cpu" "$STAGE/tools"

# Python 包（源码，跳过产物/缓存）
cp -r bitsandbytes/* "$STAGE/bitsandbytes/" 2>/dev/null || true
rm -f "$STAGE/bitsandbytes/libbitsandbytes_cpu.dll" "$STAGE/bitsandbytes/vcomp140.dll"
cp bitsandbytes/libbitsandbytes_cpu.so "$STAGE/bitsandbytes/"   # 编译出的 .so

# 示例
cp examples/cpu/* "$STAGE/examples/cpu/" 2>/dev/null || true

# 文档（中英）
cp docs_cpu/*.md "$STAGE/docs_cpu/" 2>/dev/null || true

# 灾难工具（源码；Linux 侧工具可另行编译；含 selftest_cpu）
cp tools/sector_carve.c tools/sector_carve_gui.c tools/sector_mirror.c tools/sector_mirror_gui.c \
   "$STAGE/tools/" 2>/dev/null || true
cp build_linux/selftest_cpu "$STAGE/tools/" 2>/dev/null || true

# README + LICENSE
cp README_EN.md "$STAGE/README.md" 2>/dev/null || cp README.md "$STAGE/" 2>/dev/null || true
cp LICENSE "$STAGE/" 2>/dev/null || true

# RELEASE_NOTES
cat > "$STAGE/RELEASE_NOTES.md" <<EOF
# bitsandbytes-cpu-linux_$VER — Prebuilt Linux release

CPU-training fork of bitsandbytes (fused GDN / gemm_8bit / 8-bit optimizer kernels) for
no-GPU Linux machines (AVX2 x86_64 or NEON aarch64). This package is prebuilt: extract
and use, no local toolchain needed (except OpenMP/runtime for the .so).

## Contents
- bitsandbytes/  full Python package + libbitsandbytes_cpu.so
- examples/cpu/  CPU training example
- docs_cpu/      technical guide / quickstart / tech report / disaster recovery (bilingual)
- tools/         disaster tools (source) + selftest_cpu (no-torch C self-check)

## Use
1. Unzip; add bitsandbytes/ to PYTHONPATH (or copy into site-packages).
2. Install PyTorch CPU (pip install torch --index-url https://download.pytorch.org/whl/cpu).
3. Verify: python -c 'import bitsandbytes as bnb; print(bnb.__version__)'
4. No-torch self-check: ./tools/selftest_cpu
EOF

echo "[5/5] 打包含 tar.gz ..."
TARGZ="$OUT_DIR/$PKG.tar.gz"
rm -f "$TARGZ"
tar -czf "$TARGZ" -C "$OUT_DIR" "$PKG"

echo "============================================================"
echo "DONE."
echo "  发布目录 : $STAGE"
echo "  分发包   : $TARGZ"
echo "  用法示例 : PYTHONPATH=\$PWD/$PKG python -c 'import bitsandbytes; print(bitsandbytes.__version__)'"
echo "============================================================"
