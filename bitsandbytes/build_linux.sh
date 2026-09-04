#!/usr/bin/env bash
# ============================================================
# build_linux.sh — bitsandbytes CPU 后端 Linux 编译（gcc/clang）
# ------------------------------------------------------------
# 在 Linux（x86_64 / aarch64）上编译 libbitsandbytes_cpu.so。
# 这是 bitsandbytes 的「原生平台」构建入口：
#   - x86_64 : 编译时带 AVX2（oneAPI/MKL 之外的我们自己的 AVX2 内核）
#   - aarch64: 走上游自带 NEON 路径（无需 -march，NEON 是 ARM64 基础）
#
# 产物：
#   bitsandbytes/libbitsandbytes_cpu.so      —— Python 侧加载
#   selftest_cpu                             —— 无 torch 的 C 层自检（Linux/ARM 通用）
#
# 用法：
#   bash build_linux.sh               # 本机（x86_64 用 AVX2；aarch64 用 NEON）
#   bash build_linux.sh --clang       # 强制 clang++（默认 g++）
#   bash build_linux.sh --no-avx2     # 强制关闭 AVX2（低配 x86 / 无 AVX2 机器）
#   bash build_linux.sh --selftest    # 额外编译并运行 selftest_cpu（无 torch）
#
# 依赖：g++ 或 clang++、libomp（OpenMP）、gcc-c++（或 clang）
#   （Debian/Ubuntu: sudo apt install g++ libomp-dev;  aarch64: 同上）
# ============================================================
set -e
cd "$(dirname "$0")"

COMPILER="g++"
NO_AVX2=0
RUN_SELFTEST=0
for arg in "$@"; do
  case "$arg" in
    --clang) COMPILER="clang++" ;;
    --no-avx2) NO_AVX2=1 ;;
    --selftest) RUN_SELFTEST=1 ;;
  esac
done

echo "[1/4] 检查工具链 ..."
command -v "$COMPILER" >/dev/null || { echo "ERROR: $COMPILER 未安装（g++/clang++）"; exit 1; }
echo "  $($COMPILER --version | head -1)"

# 检测 CPU 架构
ARCH="$(uname -m)"    # x86_64 | aarch64
echo "  架构: $ARCH"

echo "[2/4] 编译 libbitsandbytes_cpu.so ..."
mkdir -p build_linux
OUT="bitsandbytes/libbitsandbytes_cpu.so"
rm -f "$OUT"

# 架构特定开关
ARCH_FLAGS=""
if [ "$ARCH" = "x86_64" ] && [ "$NO_AVX2" = "0" ]; then
  # 自动检测本机是否支持 AVX2（避免硬编码 -D__AVX2__ 在无 AVX2 机器上）
  if grep -q ' avx2' /proc/cpuinfo 2>/dev/null || \
     (command -v lscpu >/dev/null && lscpu 2>/dev/null | grep -q 'avx2'); then
    ARCH_FLAGS="-march=native -D__AVX2__"
    echo "  [AVX2] 检测到本机 AVX2，启用"
  else
    # 非 AVX2 CPU：不定义 __AVX2__、用通用 x86-64，运行时走 scalar（能跑，慢）
    ARCH_FLAGS="-march=x86-64"
    echo "  [scalar] 未检测到 AVX2，编译为通用 x86-64（实验性：无 AVX2 设备测试，"
    echo "           理论能跑但不保证性能；若崩请设 BNB_CPU_NO_AVX2=1 或 --no-avx2）"
  fi
else
  echo "  [NEON/scalar] 使用架构默认路径（aarch64 自动 NEON；或 --no-avx2）"
fi

# 编译（g++ 或 clang++，两者 OpenMP 均受支持）
$COMPILER -O2 -std=c++17 -fopenmp $ARCH_FLAGS \
  -fPIC -shared -DNOMINMAX -DNDEBUG \
  -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
  -I csrc \
  csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
  -o "$OUT" \
  -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu.so

echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"

echo "[3/4] 导出符号抽查 ..."
if command -v nm >/dev/null 2>&1; then
  nm -D "$OUT" 2>/dev/null | grep -cE "c(quantize|dequantize|gemm|optimizer|gemv)" | \
    xargs -I{} echo "  CPU 符号数: {}"
else
  echo "  （nm 不可用，跳过符号检查）"
fi

echo "[4/4] selftest ..."
if [ "$RUN_SELFTEST" = "1" ]; then
  echo "  编译并运行无 torch 的 C 层自检 ..."
  $COMPILER -O2 -std=c++17 -fopenmp $ARCH_FLAGS \
    -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
    -I csrc selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
    -o build_linux/selftest_cpu
  echo "  --- 运行结果 ---"
  ./build_linux/selftest_cpu
fi

echo "============================================================"
echo "DONE."
echo "  Linux 使用：PYTHONPATH=\$PWD python -c 'import bitsandbytes; bitsandbytes.gdn_cpu.load_native()'"
echo "  完整 selftest（需 torch）：PYTHONPATH=\$PWD python -m bitsandbytes.gdn_cpu"
echo "============================================================"
