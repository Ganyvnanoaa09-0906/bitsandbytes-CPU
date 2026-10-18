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
BUILD_AVX512=0
BUILD_GENERIC=0
for arg in "$@"; do
  case "$arg" in
    --clang) COMPILER="clang++" ;;
    --no-avx2) NO_AVX2=1 ;;
    --selftest) RUN_SELFTEST=1 ;;
    --avx512) BUILD_AVX512=1 ;;
    --generic) BUILD_GENERIC=1 ;;
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
    # ★ 这里必须是【显式 ISA】而不是 -march=native：
    #   构建机若本身支持 AVX-512（如 EPYC 9Y24 容器），-march=native 会让这份
    #   "AVX2 基线"产物里出现 zmm/EVEX 指令，它在 AVX2 机器上一调用就 SIGILL。
    #   实测（2026-10-18，AVX-512 容器）：-march=native 版 AVX2 产物有 1204 条 zmm。
    #   所以代码生成只用显式 -mavx2 -mfma -mf16c；下面的 /proc/cpuinfo 检测只用来
    #   决定"要不要开 AVX2"，不参与指令集选择。
    ARCH_FLAGS="-mavx2 -mfma -mf16c -D__AVX2__"
    echo "  [AVX2] 检测到本机 AVX2，启用"
  else
    # 非 AVX2 CPU：本该走通用 x86-64，但当前 TU 里的 intrinsics 用 __GNUC__&&x86 /
    # GDN_X86 守卫而不是 __AVX2__，-march=x86-64 编译会报 target specific option
    # mismatch ⇒ 这条路径【当前不可用】（要修见 TECHNICAL_GUIDE §6.6）。
    ARCH_FLAGS="-march=x86-64"
    echo "  [scalar] 未检测到 AVX2；注意：通用 x86-64 构建目前不受支持（intrinsics 未隔离，"
    echo "           编译会失败）。x86 硬下限是 AVX2；非 AVX2 需要单独的 generic 产物。"
  fi
else
  echo "  [NEON/scalar] 使用架构默认路径（aarch64 自动 NEON；或 --no-avx2）"
fi

# 编译（g++ 或 clang++，两者 OpenMP 均受支持）
#
# -fno-math-errno 不只是为了快，是为了可移植：
#   没有它时 gcc 会调用 libm 的 sqrtf，而 glibc 2.43 给这个符号加了新版本，
#   于是产出的 .so 强制要求 GLIBC_2.43，在更老的发行版上直接加载失败。
#   加上它之后 sqrt 变成 sqrtss 指令，这个要求就消失了。
#   实测：门槛从 GLIBC_2.43 降到 GLIBC_2.32。
$COMPILER -O2 -std=c++17 -fno-math-errno -fopenmp $ARCH_FLAGS \
  -fPIC -shared -DNOMINMAX -DNDEBUG \
  -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
  -I csrc \
  csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
  -o "$OUT" \
  -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu.so

echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"

# 可移植性断言：AVX2 产物里不允许出现 EVEX(zmm) 指令 —— 一旦出现，这份 .so 在
# AVX2 机器上会 SIGILL，而它正是要发给 AVX2 用户的那份。
if command -v objdump >/dev/null 2>&1; then
  ZMM_LINES=$(objdump -d "$OUT" 2>/dev/null | grep -c zmm || true)
  echo "  可移植性检查：zmm 指令 $ZMM_LINES 条（AVX2 产物必须为 0）"
  if [ "$ZMM_LINES" -ne 0 ]; then
    echo "ERROR: AVX2 产物包含 zmm 指令，它不能在 AVX2 机器上运行。"
    echo "       检查编译 flags：代码生成必须是显式 -mavx2 -mfma -mf16c，不能用 -march=native。"
    exit 3
  fi
fi

# 可选的 AVX-512 变体：同一份源码、另加 AVX-512 flags。它【不能】发给 AVX2 机器
# （整 TU 的 -mavx512f 会把普通内核自动向量化成 zmm，AVX2 上一调用就 SIGILL），
# 载入端按 CPUID 选择，见 bitsandbytes/_cpu_caps.py。
if [ "$BUILD_AVX512" = "1" ]; then
  if [ "$ARCH" != "x86_64" ]; then
    echo "  [AVX512] 跳过：架构 $ARCH 不是 x86_64"
  else
    echo "[2b/4] 编译 AVX-512 版（libbitsandbytes_cpu_avx512.so）..."
    OUT512="bitsandbytes/libbitsandbytes_cpu_avx512.so"
    rm -f "$OUT512"
    AVX512_FLAGS="-mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl"
    if echo 'int main(){return 0;}' | "$COMPILER" -mavx512bf16 -x c++ - -o /dev/null 2>/dev/null; then
      AVX512_FLAGS="$AVX512_FLAGS -mavx512bf16"
    fi
    "$COMPILER" -O2 -std=c++17 -fno-math-errno -fopenmp $AVX512_FLAGS \
      -fPIC -shared -DNOMINMAX -DNDEBUG \
      -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
      -I csrc \
      csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
      -o "$OUT512" \
      -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu_avx512.so
    echo "  -> $OUT512 ($(du -h "$OUT512" | cut -f1))"
    echo "  注意：这个 .so 只能在支持 AVX-512 的机器上加载；选择逻辑在 bitsandbytes/_cpu_caps.py"
  fi
fi

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
  # -x c++ 是必需的：selftest_cpu.c 里有 `#ifdef __cplusplus / extern "C" {`，
  # 只有按 C++ 编译才会走到那个分支；按 C 编译会跳过它，符号被 C++ 修饰，
  # 于是每个内核符号都链接失败。MSVC 侧用 /TP 表达同一件事
  # （见 build_manual/selftest_win.bat 的注释）。显式写出 -x c++ 而不是靠
  # clang++ "见到 .c 也当 C++" 的隐式行为，是为了消掉那条 deprecated 警告。
  $COMPILER -O2 -std=c++17 -fopenmp $ARCH_FLAGS \
    -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
    -I csrc -x c++ selftest_cpu.c -x none \
    csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
    -o build_linux/selftest_cpu
  echo "  --- 运行结果 ---"
  ./build_linux/selftest_cpu
fi

# 可选的通用 x86-64 变体：编译期去掉全部 AVX2/FMA（BNB_CPU_PORTABLE），只保留 SSE2 基线，
# 可在 2013 年之前的 x86（Haswell 之前）上运行；载入端按 CPUID 选择，见 _cpu_caps.py。
if [ "$BUILD_GENERIC" = "1" ]; then
  if [ "$ARCH" != "x86_64" ]; then
    echo "  [GENERIC] 跳过：架构 $ARCH 不是 x86_64"
  else
    echo "[2c/4] 编译通用 x86-64 版（libbitsandbytes_cpu_generic.so，无 AVX2/FMA）..."
    OUTGEN="bitsandbytes/libbitsandbytes_cpu_generic.so"
    rm -f "$OUTGEN"
    "$COMPILER" -O2 -std=c++17 -fno-math-errno -fopenmp -march=x86-64 -DBNB_CPU_PORTABLE=1 \
      -fPIC -shared -DNOMINMAX -DNDEBUG \
      -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
      -I csrc \
      csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
      -o "$OUTGEN" \
      -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu_generic.so
    echo "  -> $OUTGEN ($(du -h "$OUTGEN" | cut -f1))"
    if command -v objdump >/dev/null 2>&1; then
      AVX_INSNS=$(objdump -d "$OUTGEN" 2>/dev/null | grep -cE '%[yz]mm|vfmadd|vzeroupper|vmovaps|vmulps|vaddps' || true)
      echo "  可移植性检查：AVX 指令 $AVX_INSNS 条（generic 产物必须为 0）"
      if [ "$AVX_INSNS" -ne 0 ]; then
        echo "ERROR: generic 产物包含 AVX 指令，它不能在非 AVX2 机器上运行。"
        echo "       检查是否有 AVX2 代码没被 BNB_CPU_PORTABLE 排除。"
        exit 4
      fi
    fi
  fi
fi

echo "============================================================"
echo "DONE."
echo "  Linux 使用：PYTHONPATH=\$PWD python -c 'import bitsandbytes; bitsandbytes.gdn_cpu.load_native()'"
echo "  完整 selftest（需 torch）：PYTHONPATH=\$PWD python -m bitsandbytes.gdn_cpu"
echo "============================================================"
