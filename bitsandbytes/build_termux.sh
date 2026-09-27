#!/usr/bin/env bash
# ============================================================
# build_termux.sh — bitsandbytes CPU 后端 Termux(Android/ARM64) 编译
# ------------------------------------------------------------
# 为什么需要这个脚本（而不是直接用 build_linux.sh）：
#   Termux 与普通 Linux 有四处实际差异，build_linux.sh 会在其中三处踩到：
#     1) 编译器是 clang++（来自 `pkg install clang`），且**没有 g++**
#     2) OpenMP 需要单独 `pkg install libomp`；装了才有 -fopenmp，没装会链接失败
#     3) `-march=native` 在 Termux/Android 的 clang 上行为不稳（可能生成宿主
#        不支持的指令），所以这里**不用 native**，改为探测式选择
#     4) 磁盘/内存小，不生成多余中间产物
#
# 产物：
#   bitsandbytes/libbitsandbytes_cpu.so   —— Python 侧加载（同 build_linux.sh）
#   build_termux/selftest_cpu             —— 无 torch 的 C 层自检
#
# 用法：
#   bash build_termux.sh              # 编译 + 抽查导出符号
#   bash build_termux.sh --selftest   # 额外编译并运行 C 层自检（推荐）
#   bash build_termux.sh --no-openmp  # 强制单线程（libomp 装不上时）
#
# 依赖（Termux 内）：
#   pkg update && pkg install -y clang libomp make
#   （libomp 只影响 OpenMP 并行；没有它加 --no-openmp 也能编出可用的 .so）
# ============================================================
set -e
cd "$(dirname "$0")"

COMPILER="clang++"
USE_OPENMP=1
RUN_SELFTEST=0
for arg in "$@"; do
  case "$arg" in
    --clang) COMPILER="clang++" ;;
    --gcc)   COMPILER="g++" ;;
    --no-openmp) USE_OPENMP=0 ;;
    --selftest)  RUN_SELFTEST=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg（--selftest / --no-openmp / --clang / --gcc）"; exit 2 ;;
  esac
done

echo "============================================================"
echo "bitsandbytes CPU 后端 — Termux(Android/ARM64) 构建"
echo "============================================================"

# ---------- [1/5] 环境与工具链 ----------
echo "[1/5] 检查环境与工具链 ..."
ARCH="$(uname -m)"
echo "  架构      : $ARCH"
echo "  Termux    : ${PREFIX:-<未设置>}"
if [ -n "$PREFIX" ] && [ -d "${PREFIX}/bin" ]; then
  echo "  前缀可用  : 是（Termux 环境正常）"
else
  echo "  ⚠ 未检测到 PREFIX —— 这看起来不像 Termux（普通 Linux 请用 build_linux.sh）"
  echo "    仍继续，但 -fopenmp / 头文件路径可能不同。"
fi

if ! command -v "$COMPILER" >/dev/null 2>&1; then
  # Termux 没装 clang 时给出可执行的补救命令，而不是只报一句“未找到”
  echo "ERROR: $COMPILER 不存在。"
  echo "  Termux 修复：pkg install -y clang"
  echo "  或改用系统 g++：bash build_termux.sh --gcc"
  exit 1
fi
echo "  编译器    : $($COMPILER --version | head -1)"

if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "  ⚠ 本脚本针对 Android/ARM64；当前是 $ARCH。仍按通用路径编译。"
fi

# ---------- [2/5] 探测可选能力（不用“假设”,用“试编译”） ----------
echo "[2/5] 探测编译器能力 ..."

# 探测 -mcpu=native：能过就用（Termux 的 clang 对 native 支持视版本而定），
# 不能过就退回架构默认。绝不让一个探测失败终止构建。
PROBE_DIR="$(mktemp -d 2>/dev/null || echo ./build_termux_probe)"
mkdir -p "$PROBE_DIR"
trap 'rm -rf "$PROBE_DIR"' EXIT

ARCH_FLAGS=""
try_flag() {
  local flag="$1"
  printf 'int main(void){return 0;}\n' > "$PROBE_DIR/p.c"
  if $COMPILER $flag "$PROBE_DIR/p.c" -o "$PROBE_DIR/p.out" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  if try_flag "-mcpu=native"; then
    ARCH_FLAGS="-mcpu=native"
    echo "  [NEON] 采用 -mcpu=native（探测通过）"
  else
    echo "  [NEON] -mcpu=native 不可用，使用架构默认（aarch64 自带 NEON）"
  fi
else
  # 非 ARM（例如在 Termux 里跑 x86 模拟）：沿用 build_linux.sh 的 AVX2 逻辑
  if [ "$ARCH" = "x86_64" ] && grep -q ' avx2' /proc/cpuinfo 2>/dev/null; then
    ARCH_FLAGS="-march=native -D__AVX2__"
    echo "  [AVX2] 检测到 AVX2，启用"
  else
    echo "  [scalar] 无 AVX2，编译为通用 x86-64"
  fi
fi

# 探测 OpenMP：装了 libomp 才有 -fopenmp。探测失败就自动降级为单线程，
# 而不是让链接阶段报一堆 undefined symbol。
OMP_FLAG=""
if [ "$USE_OPENMP" = "1" ]; then
  printf '#include <omp.h>\nint main(void){return omp_get_max_threads()>0?0:1;}\n' > "$PROBE_DIR/o.c"
  if $COMPILER -fopenmp "$PROBE_DIR/o.c" -o "$PROBE_DIR/o.out" >/dev/null 2>&1; then
    OMP_FLAG="-fopenmp"
    echo "  [OpenMP] 可用，启用并行"
  else
    echo "  [OpenMP] 不可用（缺 libomp）→ 自动降级单线程"
    echo "           若要并行：pkg install -y libomp"
  fi
else
  echo "  [OpenMP] 用户要求单线程（--no-openmp）"
fi

# ---------- [3/5] 编译 .so ----------
echo "[3/5] 编译 libbitsandbytes_cpu.so ..."
mkdir -p build_termux
OUT="bitsandbytes/libbitsandbytes_cpu.so"
rm -f "$OUT"

# shellcheck disable=SC2086  # 这些变量就是要按词展开的编译开关
$COMPILER -O2 -std=c++17 $OMP_FLAG $ARCH_FLAGS \
  -fPIC -shared -DNOMINMAX -DNDEBUG \
  -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
  -I csrc \
  csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
  -o "$OUT" \
  -Wl,--no-undefined -Wl,-soname,libbitsandbytes_cpu.so

echo "  -> $OUT ($(du -h "$OUT" 2>/dev/null | cut -f1))"

# ---------- [4/5] 抽查导出符号 ----------
# 只报“编译成功”是不够的：必须确认内核符号真的被导出了。
# 用固定清单 + 逐个断言，而不是数一个总数（总数对得上也可能缺关键项）。
echo "[4/5] 导出符号抽查 ..."
if command -v nm >/dev/null 2>&1; then
  MISSING=0
  for sym in cquantize_blockwise_cpu_fp32 cdequantize_blockwise_cpu_fp32 \
             cgemm_8bit_inference_cpu_fp32 cgemv_4bit_inference_cpu_fp32 \
             coptimizer_update_8bit_blockwise_cpu; do
    if nm -D "$OUT" 2>/dev/null | grep -q " $sym\$"; then
      echo "  [OK]   $sym"
    else
      echo "  [MISS] $sym"
      MISSING=$((MISSING + 1))
    fi
  done
  # 统计全部 CPU 符号（参考值，不作判据）
  TOTAL=$(nm -D "$OUT" 2>/dev/null | grep -cE "c(quantize|dequantize|gemm|optimizer|gemv)" || true)
  echo "  CPU 相关符号总数: $TOTAL"
  if [ "$MISSING" != "0" ]; then
    echo "ERROR: 有 $MISSING 个关键符号未导出 —— .so 不可用。"
    exit 1
  fi
else
  echo "  （nm 不可用，跳过符号检查；不影响 .so 生成）"
fi

# ---------- [5/5] C 层自检 ----------
# 自检共 4 项：quantize 往返 / gemm_8bit / gemv_4bit / AdamW8bit。
# （旧文档曾写“5 项含 gdn_fwd”，实测代码里只有 4 项 —— 以 selftest_cpu.c 为准。）
echo "[5/5] C 层自检 ..."
if [ "$RUN_SELFTEST" = "1" ]; then
  echo "  编译无 torch 自检 ..."
  # -x c++ 是必需的，不是装饰：
  #   文件名叫 selftest_cpu.c，但它内部有
  #       #ifdef __cplusplus
  #       extern "C" {
  #       #endif
  #   把内核入口声明成 C 链接。**只有按 C++ 编译**才定义 __cplusplus、才会走那个分支；
  #   若按 C 编译，块被跳过 ⇒ 符号按 C++ 修饰 ⇒ 每一个内核符号都链接失败。
  #   （build_manual/selftest_win.bat 里的 /TP 就是同一件事的 MSVC 写法，其注释已写明。）
  # 显式写出 -x c++ 而不是依赖"clang++ 见到 .c 也当 C++"，是为了消掉
  #   warning: treating 'c' input as 'c++' when in C++ mode, this behavior is deprecated
  # 那条警告 —— 它是靠扩展名猜语言，将来会变成硬错误。MSVC 侧靠 /TP 显式指定，
  # 这里就靠 -x c++ 显式指定，两边对齐。
  # shellcheck disable=SC2086
  $COMPILER -O2 -std=c++17 $OMP_FLAG $ARCH_FLAGS \
    -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
    -I csrc -x c++ selftest_cpu.c -x none \
    csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
    -o build_termux/selftest_cpu
  echo "  --- 运行结果 ---"
  ./build_termux/selftest_cpu
  echo "  --- 自检结束（上方 [PASSED] 即通过）---"
else
  echo "  已跳过（未加 --selftest）"
  echo "  建议：bash build_termux.sh --selftest"
fi

echo "============================================================"
echo "DONE."
echo "  .so  : $(pwd)/$OUT"
echo "  架构 : $ARCH   OpenMP: ${OMP_FLAG:-<关闭>}   额外标志: ${ARCH_FLAGS:-<无>}"
echo "  自检 : bash build_termux.sh --selftest"
echo "  说明 : Termux 一般装不上 torch，所以验证到【C 层自检】为止"
echo "         （验证内核数值正确性，不是训练）。"
echo "============================================================"
