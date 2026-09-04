#!/usr/bin/env bash
# ============================================================
# wsl_verify.sh — WSL 里一键验证 bitsandbytes CPU 后端 (Linux)
# ------------------------------------------------------------
# 重启进入 WSL 后执行本脚本，验证「Linux 支持」全链路：
#   1. 编译 libbitsandbytes_cpu.so（build_linux.sh）
#   2. 无 torch 的 C 层自检 selftest_cpu（量化/GEMM/优化器/4bitGEMV）
#   3. 安装 torch（Linux x86_64 官方 CPU wheel）
#   4. 带 torch 的完整 selftest（py -3.11 -m bitsandbytes.gdn_cpu）
#   5. 低内存训练冒烟（torch_cpu_kit + 8bit 优化器）
#
# 用法（在仓库根目录执行，无需写死绝对路径）:
#   bash wsl_verify.sh            # 全链路
#   bash wsl_verify.sh --no-torch # 跳过 torch 安装（fast，只到 C 层自检）
#   bash wsl_verify.sh --no-build # 跳过编译（用已存在的 .so）
# ============================================================
set -e
cd "$(dirname "$0")"

NO_TORCH=0
NO_BUILD=0
for a in "$@"; do [ "$a" = "--no-torch" ] && NO_TORCH=1; [ "$a" = "--no-build" ] && NO_BUILD=1; done

echo "=== [WSL 验证] 环境 ==="
uname -a
echo "  python: $(python3 --version 2>/dev/null || echo 无)"

if [ "$NO_BUILD" = "0" ]; then
  echo ""
  echo "=== Step 1/5: build_linux.sh 编译 .so ==="
  bash build_linux.sh
fi

echo ""
echo "=== Step 2/5: C 层自检（无 torch，验证内核数值） ==="
# 直接编译 selftest_cpu（复用 build_linux.sh 的编译器逻辑）
g++ -O2 -std=c++17 -fopenmp -march=native -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \
  -I csrc selftest_cpu.c csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
  -o build_linux/selftest_cpu
./build_linux/selftest_cpu
echo "  [WSL] C 层自检完成"

if [ "$NO_TORCH" = "0" ]; then
  echo ""
  echo "=== Step 3/5: 安装 torch (Linux CPU wheel) ==="
  # 不全局装，避免污染；用 venv
  python3 -m venv build_linux/venv 2>/dev/null || true
  . build_linux/venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet diffusers peft transformers tqdm psutil 2>/dev/null || true
  echo "  torch: $(python -c 'import torch;print(torch.__version__)')"

  echo ""
  echo "=== Step 4/5: 带 torch 完整 selftest ==="
  PYTHONPATH="$PWD" python -m bitsandbytes.gdn_cpu 2>&1 | tail -5
  echo "  [WSL] gdn_cpu selftest 完成"

  echo ""
  echo "=== Step 5/5: 8bit 优化器 + 量化冒烟 ==="
  PYTHONPATH="$PWD" python - <<'PY'
import torch
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
print("  bnb:", bnb.__version__ if hasattr(bnb,'__version__') else '?', "DLL:", bnb.gdn_cpu.load_native() is not None)
# 8bit 量化往返
code = torch.arange(256, dtype=torch.float32) * (2.0/255.0) - 1.0
w = torch.randn(320, 320) * 0.2
q, st = quantize_blockwise(w.reshape(-1), code=code, blocksize=256)
wd = dequantize_blockwise(q, st)
err = (wd.reshape(320,320) - w).abs().max().item()
print(f"  量化往返误差: {err:.4e}  {'OK' if err < 0.01 else 'FAIL'}")
# 8bit 优化器
opt = bnb.optim.AdamW8bit([torch.nn.Parameter(torch.randn(64,64)*0.1)], lr=1e-4)
p = next(iter(opt.param_groups[0]['params']))
p.sum().backward(); opt.step()
print("  AdamW8bit step OK")
print("  [WSL] 8bit 优化器/量化冒烟完成")
PY
else
  echo ""
  echo "=== Step 3-5 跳过（--no-torch）：仅验证到 C 层自检 ==="
fi

echo ""
echo "============================================================"
echo "WSL 验证完成。若以上全部 OK，则 Linux 支持（x86_64/AVX2）已验证。"
echo "============================================================"
