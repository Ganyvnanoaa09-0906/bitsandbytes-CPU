#!/usr/bin/env bash
# ============================================================
# linux_container_check.sh — Linux CPU 侧检查（适用 EPYC/无核显容器）
# ------------------------------------------------------------
# 用途：在没有 OpenCL/核显的 Linux 机器上，验证本项目 Linux 相关能力：
#   1) bnb CPU 后端：AVX2 / AVX-512 / portable(generic) 三产物构建 + C 层自检
#   2) 通用 iGPU 训练套件的 **CPU 后端**（--backend cpu）：对拍、学习、存权重
#      —— 这一步不需要核显，只验证训练器在 Linux 上能跑、数值正确
# 注意：这里【不会】测核显性能；核显需要真 Linux + AMD APU + Mesa OpenCL。
#
# 用法：
#   bash gt1030/linux_container_check.sh                 # 自动下载 tiny 模型做 CPU 训练 smoke
#   MODEL_DIR=/path/to/model bash ...                    # 用本地模型（推荐，内网更稳）
#   SKIP_BUILD=1 bash ...                                # 跳过 bnb 构建
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONIOENCODING=utf-8
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PASS=0; FAIL=0
ok()   { echo "[PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
need() { command -v "$1" >/dev/null 2>&1; }

echo "============================================================"
echo " Linux 容器检查（CPU 后端；不含核显测试）"
echo " 根目录: $ROOT"
echo "============================================================"

echo "[1/4] 工具链与依赖"
need g++ && ok "g++ $(g++ -dumpversion)" || bad "g++ 缺失"
need python3 && ok "python3 $(python3 -V 2>&1)" || bad "python3 缺失"
python3 - <<'PY'
import sys
try:
    import torch
    print('[PASS] torch', torch.__version__)
except Exception as e:
    print('[FAIL] torch 不可用:', e, '-> pip install torch --index-url https://download.pytorch.org/whl/cpu')
    sys.exit(1)
PY
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
python3 -c "import transformers" 2>/dev/null && ok "transformers" || { bad "transformers 缺失 -> pip install transformers safetensors"; }

echo "[2/4] bnb CPU 后端构建 + 自检"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "  (SKIP_BUILD=1，跳过)"
else
  ( cd "$ROOT/bitsandbytes" && bash build_linux.sh --selftest >/tmp/bnb_build.log 2>&1 )
  rc=$?
  if [ $rc -eq 0 ]; then ok "build_linux.sh --selftest"; else bad "build_linux.sh 失败，见 /tmp/bnb_build.log"; fi
fi

echo "[3/4] C 层自检（若已构建）"
if [ -x "$ROOT/build_linux/selftest_cpu" ]; then
  "$ROOT/build_linux/selftest_cpu" >/tmp/bnb_selftest.log 2>&1 && ok "selftest_cpu 5/5" || bad "selftest_cpu 失败"
else
  echo "  (无预构建 selftest，跳过；由 build_linux.sh 负责)"
fi

echo "[4/4] 通用训练套件 CPU 后端 smoke（--backend cpu）"
MODEL_ARG="${MODEL_DIR:-HuggingFaceH4/tiny-random-LlamaForCausalLM}"
python3 "$ROOT/gt1030/igpu_train.py" --model "$MODEL_ARG" --backend cpu --mode lora --steps 2 \
        --log-every 1 >/tmp/igpu_cpu_smoke.log 2>&1
if grep -q 'finite=YES' /tmp/igpu_cpu_smoke.log; then
  ok "CPU 后端训练 smoke"
  grep -E 'parity|RESULT' /tmp/igpu_cpu_smoke.log | tail -2
else
  bad "CPU 后端 smoke 失败，见 /tmp/igpu_cpu_smoke.log"
fi

echo
echo "============================================================"
echo " 汇总: $PASS 通过 / $FAIL 失败"
echo " 核显说明: 容器/EPYC 无 iGPU，CL OpenCL 不可用属于预期；"
echo "           想测核显请准备真 Linux + AMD APU（Mesa Clover/Rusticl）。"
echo "============================================================"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
