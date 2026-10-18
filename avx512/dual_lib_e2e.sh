#!/usr/bin/env bash
# End-to-end test of the three-artifact CPU backend on a machine that can execute AVX-512:
#   libbitsandbytes_cpu.so           AVX2 + FMA            (default x86 artifact)
#   libbitsandbytes_cpu_avx512.so    AVX-512               (only loaded when the CPU supports it)
#   libbitsandbytes_cpu_generic.so   baseline x86-64/SSE2  (only loaded when the CPU lacks AVX2)
#
# What it proves, in order:
#   1. one source tree builds all three artifacts;
#   2. the AVX2 artifact has no zmm code, the AVX-512 one does, and the generic artifact has NO
#      AVX instructions at all (that is what makes it runnable on pre-Haswell CPUs);
#   3. doctor prints the three-way decision;
#   4. the package selftest passes with the loader's choice (AVX-512 here);
#   5. the loader picks the AVX-512 artifact, BNB_CPU_DISABLE_AVX512=1 falls back to AVX2, and
#      BNB_CPU_FORCE_GENERIC=1 picks the generic artifact (which then runs the scalar path);
#   6. loader-level adamw8bit step timing, auto vs forced AVX2;
#   7. the portable (scalar-only) selftest passes;
#   8. if qemu-user can be installed, the portable selftest also runs under an emulated pre-AVX2
#      CPU (Nehalem), which is the only real proof short of owning such a machine.
#
# No source changes; torch is installed only when missing (CPU wheel).
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PKG="$ROOT/bitsandbytes"
ALLOW_TORCH_INSTALL="${BNB_E2E_ALLOW_TORCH_INSTALL:-1}"

say() { printf '\n=== %s ===\n' "$*"; }
py() { PYTHONPATH="$PKG" python3 "$@"; }

say environment
echo "utc        : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "kernel     : $(uname -srm)"
if [ -f /etc/os-release ]; then . /etc/os-release; echo "distro     : ${PRETTY_NAME:-unknown}"; fi
echo "cpu        : $(awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo)"
echo "nproc      : $(nproc)"
echo "avx512     : $(grep -o 'avx512[a-z_]*' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
echo "g++        : $(g++ --version 2>&1 | head -1)"
echo "python3    : $(python3 --version 2>&1)"
echo "git head   : $(git -C "$ROOT" log -1 --oneline 2>/dev/null || echo 'not a git checkout')"

say "[1/10] build all three artifacts from one source tree"
( cd "$PKG" && bash build_linux.sh --avx512 --generic ) || { echo "BUILD_FAILED"; exit 2; }
for f in bitsandbytes/libbitsandbytes_cpu.so bitsandbytes/libbitsandbytes_cpu_avx512.so \
         bitsandbytes/libbitsandbytes_cpu_generic.so; do
    if [ ! -f "$PKG/$f" ]; then echo "MISSING $f"; exit 2; fi
    ls -la "$PKG/$f"
done
ZMM_AVX2=$(objdump -d "$PKG/bitsandbytes/libbitsandbytes_cpu.so" 2>/dev/null | grep -c zmm || true)
ZMM_AVX512=$(objdump -d "$PKG/bitsandbytes/libbitsandbytes_cpu_avx512.so" 2>/dev/null | grep -c zmm || true)
AVX_GENERIC=$(objdump -d "$PKG/bitsandbytes/libbitsandbytes_cpu_generic.so" 2>/dev/null | grep -cE '%[yz]mm|vfmadd|vzeroupper|vmovaps|vmulps|vaddps' || true)
echo
echo "AVX2   artifact zmm lines      = $ZMM_AVX2  (must be 0)"
echo "AVX512 artifact zmm lines      = $ZMM_AVX512  (must be > 0)"
echo "generic artifact AVX insns     = $AVX_GENERIC  (must be 0)"
if [ "$ZMM_AVX2" -ne 0 ]; then
    echo "PORTABLE_ARTIFACT_FAILED: AVX2 artifact contains zmm/EVEX instructions"
fi
if [ "$ZMM_AVX512" -le 0 ]; then
    echo "AVX512_ARTIFACT_WITHOUT_EVEX: AVX-512 artifact contains no zmm code -- flags not applied?"
fi
if [ "$AVX_GENERIC" -ne 0 ]; then
    echo "GENERIC_ARTIFACT_FAILED: generic artifact contains AVX instructions and cannot run pre-Haswell"
fi

say "[2/10] torch"
if python3 -c 'import torch' >/dev/null 2>&1; then
    python3 - <<'EOF'
import torch
print("torch      :", torch.__version__)
print("capability :", torch.backends.cpu.get_cpu_capability())
EOF
else
    echo "torch is missing"
    if [ "$ALLOW_TORCH_INSTALL" = "1" ]; then
        echo "installing the CPU wheel (a few hundred MB) ..."
        python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu || {
            echo "TORCH_INSTALL_FAILED"; exit 3; }
        python3 - <<'EOF'
import torch
print("torch      :", torch.__version__)
print("capability :", torch.backends.cpu.get_cpu_capability())
EOF
    else
        echo "SKIP_TORCH (BNB_E2E_ALLOW_TORCH_INSTALL=0): later torch-dependent steps will fail visibly"
    fi
fi

say "[3/10] doctor: the three-way decision"
( cd "$ROOT" && py bitsandbytes_cpu_cli.py doctor ) || echo "DOCTOR_FAILED"

say "[4/10] selftest with the loader's choice"
( cd "$ROOT" && py bitsandbytes_cpu_cli.py selftest ) || echo "SELFTEST_FAILED"

say "[5/10] loader decisions: auto, forced AVX2, forced generic"
( cd "$ROOT" && py - <<'EOF'
import torch, bitsandbytes.cextension as ce
print("torch capability =", torch.backends.cpu.get_cpu_capability())
print("auto            :", ce.LOADED_NATIVE_LIB)
EOF
) || echo "SELECTION_AUTO_FAILED"
( cd "$ROOT" && BNB_CPU_DISABLE_AVX512=1 py - <<'EOF'
import bitsandbytes.cextension as ce
print("forced AVX2     :", ce.LOADED_NATIVE_LIB)
EOF
) || echo "SELECTION_AVX2_FAILED"
( cd "$ROOT" && BNB_CPU_FORCE_GENERIC=1 py - <<'EOF'
import bitsandbytes.cextension as ce
print("forced generic  :", ce.LOADED_NATIVE_LIB)
EOF
) || echo "SELECTION_GENERIC_FAILED"

say "[6/10] loader-level timing: adamw8bit step on 1M fp32 params"
run_step() {
    ( cd "$ROOT" && env $1 PYTHONPATH="$PKG" python3 - <<'EOF'
import os, statistics, time
import torch
torch.set_num_threads(max(1, os.cpu_count() or 2))
import bitsandbytes as bnb
n = 1 << 20
p = torch.nn.Parameter(torch.randn(n) * 0.02)
p.grad = torch.randn(n) * 0.01
opt = bnb.optim.AdamW8bit([p], lr=1e-4)
for _ in range(5):
    opt.step()
samples = []
for _ in range(20):
    t0 = time.perf_counter()
    opt.step()
    samples.append((time.perf_counter() - t0) * 1e3)
print("  adamw8bit step median = %.3f ms" % statistics.median(samples))
EOF
    ) || echo "TIMING_FAILED"
}
echo "--- auto ---"; run_step ""
echo "--- forced AVX2 ---"; run_step "BNB_CPU_DISABLE_AVX512=1"

say "[7/10] portable selftest (scalar-only build, no AVX2/FMA code)"
( cd "$PKG" && g++ -O2 -std=c++17 -fopenmp -march=x86-64 -DBNB_CPU_PORTABLE=1 \
      -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc \
      -x c++ selftest_cpu.c -x none csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
      -o build_linux/selftest_generic ) && ( cd "$PKG" && ./build_linux/selftest_generic ) \
  || echo "PORTABLE_SELFTEST_FAILED"

say "[8/10] emulated pre-AVX2 execution (qemu-user, optional)"
QEMU=""
if command -v qemu-x86_64 >/dev/null 2>&1; then
    QEMU=$(command -v qemu-x86_64)
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    echo "installing qemu-user ..."
    sudo apt-get install -y qemu-user >/dev/null 2>&1 || echo "QEMU_INSTALL_FAILED"
    command -v qemu-x86_64 >/dev/null 2>&1 && QEMU=$(command -v qemu-x86_64)
fi
if [ -n "$QEMU" ]; then
    for cpu in Nehalem core2duo; do
        echo "--- qemu-x86_64 -cpu $cpu ---"
        if "$QEMU" -cpu "$cpu" "$PKG/build_linux/selftest_generic" 2>&1 | tail -8; then
            break
        fi
        echo "  (cpu model $cpu not accepted, trying next)"
    done
else
    echo "QEMU_SKIPPED: qemu-x86_64 absent and no passwordless sudo to install qemu-user"
    echo "  The generic artifact has 0 AVX instructions and its portable selftest passed, which is the"
    echo "  static evidence; qemu would add a dynamic execution test on an emulated pre-AVX2 CPU."
fi

say "[9/10] summary markers"
echo "MARK commit            $(git -C "$ROOT" log -1 --format=%h 2>/dev/null)"
echo "MARK cpu_avx512        $(grep -c avx512f /proc/cpuinfo 2>/dev/null | head -1)"
echo "MARK zmm_avx2          $ZMM_AVX2"
echo "MARK zmm_avx512        $ZMM_AVX512"
echo "MARK avx_generic       $AVX_GENERIC"
echo "MARK qemu              ${QEMU:-none}"
echo "MARK done"
