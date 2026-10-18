#!/usr/bin/env bash
# End-to-end test of the AVX-512 dual library on a machine that can execute AVX-512.
#
# What it proves, in order:
#   1. the same source tree builds BOTH artifacts (AVX2 + AVX-512) on an AVX-512 host;
#   2. the AVX2 artifact really contains no EVEX (zmm) instructions while the AVX-512 one does;
#   3. `bitsandbytes-cpu doctor` reports the capability and the library it will load;
#   4. on this CPU the loader actually picks libbitsandbytes_cpu_avx512.so;
#   5. BNB_CPU_DISABLE_AVX512=1 forces the AVX2 artifact back (the switch works on real hardware);
#   6. the package still passes its own selftest with the AVX-512 library loaded;
#   7. a loader-level timing of one 8-bit optimizer step, auto vs forced AVX2.
#
# No source changes; torch is installed only if it is missing (CPU wheel).
# Run from the repo root:  bash avx512/dual_lib_e2e.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PKG="$ROOT/bitsandbytes"
ALLOW_TORCH_INSTALL="${BNB_E2E_ALLOW_TORCH_INSTALL:-1}"

say() { printf '\n=== %s ===\n' "$*"; }

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

say "[1/7] build both libraries from one source tree"
( cd "$PKG" && bash build_linux.sh --avx512 ) || { echo "BUILD_FAILED"; exit 2; }
for f in bitsandbytes/libbitsandbytes_cpu.so bitsandbytes/libbitsandbytes_cpu_avx512.so; do
    if [ ! -f "$PKG/$f" ]; then echo "MISSING $f"; exit 2; fi
    ls -la "$PKG/$f"
    echo "    zmm instruction lines: $(objdump -d "$PKG/$f" 2>/dev/null | grep -c zmm || true)"
done

say "[2/7] torch"
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
        echo "SKIP_TORCH (BNB_E2E_ALLOW_TORCH_INSTALL=0): steps 3-7 that need torch still run; only bitsandbytes imports will fail"
    fi
fi

say "[3/7] doctor: the decision, printed before any kernel is executed"
( cd "$ROOT" && PYTHONPATH="$PKG" python3 bitsandbytes_cpu_cli.py doctor ) || echo "DOCTOR_FAILED"

say "[4/7] selftest with the loader's choice"
( cd "$ROOT" && PYTHONPATH="$PKG" python3 bitsandbytes_cpu_cli.py selftest ) || echo "SELFTEST_FAILED"

say "[5/7] which library the loader picked (auto, then forced AVX2)"
( cd "$ROOT" && PYTHONPATH="$PKG" python3 - <<'EOF'
import ctypes
import torch
import bitsandbytes
import bitsandbytes.cextension as ce
chosen = ce.LOADED_NATIVE_LIB
print("torch capability =", torch.backends.cpu.get_cpu_capability())
print("LOADED_NATIVE_LIB =", chosen)
lib = ctypes.CDLL(str(chosen))
for sym in ("has_avx512f_cpu", "has_avx512bf16_cpu"):
    print(f"  {sym:22s} {'present' if hasattr(lib, sym) else 'ABSENT'}")
EOF
) || echo "SELECTION_AUTO_FAILED"
( cd "$ROOT" && BNB_CPU_DISABLE_AVX512=1 PYTHONPATH="$PKG" python3 - <<'EOF'
import bitsandbytes.cextension as ce
print("forced AVX2: LOADED_NATIVE_LIB =", ce.LOADED_NATIVE_LIB)
EOF
) || echo "SELECTION_FORCED_FAILED"

say "[6/7] loader-level timing: adamw8bit step on 1M fp32 params"
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
echo "--- auto (loader chooses) ---"; run_step ""
echo "--- forced AVX2 (BNB_CPU_DISABLE_AVX512=1) ---"; run_step "BNB_CPU_DISABLE_AVX512=1"

say "[7/7] summary markers"
echo "MARK commit            $(git -C "$ROOT" log -1 --format=%h 2>/dev/null)"
echo "MARK cpu_avx512        $(grep -c avx512f /proc/cpuinfo 2>/dev/null | head -1)"
echo "MARK avx2_lib         $(ls -l "$PKG/bitsandbytes/libbitsandbytes_cpu.so" 2>/dev/null | awk '{print $5}')"
echo "MARK avx512_lib       $(ls -l "$PKG/bitsandbytes/libbitsandbytes_cpu_avx512.so" 2>/dev/null | awk '{print $5}')"
echo "MARK done"
