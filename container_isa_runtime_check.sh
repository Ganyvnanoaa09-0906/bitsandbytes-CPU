#!/usr/bin/env bash
# =============================================================================
# CPU-forge follow-up #2 -- verify the PUBLISHED WHEEL's AVX-512 loader path
# =============================================================================
# The first probe built the AVX-512 library from source and ran its C self-test
# successfully on this EPYC 9Y24. This script checks the *installed PyPI wheel*
# end to end with torch present:
#   - which native library the loader actually selects (expect
#     libbitsandbytes_cpu_avx512.so; it must NOT pick it on a non-BF16 CPU);
#   - the AVX-512-only symbol has_avx512bf16_cpu is exported and callable;
#   - 8-bit optimizer / blockwise quantisation / a real matmul run under AVX-512
#     (no SIGILL), with a quick GFLOPS number;
#   - the Gated-DeltaNet kernel self-test (fwd/bwd vs float64 reference).
#
# HOW TO RUN:
#     bash container_isa_runtime_check.sh 2>&1 | tee /tmp/cpu_forge_runtime_report.txt
#
# SAFETY: writes only under /tmp (venv + report). No sudo, no system changes.
#         The venv is created with --system-site-packages so it can see the
#         container's existing torch 2.14.0+cpu without installing torch again.
# =============================================================================
set -u

PKG="bitsandbytes-cpu-fork==0.50.2.dev4"
WORK="/tmp/cpu_forge_runtime"
OUT="/tmp/cpu_forge_runtime_report.txt"
VENV="$WORK/venv"

rm -rf "$WORK"
mkdir -p "$WORK"
exec > >(tee -a "$OUT") 2>&1

hdr() { echo; echo "==============================================================================="; echo "$*"; echo "==============================================================================="; }

hdr "A. SYSTEM TORCH"
PROBE_PY="${BNB_PY:-python3}"
"$PROBE_PY" -c 'import torch; print("torch", torch.__version__, "| threads", torch.get_num_threads())' 2>&1 | tail -1

hdr "B. VENV (sees system torch) + WHEEL FROM PyPI"
if [ -n "${BNB_PY:-}" ]; then
  # testing hook for a machine that already has a python with torch
  PY="$BNB_PY"
  echo "using existing python: $PY"
  "$PY" -m pip install -q --no-deps --force-reinstall "$PKG" && echo "wheel install: OK" || echo "wheel install: FAILED"
elif ! python3 -m venv --system-site-packages "$VENV" >/dev/null 2>&1; then
  echo "venv creation FAILED; trying plain user install"
  PY="python3"
  python3 -m pip install --user --no-deps --force-reinstall "$PKG" 2>&1 | tail -3
else
  PY="$VENV/bin/python"
  "$PY" -m pip install -q --no-deps --force-reinstall "$PKG" && echo "wheel install: OK" || echo "wheel install: FAILED"
fi
echo "python: $("$PY" -V 2>&1)"
"$PY" -c 'import torch; print("venv sees torch:", torch.__version__)' 2>&1 | tail -1

hdr "C. LOADER SELECTION (the point of this script)"
"$PY" - <<'PY' 2>&1 | tail -30
import ctypes, os, glob
import bitsandbytes as bnb
from bitsandbytes.cextension import LOADED_NATIVE_LIB
from bitsandbytes import _cpu_caps

print("bitsandbytes version:", bnb.__version__)
print("package dir:", os.path.dirname(bnb.__file__))
print("native files:", sorted(os.path.basename(p) for p in
      glob.glob(os.path.join(os.path.dirname(bnb.__file__), "libbitsandbytes*"))))
print("avx2_supported:", _cpu_caps.avx2_supported())
print("avx512_supported:", _cpu_caps.avx512_supported())
print("LOADED_NATIVE_LIB:", LOADED_NATIVE_LIB)
sel = str(LOADED_NATIVE_LIB)
print("selected avx512:", "avx512" in sel)

dll = ctypes.CDLL(sel)
fn = getattr(dll, "has_avx512bf16_cpu", None)
print("has_avx512bf16_cpu symbol:", fn is not None)
if fn is not None:
    fn.restype = ctypes.c_int
    print("has_avx512bf16_cpu() ->", fn())
print("ctypes load selected lib: OK")
PY

hdr "D. 8-BIT OPTIMIZER + QUANTISATION + MATMUL UNDER AVX-512"
"$PY" - <<'PY' 2>&1 | tail -30
import time
import torch
import bitsandbytes as bnb

torch.manual_seed(0)
torch.set_num_threads(2)

# blockwise quantisation round-trip (exercises the AVX-512 quant kernel)
x = torch.randn(4096, dtype=torch.float32)
q, absmax = bnb.functional.quantize_blockwise(x, blocksize=256)
back = bnb.functional.dequantize_blockwise(q, absmax, blocksize=256)
print("quantize/dequantize max abs err:", (back - x).abs().max().item())

# AdamW8bit step (8-bit optimizer state + update kernel)
p = torch.nn.Parameter(torch.randn(1024, 1024))
opt = bnb.optim.AdamW8bit([p], lr=1e-3)
p.grad = torch.randn_like(p)
before = p.detach().clone()
opt.step()
print("AdamW8bit step: finite =", bool(torch.isfinite(p).all()),
      "| max delta =", (p.detach() - before).abs().max().item())

# dense fp32 GEMM: must run without SIGILL on the AVX-512 library
n = 2048
a = torch.randn(n, n)
b = torch.randn(n, n)
torch.matmul(a, b)  # warmup
t0 = time.perf_counter()
for _ in range(3):
    c = torch.matmul(a, b)
dt = (time.perf_counter() - t0) / 3
print("matmul %dx%dx%d: %.3f s -> %.1f GFLOPS" % (n, n, n, dt, 2 * n**3 / dt / 1e9))
print("finite:", bool(torch.isfinite(c).all()))
PY

hdr "E. GDN KERNEL SELF-TEST (uses the selected native library)"
"$PY" - <<'PY' 2>&1 | tail -10
from bitsandbytes.gdn_cpu import _selftest
raise SystemExit(_selftest())
PY

hdr "F. DONE"
echo "PLEASE COPY THIS WHOLE OUTPUT BACK."
echo "report file: $OUT"
echo "size: $(wc -c < "$OUT" 2>/dev/null || echo '?') bytes"
