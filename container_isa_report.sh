#!/usr/bin/env bash
# =============================================================================
# CPU-forge ISA compatibility probe -- for the EPYC / Linux container
# =============================================================================
# WHAT THIS DOES
#   1. prints the CPU model + every ISA flag we care about (AVX2 / FMA / F16C /
#      AVX-512 F/BW/DQ/VL/BF16/VNNI, AMX);
#   2. checks toolchain (g++, clang++, python3, pip, torch, transformers);
#   3. checks network reachability (PyPI, hf-mirror);
#   4. installs bitsandbytes-cpu-fork==0.50.2.dev4 into an isolated venv and runs
#      its torch-free CLI;
#   5. downloads the sdist and builds the CPU library matrix from source:
#        - AVX2 (or portable if the CPU has no AVX2)
#        - AVX-512 variant (only when the CPU has avx512f/bw/dq/vl)
#        - generic x86-64 (no AVX at all)
#      and runs the C self-test against the base and the generic libraries;
#   6. if torch is present, imports bitsandbytes and reports which native library
#      the loader actually selected (AVX2 / AVX-512 / generic), then runs the
#      Gated-DeltaNet kernel self-test;
#   7. writes everything to /tmp/cpu_forge_isa_report.txt and prints it.
#
# HOW TO RUN (from the directory containing this file):
#     bash container_isa_report.sh 2>&1 | tee /tmp/cpu_forge_isa_report.txt
#   Probe-only, no network/builds:
#     bash container_isa_report.sh --probe-only
#   Offline (skip all downloads/installs; needs a local sdist path):
#     BNB_SRC_TARBALL=/path/to/bitsandbytes_cpu_fork-0.50.2.dev4.tar.gz \
#         bash container_isa_report.sh --offline
#
# SAFETY: writes only under /tmp and a venv in /tmp. No sudo, no system files,
#         nothing outside /tmp. Safe to run on a shared machine.
# =============================================================================
set -u

PKG="bitsandbytes-cpu-fork==0.50.2.dev4"
PKG_SDIST_VER="0.50.2.dev4"
OUT="/tmp/cpu_forge_isa_report.txt"
WORK="/tmp/cpu_forge_isa"
VENV="$WORK/venv"
MODE="full"
SRC_TARBALL="${BNB_SRC_TARBALL:-}"

for arg in "$@"; do
  case "$arg" in
    --probe-only) MODE="probe" ;;
    --offline) MODE="offline" ;;
  esac
done

mkdir -p "$WORK"
: > "$OUT"
# tee everything (stdout+stderr) into the report file
exec > >(tee -a "$OUT") 2>&1

hdr() { echo; echo "==============================================================================="; echo "$*"; echo "==============================================================================="; }
have() { command -v "$1" >/dev/null 2>&1; }
flag_in() { # flag_in <flags-string> <flag>
  case " $1 " in *" $2 "*) echo "yes" ;; *) echo "no" ;; esac
}

hdr "0. QUESTIONS FOR YOU (answer the ones the probe cannot)"
cat <<'EOF'
  CPU model / vendor:
  AVX-512 flags present (avx512f/bw/dq/vl/bf16/vnni):
  g++ / clang++ version:
  python3 / pip version, is venv available:
  torch installed (version), transformers version:
  network: can pip reach PyPI? can you reach hf-mirror.com?
  do you have sudo/root:
  can you compile C++ (g++ works):
  notes / anything unusual:
EOF

hdr "1. DATE / KERNEL / DISTRO"
date -u 2>/dev/null || true
uname -a 2>/dev/null || true
( cat /etc/os-release 2>/dev/null | head -8 ) || true
echo "container hint: $(cat /proc/1/cgroup 2>/dev/null | head -2 | tr '\n' ' ')"

hdr "2. CPU / MEMORY / DISK"
echo "nproc: $(nproc 2>/dev/null || echo '?')"
(grep -m1 'model name' /proc/cpuinfo 2>/dev/null) || true
(lscpu 2>/dev/null | grep -E 'Model name|Vendor ID|Architecture|CPU\(s\)|Flags' | head -6) || true
free -h 2>/dev/null || true
df -h /tmp 2>/dev/null | tail -2 || true

FLAGS="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | sed 's/^[^:]*: *//' || true)"
echo
echo "ISA flags we care about:"
for f in avx avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_bf16 avx512vnni amx_bf16 amx_tile sse4_2; do
  printf '  %-14s %s\n' "$f" "$(flag_in "$FLAGS" "$f")"
done
echo
echo "AVX-512* flags actually reported by the kernel:"
echo "  $(echo "$FLAGS" | tr ' ' '\n' | grep -E '^avx512' | sort | tr '\n' ' ')"

hdr "3. TOOLCHAIN"
for t in g++ gcc clang++ python3 pip3 cmake make git; do
  if have "$t"; then
    printf '  %-8s %s\n' "$t" "$(command -v "$t") :: $("$t" --version 2>/dev/null | head -1)"
  else
    printf '  %-8s MISSING\n' "$t"
  fi
done
echo
echo "python3: $(python3 --version 2>&1 || echo missing)"
echo -n "torch: "; python3 -c 'import torch; print(torch.__version__)' 2>&1 | tail -1
echo -n "transformers: "; python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1
echo -n "numpy: "; python3 -c 'import numpy; print(numpy.__version__)' 2>&1 | tail -1

hdr "4. NETWORK"
probe_url() {
  python3 - "$1" <<'PY' 2>&1 | tail -1
import sys, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=20) as r:
        print("OK", r.status, url)
except Exception as e:
    print("FAIL", url, type(e).__name__, str(e)[:120])
PY
}
probe_url "https://pypi.org/simple/bitsandbytes-cpu-fork/"
probe_url "https://hf-mirror.com/"

if [ "$MODE" = "probe" ]; then
  hdr "9. DONE (probe-only mode: no installs/builds)"
  echo "report written to: $OUT"
  exit 0
fi

hdr "5. ISOLATED VENV + WHEEL INSTALL"
if have python3; then
  if python3 -m venv "$VENV" >/dev/null 2>&1; then
    echo "venv: $VENV"
    PY="$VENV/bin/python"
    "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
    if [ "$MODE" != "offline" ]; then
      echo "installing $PKG from PyPI (timeout 300s)..."
      timeout 300 "$PY" -m pip install -q --no-deps "$PKG" && echo "install: OK" || echo "install: FAILED"
      echo
      echo "torch-free CLI:"
      "$VENV/bin/bitsandbytes-cpu" version 2>&1 | tail -3 || true
      "$VENV/bin/bitsandbytes-cpu" detect 2>&1 | head -12 || true
      "$VENV/bin/bitsandbytes-cpu" doctor 2>&1 | tail -25 || true
    else
      echo "offline mode: skip PyPI install"
    fi
  else
    echo "venv creation FAILED (python3-venv missing?)"
    PY="python3"
  fi
else
  PY="python3"
fi

hdr "6. BUILD MATRIX FROM SDIST (g++ required)"
SRC_OK=0
if [ -n "$SRC_TARBALL" ] && [ -f "$SRC_TARBALL" ]; then
  cp "$SRC_TARBALL" "$WORK/" && SRC_OK=1
elif [ "$MODE" != "offline" ] && [ -x "$VENV/bin/python" ]; then
  echo "downloading sdist into $WORK ..."
  timeout 240 "$VENV/bin/python" -m pip download -q --no-deps --no-binary=:all: \
      -d "$WORK" "$PKG" && SRC_OK=1 || echo "sdist download FAILED"
fi
if [ "$SRC_OK" = "1" ]; then
  cd "$WORK" || exit 0
  TARBALL=$(ls -1 bitsandbytes_cpu_fork-*.tar.gz 2>/dev/null | head -1)
  if [ -n "$TARBALL" ]; then
    tar xzf "$TARBALL"
    SRCDIR="$WORK/bitsandbytes_cpu_fork-$PKG_SDIST_VER"
    echo "source dir: $SRCDIR"
    if have g++; then
      cd "$SRCDIR" || exit 0
      HEAD_AVX2=$(flag_in "$FLAGS" avx2)
      HEAD_AVX512=$(flag_in "$FLAGS" avx512f)
      echo
      echo "--- build base (AVX2 if available) + selftest ---"
      if [ "$HEAD_AVX2" = "yes" ]; then
        timeout 600 bash build_linux.sh --selftest 2>&1 | tail -12 || echo "base build FAILED"
      else
        timeout 600 bash build_linux.sh --no-avx2 --selftest 2>&1 | tail -12 || echo "base build(non-avx2) FAILED"
      fi
      echo
      echo "--- build generic x86-64 + selftest (portable fallback) ---"
      timeout 600 bash build_linux.sh --generic 2>&1 | tail -8 || echo "generic build FAILED"
      if [ -f bitsandbytes/libbitsandbytes_cpu_generic.so ] && have g++; then
        g++ -O2 -std=c++17 -DNOMINMAX -DNDEBUG -I csrc selftest_cpu.c \
            -L bitsandbytes -l:libbitsandbytes_cpu_generic.so \
            -Wl,-rpath,"$SRCDIR/bitsandbytes" -o "$WORK/selftest_generic" 2>&1 | tail -3 \
          && "$WORK/selftest_generic" 2>&1 | tail -4 || echo "generic selftest build FAILED"
      fi
      echo
      if [ "$HEAD_AVX512" = "yes" ]; then
        echo "--- build AVX-512 variant + selftest ---"
        timeout 900 bash build_linux.sh --avx512 2>&1 | tail -10 || echo "avx512 build FAILED"
        if [ -f bitsandbytes/libbitsandbytes_cpu_avx512.so ]; then
          echo -n "avx512 artifact contains bf16 instructions: "
          objdump -d bitsandbytes/libbitsandbytes_cpu_avx512.so 2>/dev/null \
            | grep -cE 'vdpbf16ps|vcvtneps2bf16|vcvtne2ps2bf16' || echo 0
          g++ -O2 -std=c++17 -DNOMINMAX -DNDEBUG -I csrc selftest_cpu.c \
              -L bitsandbytes -l:libbitsandbytes_cpu_avx512.so \
              -Wl,-rpath,"$SRCDIR/bitsandbytes" -o "$WORK/selftest_avx512" 2>&1 | tail -3 \
            && "$WORK/selftest_avx512" 2>&1 | tail -4 || echo "avx512 selftest build FAILED"
          echo -n "ctypes load avx512: "
          python3 -c "import ctypes; ctypes.CDLL('$SRCDIR/bitsandbytes/libbitsandbytes_cpu_avx512.so'); print('OK')" 2>&1 | tail -1
        fi
      else
        echo "--- AVX-512 variant: SKIPPED (CPU does not report avx512f) ---"
      fi
      echo
      echo "--- artifacts produced ---"
      ls -la bitsandbytes/libbitsandbytes*.so 2>/dev/null || true
    else
      echo "g++ missing: cannot build the matrix"
    fi
  else
    echo "no sdist tarball found in $WORK"
  fi
else
  echo "no sdist available (offline and BNB_SRC_TARBALL not set): skipping build matrix"
fi

hdr "7. RUNTIME SELECTION + GDN KERNEL (needs torch)"
if python3 -c 'import torch' >/dev/null 2>&1; then
  echo "torch is importable for python3"
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import bitsandbytes' >/dev/null 2>&1; then
    "$VENV/bin/python" - <<'PY' 2>&1 | tail -20
import bitsandbytes as bnb
from bitsandbytes.cextension import LOADED_NATIVE_LIB
print("bitsandbytes:", bnb.__version__)
print("selected native library:", LOADED_NATIVE_LIB)
from bitsandbytes import _cpu_caps
print("avx2_supported:", _cpu_caps.avx2_supported(), "| avx512_supported:", _cpu_caps.avx512_supported())
try:
    bnb.gdn_cpu.load_native()
    print("gdn load_native: OK")
except Exception as e:
    print("gdn load_native FAILED:", e)
PY
    echo
    echo "--- gdn_cpu self-test (fwd/bwd vs fp64 reference) ---"
    timeout 300 "$VENV/bin/python" -m bitsandbytes.gdn_cpu 2>&1 | tail -10 || echo "gdn selftest FAILED"
  else
    echo "package not installed in the venv (offline / install failed): skipping bitsandbytes runtime checks"
  fi
else
  echo "torch not installed for python3: skipping runtime selection (use 'bitsandbytes-cpu doctor' output above)"
fi

hdr "8. DONE"
echo "PLEASE COPY THE WHOLE OUTPUT BACK (or the file below):"
echo "  $OUT"
echo "size: $(wc -c < "$OUT" 2>/dev/null || echo '?') bytes"
