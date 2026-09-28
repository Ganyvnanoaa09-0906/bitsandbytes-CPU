#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# setup_termux_test.sh -- one-shot Termux acceptance for the CPU-forge port.
#
# RUN THIS INSIDE TERMUX (the phone), not on the PC. One line:
#
#     curl -fsSL http://127.0.0.1:8899/setup_termux_test.sh | bash
#
# 127.0.0.1 IS THE PC. The PC port 8899 is forwarded to the phone over the USB
# cable with `adb reverse tcp:8899 tcp:8899`, so this works with no Wi-Fi at all.
# That matters here: this phone CANNOT reach the PC over the LAN. Measured on the
# device: `ping 10.37.193.21` gives 100% packet loss even though both are on
# 10.37.193.0/24, i.e. the access point isolates wireless clients from each other.
# So do not "fix" this to the LAN address; the USB tunnel is the working path.
#
# Use a real script file rather than `curl | bash` if the pipe gives you trouble:
#     curl -fsSL -o t.sh http://127.0.0.1:8899/setup_termux_test.sh && bash t.sh
#
# WHAT IT DOES
#   Phase 0  environment report (arch, termux prefix, tools, python)
#   Phase 1  install what is missing: clang, libomp, make, python, numpy
#   Phase 2  fetch the source bundle from the PC and unpack it
#   Phase 3  build libbitsandbytes_cpu.so for ARM64 and run the C selftest
#   Phase 4  run the python-side checks (ctypes / quantize / gemm / balancer)
#   Phase 5  print a verdict and leave the log on shared storage
#
# The log is copied to /data/local/tmp/bnb_termux.log, which the PC can read
# back over adb; Termux's own home directory is unreadable without root.
#
# WHY IT CANNOT RUN THE FULL TEST SUITE: there is no Android/arm64 torch wheel
# (manylinux arm64 targets glibc; Android uses bionic). Phase 4 covers the part
# that is verifiable without torch, which is exactly the part this port changed.
#
# ASCII only: this is piped to bash and lands in logs.
# ============================================================================
set -u

HOST="${BNB_HOST:-127.0.0.1}"
PORT="${BNB_PORT:-8899}"
WORK="${HOME:-$PREFIX/tmp}/bnb_termux"
TMPLOG="/data/local/tmp/bnb_termux.log"
SDLOG="/sdcard/Download/bnb_termux.log"

# NO OUTPUT REDIRECTION HERE, DELIBERATELY.
#
# The first version started with `exec > "$LOG" 2>&1`. When this script is run
# as `curl ... | bash`, $HOME is not set, so "$WORK.log" expanded to
# "/bnb_termux.log", the redirect failed on a read-only root, and bash exited
# immediately -- producing a completely silent prompt. The server log showed the
# download had happened, so the script DID start; the redirect killed it before
# it printed one character. A script a human runs by hand must never be silent.
#
# Output now goes to the terminal. To also keep a copy:
#     curl -fsSL $URL -o t.sh && bash t.sh 2>&1 | tee run.log
echo "### bnb termux acceptance -- $(date)"
echo "### device   : $(getprop ro.product.model 2>/dev/null) / $(getprop ro.build.version.release 2>/dev/null)"
echo "### home     : ${HOME:-<unset>}"
echo "### work dir : $WORK"
echo "### host     : $HOST:$PORT (USB tunnel via adb reverse)"
echo
echo "NOTE: output is going to your screen only. If you want a file copy run:"
echo "      curl -fsSL http://$HOST:$PORT/setup_termux_test.sh -o t.sh && bash t.sh 2>&1 | tee run.log"

step() { echo; echo "============================================================"; echo "== $*"; echo "============================================================"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- phase 0
step "Phase 0 -- environment"
echo "arch      : $(uname -m)"
echo "kernel    : $(uname -r)"
echo "prefix    : ${PREFIX:-<unset>}"
echo "home      : $HOME"
echo "cpus      : $(nproc 2>/dev/null || echo '?')"
if [ -r /proc/meminfo ]; then
  awk '/MemTotal|MemAvailable/ {printf "%-10s: %.0f MB\n", $1, $2/1024}' /proc/meminfo
fi
df -h /data 2>/dev/null | tail -1

echo
echo "--- tools ---"
for t in clang clang++ gcc g++ make cmake python python3 pip curl wget git tar nm; do
  if have "$t"; then printf '  %-8s %s\n' "$t" "$(command -v $t)"; else printf '  %-8s MISSING\n' "$t"; fi
done

echo
echo "--- python ---"
if have python; then python -c 'import sys,platform;print("  ",sys.version.split()[0],platform.machine())' 2>&1; fi
python - <<'PY' 2>&1 || true
try:
    import numpy
    print("   numpy", numpy.__version__)
except Exception as e:
    print("   numpy MISSING:", type(e).__name__)
try:
    import torch
    print("   torch", torch.__version__)
except Exception as e:
    print("   torch MISSING:", type(e).__name__, "(expected on Android)")
PY

# ---------------------------------------------------------------- phase 1
step "Phase 1 -- install missing tools"
NEED=""
for t in clang make python; do
  have "$t" || NEED="$NEED $t"
done
have nm || NEED="$NEED binutils"
if [ -n "$NEED" ]; then
  echo "installing:$NEED   (plus libomp for OpenMP)"
  # --no-install-recommends and -y keep this non-interactive
  pkg install -y clang make binutils python libomp 2>&1 | tail -20
else
  echo "clang / make / python already present"
  # libomp may still be missing; try quietly and ignore failure
  pkg install -y libomp >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# numpy and psutil come from pkg, NOT from pip.
#
# Measured on this device (Android 10, aarch64, python 3.14.6): `pip install
# numpy` does not find a wheel and tries to build from source, which fails with
#     "Encountered error while generating package metadata"
# There is no Android aarch64 numpy wheel on PyPI -- the manylinux aarch64
# wheels target glibc and Android uses bionic. Termux ships its own prebuilt
# packages, which is the supported route. pip is kept only as a fallback so the
# script still gets as far as it can if pkg has no candidate.
#
# disk_balancer needs psutil; latent_chunk_store needs numpy.
# ---------------------------------------------------------------------------
for p in python-numpy python-psutil; do
  if ! python -c "import ${p#python-}" >/dev/null 2>&1; then
    echo "pkg install -y $p"
    yes | pkg install -y "$p" 2>&1 | tail -8
  fi
done
for m in numpy psutil; do
  if python -c "import $m" >/dev/null 2>&1; then
    echo "  [OK]   $m $(python -c "import $m;print(getattr($m,'__version__','?'))" 2>/dev/null)"
  else
    echo "  [MISS] $m -- retrying via pip (may fail on Android, see comment)"
    pip install --no-input --disable-pip-version-check "$m" 2>&1 | tail -4
  fi
done

# ---------------------------------------------------------------------------
# torch: probe, then install only if it is genuinely available and affordable.
#
# This is what decides whether the python TRAINING path (disk_balancer,
# latent_chunk_store, train_1000_steps) is testable on this device at all. PyPI
# cannot supply it: every aarch64 wheel is manylinux_2_28, i.e. glibc, and
# Android is bionic. Termux's own repo carries a community pytorch package, so
# ask the package manager rather than assuming either way.
#
# The device has ~5.8 GB free, so a multi-GB download is refused rather than
# started and abandoned halfway.
# ---------------------------------------------------------------------------
echo
echo "--- torch availability ---"
TORCH_PKG=""
for cand in python-pytorch pytorch python-torch; do
  if pkg show "$cand" >/dev/null 2>&1; then
    TORCH_PKG="$cand"
    echo "  found in Termux repo: $cand"
    pkg show "$cand" 2>/dev/null | grep -E '^(Package|Version|Installed-Size|Download-Size|Depends):' | sed 's/^/    /'
    break
  fi
done
if [ -z "$TORCH_PKG" ]; then
  echo "  no torch package in the Termux repo."
  echo "  => the python training path cannot run here. The C kernels, the ctypes"
  echo "     boundary, 8-bit quantize and gemm_8bit are still fully verified by"
  echo "     Phase 3 and Phase 4 (T2-T5), which are torch-free by design."
elif [ "${BNB_SKIP_TORCH:-0}" = "1" ]; then
  echo "  BNB_SKIP_TORCH=1 -- not installing"
else
  FREE_MB=$(df -Pm /data 2>/dev/null | awk 'NR==2{print $4}')
  echo "  free on /data: ${FREE_MB:-?} MB"
  if [ -n "${FREE_MB:-}" ] && [ "$FREE_MB" -lt 6000 ]; then
    echo "  refusing: less than 6000 MB free. The python training path stays"
    echo "  untested here rather than filling the device."
  else
    echo "  installing $TORCH_PKG (this is the big one; it may take a while)"
    yes | pkg install -y "$TORCH_PKG" 2>&1 | tail -15
  fi
fi
if python -c 'import torch' >/dev/null 2>&1; then
  echo "  [OK]   torch $(python -c 'import torch;print(torch.__version__)')"
else
  echo "  [--]   torch not importable here"
fi

echo
echo "--- after install ---"
for t in clang clang++ make python nm; do
  if have "$t"; then printf '  %-8s OK\n' "$t"; else printf '  %-8s STILL MISSING\n' "$t"; fi
done
python -c 'import sys;print("  python",sys.version.split()[0])' 2>&1

# ---------------------------------------------------------------- phase 2
step "Phase 2 -- fetch and unpack the source bundle"
rm -rf "$WORK"; mkdir -p "$WORK"
URL="http://$HOST:$PORT/bnb_termux.tar.gz"
echo "GET $URL"
if have curl; then
  curl -fsSL --connect-timeout 15 -o "$WORK/b.tar.gz" "$URL" || echo "curl failed"
elif have wget; then
  wget -q -T 15 -O "$WORK/b.tar.gz" "$URL" || echo "wget failed"
else
  echo "FATAL: neither curl nor wget"
fi
if [ ! -s "$WORK/b.tar.gz" ]; then
  echo "FATAL: bundle not downloaded (size 0 or missing)."
  echo "       The PC reaches this device only through the USB tunnel, so check:"
  echo "         1. the USB cable is still connected"
  echo "         2. on the PC:  adb reverse tcp:$PORT tcp:$PORT"
  echo "         3. on the PC:  the http server is still running on port $PORT"
  echo "       Then retry this script."
  exit 1
fi
echo "downloaded $(wc -c < "$WORK/b.tar.gz") bytes"
tar xzf "$WORK/b.tar.gz" -C "$WORK" && echo "unpacked to $WORK"
ls "$WORK" "$WORK/bnb" 2>/dev/null | head -20

# ---------------------------------------------------------------- phase 3
step "Phase 3 -- ARM64 build + C selftest"
cd "$WORK/bnb" || { echo "FATAL: no bnb dir"; exit 1; }
# --selftest makes build_termux.sh also compile and run the torch-free C checks.
# Capture the code on the very next statement: `var=$?` after anything else picks
# up that something-else's status instead.
bash build_termux.sh --selftest 2>&1
BUILD_RC=$?
echo
echo "[build_termux.sh exit code = $BUILD_RC]"

# ---------------------------------------------------------------- phase 4
step "Phase 4 -- python-side checks (ctypes / quantize / gemm / balancer)"
cd "$WORK/bnb" || exit 1
if have python; then
  python termux_check.py 2>&1
  PY_RC=$?
  echo
  echo "[termux_check.py exit code = $PY_RC]   (0 pass, 77 nothing runnable, 1 fail)"
else
  echo "python not available; skipped"
  PY_RC=77
fi

# ---------------------------------------------------------------- phase 5
step "Phase 5 -- verdict"
SO="$WORK/bnb/bitsandbytes/libbitsandbytes_cpu.so"
if [ -f "$SO" ]; then
  echo ".so        : $(ls -l "$SO" | awk '{print $5}') bytes"
  echo "file type  : $(head -c 20 "$SO" | od -An -tx1 | tr -s ' ' | head -1)"
else
  echo ".so        : NOT BUILT"
fi
echo "build rc   : $BUILD_RC"
echo "python rc  : $PY_RC"
if [ "$BUILD_RC" = "0" ] && [ "$PY_RC" = "0" ]; then
  echo
  echo "########## TERMUX ACCEPTANCE: PASSED ##########"
else
  echo
  echo "########## TERMUX ACCEPTANCE: needs attention (see above) ##########"
fi

# There is no log file to copy any more -- output went to the terminal. What we
# CAN leave behind is the built artefacts and the python-side result, so the PC
# can inspect them over adb. /data/local/tmp is world-writable and readable by
# the adb shell, and Termux can write there.
RESULT="$WORK/bnb/RESULT.txt"
{
  echo "arch      : $(uname -m)"
  echo "prefix    : ${PREFIX:-<unset>}"
  echo "build rc  : $BUILD_RC"
  echo "python rc : $PY_RC"
  echo "so        : $(ls -l "$SO" 2>/dev/null || echo MISSING)"
  uname -a
} > "$RESULT" 2>/dev/null && {
  cp "$RESULT" "$TMPLOG" 2>/dev/null && echo "### result copied to $TMPLOG"
  cp "$RESULT" "$SDLOG" 2>/dev/null && echo "### result copied to $SDLOG"
}
echo "### done -- $(date)"
