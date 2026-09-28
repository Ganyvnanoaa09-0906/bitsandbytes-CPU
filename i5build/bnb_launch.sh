#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# bnb_launch.sh -- self-contained Termux runner for the CPU-forge ARM64 test.
#
# Everything it needs is already on the device, so it does NOT depend on the PC
# being reachable, on curl, or on the USB tunnel:
#
#     /sdcard/Download/bnb_termux.tar.gz    the source bundle (341 KB)
#
# RUN THIS IN TERMUX. The file on /sdcard CANNOT be executed in place, because
# /sdcard is a FUSE filesystem that mounts everything non-executable, so
#     bash /sdcard/Download/bnb_launch.sh      -> "Permission denied"
# and chmod does not help. Copy it into the Termux home directory first, which
# is a normal filesystem. Two lines:
#
#     cp /sdcard/Download/bnb_launch.sh ~/ && bash ~/bnb_launch.sh
#
# If /sdcard is not readable at all, grant it once with `termux-setup-storage`
# (tap Allow) and change the path to:
#     /storage/emulated/0/Download/bnb_launch.sh
#
# OUTPUT GOES TO THE SCREEN AND TO A LOG. The first version redirected
# everything into a file, which left the user staring at a silent prompt with no
# way to tell whether it was working -- a bad property for a script someone runs
# by hand. Now it tees. `set -o pipefail` is deliberately NOT set: the tee child
# exiting first must not be mistaken for a test failure.
#
# KEEP THE TERMUX WINDOW IN THE FOREGROUND while it runs. `pkg install` wants a
# terminal; sending the app to the background mid-install can stall it. The
# first run installs clang (a few hundred MB), so allow a few minutes.
#
# ASCII only.
# ============================================================================
set -u

BUNDLE_SD="${BNB_BUNDLE:-/sdcard/Download/bnb_termux.tar.gz}"
WORK="$HOME/bnb_termux"
LOG="$WORK.log"
TMPLOG="/data/local/tmp/bnb_termux.log"
SDLOG="/sdcard/Download/bnb_termux.log"

# Show progress on screen AND keep a copy. --line-buffered so the terminal is
# not left behind when the script is slow.
exec > >(tee "$LOG") 2>&1

echo "### bnb termux run -- $(date)"
echo "### device : $(getprop ro.product.model 2>/dev/null) / Android $(getprop ro.build.version.release 2>/dev/null)"
echo "### prefix : ${PREFIX:-<unset>}"
echo "### arch   : $(uname -m)"
echo "### bundle : $BUNDLE_SD"
echo

step() { echo; echo "============================================================"; echo "== $*"; echo "============================================================"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- storage
step "storage"
if [ ! -r "$BUNDLE_SD" ]; then
  echo "cannot read $BUNDLE_SD"
  echo "listing /sdcard/Download (first 20):"
  ls -la /sdcard/Download 2>&1 | head -20
  echo
  echo "=> run 'termux-setup-storage' once (tap Allow), then re-run this script."
  cp "$LOG" "$TMPLOG" 2>/dev/null
  exit 1
fi
echo "bundle visible: $(ls -l "$BUNDLE_SD" | awk '{print $5}') bytes"

# ---------------------------------------------------------------- phase 0
step "Phase 0 -- environment"
echo "cpus : $(nproc 2>/dev/null || echo '?')"
awk '/MemTotal|MemAvailable/ {printf "%-9s: %.0f MB\n", $1, $2/1024}' /proc/meminfo 2>/dev/null
df -h /data 2>/dev/null | tail -1
echo
echo "--- tools before install ---"
for t in clang clang++ make nm python pip curl tar; do
  if have "$t"; then printf '  %-8s %s\n' "$t" "$(command -v $t)"; else printf '  %-8s MISSING\n' "$t"; fi
done

# ---------------------------------------------------------------- phase 1
step "Phase 1 -- install missing packages"
# DEBIAN_FRONTEND keeps apt from stopping on a config prompt; -y on pkg too.
export DEBIAN_FRONTEND=noninteractive
NEED=""
for t in clang make python nm; do have "$t" || NEED="$NEED $t"; done
if [ -n "$NEED" ]; then
  echo "pkg install -y clang make binutils python libomp   (missing:$NEED)"
  yes | pkg install -y clang make binutils python libomp 2>&1 | tail -25
else
  echo "toolchain already present; ensuring libomp"
  yes | pkg install -y libomp >/dev/null 2>&1 || true
fi
if ! python -c 'import numpy' >/dev/null 2>&1; then
  echo "pip install numpy ..."
  pip install --no-input --disable-pip-version-check numpy 2>&1 | tail -5
fi
echo
echo "--- tools after install ---"
for t in clang clang++ make nm python; do
  if have "$t"; then printf '  %-8s OK  %s\n' "$t" "$($t --version 2>&1 | head -1)"; else printf '  %-8s STILL MISSING\n' "$t"; fi
done
python -c 'import sys;print("  python",sys.version.split()[0])' 2>&1

# ---------------------------------------------------------------- phase 2
step "Phase 2 -- unpack the bundle"
rm -rf "$WORK"; mkdir -p "$WORK"
tar xzf "$BUNDLE_SD" -C "$WORK"
if [ ! -d "$WORK/bnb" ]; then
  echo "FATAL: bundle did not contain bnb/"
  ls -la "$WORK"
  cp "$LOG" "$TMPLOG" 2>/dev/null
  exit 1
fi
echo "unpacked:"; ls "$WORK/bnb" | head -25

# ---------------------------------------------------------------- phase 3
step "Phase 3 -- ARM64 build + C selftest"
cd "$WORK/bnb" || exit 1
bash build_termux.sh --selftest 2>&1
BUILD_RC=$?
echo
echo "[build_termux.sh exit code = $BUILD_RC]"

# ---------------------------------------------------------------- phase 4
step "Phase 4 -- python-side checks (ctypes / quantize / gemm / balancer)"
if have python && [ -f termux_check.py ]; then
  python termux_check.py 2>&1
  PY_RC=$?
  echo
  echo "[termux_check.py exit code = $PY_RC]   (0 pass, 77 nothing runnable, 1 fail)"
else
  echo "python or termux_check.py unavailable; skipped"
  PY_RC=77
fi

# ---------------------------------------------------------------- phase 5
step "Phase 5 -- verdict"
SO="$WORK/bnb/bitsandbytes/libbitsandbytes_cpu.so"
if [ -f "$SO" ]; then
  echo ".so       : $(ls -l "$SO" | awk '{print $5}') bytes"
  echo "elf head  : $(head -c 20 "$SO" | od -An -tx1 | tr -s ' ' | head -1)"
else
  echo ".so       : NOT BUILT"
fi
echo "build rc  : $BUILD_RC"
echo "python rc : $PY_RC"
echo
if [ "$BUILD_RC" = "0" ] && [ "$PY_RC" = "0" ]; then
  echo "########## TERMUX ACCEPTANCE: PASSED ##########"
else
  echo "########## TERMUX ACCEPTANCE: needs attention ##########"
fi

# ---------------------------------------------------------------- log out
cp "$LOG" "$TMPLOG" 2>/dev/null && echo "### log copied to $TMPLOG"
cp "$LOG" "$SDLOG" 2>/dev/null && echo "### log copied to $SDLOG"
echo "### log: $LOG"
echo "### done -- $(date)"
