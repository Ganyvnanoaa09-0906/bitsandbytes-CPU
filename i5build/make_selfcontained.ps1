# make_selfcontained.ps1 -- build ONE shell script that carries the whole bundle
# inside itself as base64, so the phone needs no network and no server.
#
# WHY: the acceptance run kept being blocked by the PC side of the link rather
# than by the phone. A backgrounded `python -m http.server` died mid-session
# (along with the adb daemon), so the user's `curl` hung on a port with nothing
# listening -- a failure of MY transport, not of the code under test. Pushing an
# archive to /sdcard does not work either: adb creates it root:sdcard_rw mode
# 0660, and the Termux uid is not in that group, so cp gives "Permission denied".
#
# /data/local/tmp has neither problem: it is 0777 and the adb shell can write it.
# So: adb push one .sh there, and running it needs nothing but the device.
#
# Usage: powershell -File make_selfcontained.ps1
# Output: D:\work\termux_serve\bnb_selfcontained.sh  (+ pushed to the device)

$ErrorActionPreference = 'Stop'

$serve = 'D:\work\termux_serve'
$tar   = "$serve\bnb_termux.tar.gz"
$out   = "$serve\bnb_selfcontained.sh"

if (-not (Test-Path $tar)) { throw "missing $tar -- run build_termux_bundle.ps1 first" }

Write-Host "=== base64-encoding the bundle ==="
$bytes = [System.IO.File]::ReadAllBytes($tar)
$b64   = [System.Convert]::ToBase64String($bytes)
Write-Host ("   {0:N0} bytes -> {1:N0} chars of base64" -f $bytes.Length, $b64.Length)

# 76-char lines: base64 -d accepts them and the file stays editable/readable.
$lines = for ($i = 0; $i -lt $b64.Length; $i += 76) {
    $b64.Substring($i, [Math]::Min(76, $b64.Length - $i))
}

$header = @'
#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
# bnb_selfcontained.sh -- run the whole Termux acceptance with NO network.
#
# The source bundle is embedded below as base64, so this file needs nothing from
# the PC: no http server, no adb reverse tunnel, no curl. That matters because
# every earlier attempt was blocked by the PC side of the link going away
# mid-session, which says nothing about the code being tested.
#
# RUN (the file is on /data/local/tmp, which the adb shell can write and Termux
# can read; /sdcard cannot be used because adb-created files there are mode 0660
# root:sdcard_rw and the Termux uid is not in that group):
#
#     bash /data/local/tmp/bnb_selfcontained.sh
#
# Optional:
#     BNB_NO_OPENMP=1 bash /data/local/tmp/bnb_selfcontained.sh
#         rebuild the .so without OpenMP as well, which is what makes the
#         torch + bitsandbytes path testable in one process (see below).
#
# WHAT IT DOES: unpack -> ARM64 build + 5-check C selftest -> python checks
# (ctypes/quantize/gemm torch-free, then torch-only, then torch+bnb in its own
# process). Output goes to your screen; a copy lands in /sdcard/Download.
#
# ASCII only.
# ============================================================================
set -u

WORK="$HOME/bnb_termux"
TMPLOG=/data/local/tmp/bnb_termux_result.txt
SDLOG=/sdcard/Download/bnb_termux_result.txt

step() { echo; echo "============================================================"; echo "== $*"; echo "============================================================"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "unpack (embedded bundle, no network)"
rm -rf "$WORK"; mkdir -p "$WORK"
B64="$WORK/b.tar.gz.b64"
cat > "$B64" <<'__B64_EOF__'
'@

$footer = @'
__B64_EOF__
base64 -d "$B64" > "$WORK/b.tar.gz" || { echo "FATAL: base64 decode failed"; exit 1; }
rm -f "$B64"
echo "bundle: $(wc -c < "$WORK/b.tar.gz") bytes"
tar xzf "$WORK/b.tar.gz" -C "$WORK" || { echo "FATAL: untar failed"; exit 1; }
[ -d "$WORK/bnb" ] || { echo "FATAL: no bnb/ in bundle"; exit 1; }
ls "$WORK/bnb" | head -25

step "environment"
echo "arch   : $(uname -m)"
echo "prefix : ${PREFIX:-<unset>}"
python -c 'import sys;print("python :",sys.version.split()[0])' 2>/dev/null
for m in numpy psutil torch; do
  python -c "import $m;print('  $m', getattr($m,'__version__','?'))" 2>/dev/null || echo "  $m MISSING"
done

step "ARM64 build + C selftest"
cd "$WORK/bnb" || exit 1
bash build_termux.sh --selftest
BUILD_RC=$?
echo "[build_termux.sh exit code = $BUILD_RC]"

step "python checks (torch-free first, then torch, then torch+bnb)"
python termux_check.py
PY_RC=$?
echo "[termux_check.py exit code = $PY_RC]"

# ---------------------------------------------------------------------------
# Rebuild WITHOUT OpenMP, then TRAIN.
#
# This is not optional, it is the order the platform forces:
#   * build_termux.sh links libomp statically by default, and torch ships its own
#     copy; two OpenMP runtimes in one process make LLVM abort (Error #15). The
#     training script imports torch AND bitsandbytes, so with the default build it
#     cannot even start.
#   * a single-threaded .so cannot collide with anything, and numerical
#     correctness does not depend on the thread count -- the C selftest asserts
#     vector == scalar bit-for-bit, and the measured results are identical.
#
# KMP_DUPLICATE_LIB_OK is deliberately NOT used to paper over the abort: its own
# documentation says it "may cause crashes or silently produce incorrect
# results", and a run whose purpose is to show that the loss descends correctly
# must not execute under a flag that permits silently wrong answers.
# ---------------------------------------------------------------------------
step "rebuild WITHOUT OpenMP (required before training can start)"
bash build_termux.sh --no-openmp --selftest
NOMP_RC=$?
echo "[no-openmp build rc = $NOMP_RC]"

step "training smoke (${BNB_STEPS:-50} steps)"
python termux_train.py --steps "${BNB_STEPS:-50}" --threads "${BNB_THREADS:-4}"
TRAIN_RC=$?
echo "[termux_train.py exit code = $TRAIN_RC]"

step "re-run python checks against the single-threaded .so"
python termux_check.py
PY2_RC=$?
echo "[termux_check.py after no-openmp = $PY2_RC]"

step "verdict"
SO="$WORK/bnb/bitsandbytes/libbitsandbytes_cpu.so"
if [ -f "$SO" ]; then
  echo "size      : $(wc -c < "$SO") bytes"
  echo "elf head  : $(head -c 20 "$SO" | od -An -tx1 | tr -s ' ')"
else
  echo "size      : NOT BUILT"
fi
echo "build rc  : $BUILD_RC"
echo "python rc : $PY_RC"
echo "noomp rc  : $NOMP_RC"
echo "train rc  : $TRAIN_RC"
echo "python rc after noomp: $PY2_RC"
echo
if [ "$BUILD_RC" = "0" ] && [ "$TRAIN_RC" = "0" ]; then
  echo "########## TERMUX ACCEPTANCE: PASSED (C selftest + python checks + training) ##########"
else
  echo "########## TERMUX ACCEPTANCE: needs attention -- see the rcs above ##########"
fi

{ echo "arch=$(uname -m)"; echo "build_rc=$BUILD_RC"; echo "python_rc=$PY_RC";
  echo "noomp_rc=$NOMP_RC"; echo "train_rc=$TRAIN_RC"; echo "python_rc_after_noomp=$PY2_RC";
  ls -l "$SO" 2>/dev/null; } > "$TMPLOG" 2>/dev/null \
  || { TMPLOG="$WORK/bnb_termux_result.txt"; { echo "arch=$(uname -m)";
        echo "build_rc=$BUILD_RC"; echo "python_rc=$PY_RC";
        echo "noomp_rc=$NOMP_RC"; echo "train_rc=$TRAIN_RC";
        echo "python_rc_after_noomp=$PY2_RC";
        ls -l "$SO" 2>/dev/null; } > "$TMPLOG"; }
cp "$TMPLOG" "$SDLOG" 2>/dev/null && echo "result copied to $SDLOG"
echo "result file: $TMPLOG"
'@

$content = $header + "`n" + ($lines -join "`n") + "`n" + $footer
# LF endings: this is a shell script for a Linux-y environment; CRLF would make
# bash choke on every line ("\r: command not found").
$content = $content -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($out, $content, [System.Text.UTF8Encoding]::new($false))

# --- structural checks, because `bash -n` alone did not catch a real defect ---
# PowerShell's here-string syntax STRIPS the newline before the closing '@, so the
# first version concatenated the base64 straight onto the heredoc opener:
#     cat > "$B64" <<'__B64_EOF__'H4sIAATcuWoA...
# and bash reported "here-document delimited by end-of-file". `bash -n` printed
# that warning but still exited 0, so an exit-code check is not sufficient here.
# Assert the exact lines instead.
$gen = [System.IO.File]::ReadAllLines($out)
$opener = ($gen | Where-Object { $_ -eq "cat > `"`$B64`" <<'__B64_EOF__'" }).Count
$closer = ($gen | Where-Object { $_ -eq '__B64_EOF__' }).Count
if ($opener -ne 1) { throw "heredoc opener line malformed or missing (found $opener)" }
if ($closer -ne 1) { throw "heredoc closer line malformed or missing (found $closer)" }
$b64lines = $gen | Where-Object { $_ -match '^[A-Za-z0-9+/=]+$' }
if ($b64lines.Count -lt 100) { throw "base64 payload looks wrong ($($b64lines.Count) lines)" }
Write-Host "   structure: opener OK, closer OK, $($b64lines.Count) base64 lines"

$fi = Get-Item $out
Write-Host ("=== wrote {0} = {1:N0} bytes ({2:N1} MB)" -f $fi.Name, $fi.Length, ($fi.Length / 1MB))

Write-Host "=== syntax check with bash -n ==="
$bash = 'C:\Program Files\Git\bin\bash.exe'
if (Test-Path $bash) {
    # Capture stderr explicitly: `bash -n` printed a here-document warning and
    # STILL exited 0 in the first version, so the exit code alone proves nothing.
    $syn = & $bash -n $out 2>&1
    if ($syn) { throw "bash -n reported: $syn" }
    Write-Host "   OK (no output at all)"
} else {
    Write-Host "   bash not found, skipped"
}

Write-Host "=== payload round-trip (decode and compare against the original) ==="
$verifier = Join-Path $PSScriptRoot 'verify_selfcontained.py'
if (Test-Path $verifier) {
    & python $verifier $out $tar
    if ($LASTEXITCODE -ne 0) { throw "payload verification failed" }
} else {
    Write-Host "   verify_selfcontained.py not found, skipped"
}

Write-Host "=== push to the device ==="
$adb = 'C:\Program Files\Netease\MuMu\nx_main\adb.exe'
# /data/local/tmp, not /sdcard: it is 0777 and the adb shell owns what it writes,
# so Termux can actually read it. Files adb creates under /sdcard come out
# root:sdcard_rw mode 0660 and Termux is not in that group ("Permission denied").
& $adb push $out /data/local/tmp/bnb_selfcontained.sh
& $adb shell 'ls -la /data/local/tmp/bnb_selfcontained.sh'
