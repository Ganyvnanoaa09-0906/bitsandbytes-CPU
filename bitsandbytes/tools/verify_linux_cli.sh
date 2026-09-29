#!/usr/bin/env bash
# What a Linux user sees right after pip install -- with no torch on the machine.
#
# WHY THIS EXISTS
# ---------------
# This distribution deliberately does not depend on torch (on PyPI that name is the CUDA
# build, about 2.5 GB of nvidia-* wheels), so a user can have the package installed and
# torch absent. That is exactly the user who runs `help`, because the help is what says
# to install torch. Pointed at bitsandbytes.cli:main, the console script died with
# `ModuleNotFoundError: No module named 'torch'` before printing a word. Reproduced on
# Ubuntu 26.04 / Python 3.14 from a plain sdist install, then fixed by moving the
# reference into the torch-free top-level module bitsandbytes_cpu_cli.py.
#
# This script is the regression test for that, and it also builds the sdist for real --
# pip compiles libbitsandbytes_cpu.so through build_linux.sh, which is the other half of
# the Linux path and had never once succeeded before the <cstdio> fix in csrc/cpu_cache.h.
#
# USAGE
#     bash tools/verify_linux_cli.sh [artifact.tar.gz]
#
# With an artifact it pip-installs that file first (needs --break-system-packages on a
# distribution-managed Python, or use a venv); without one it checks whatever is already
# installed. Exit status is 0 only if every command behaves.
set -u

ARTIFACT="${1:-}"
PY="${PYTHON:-python3}"
PIP_EXTRA="${PIP_EXTRA:---break-system-packages}"
fails=0

say()  { printf '\n=== %s ===\n' "$*"; }

# The commands that must work with no torch, and the ones that must refuse cleanly.
check_ok() {
    local label="$1"; shift
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ "$rc" -eq 0 ]; then
        printf '  ok    %-34s exit=0\n' "$label"
        printf '%s\n' "$out" | head -3 | sed 's/^/          /'
    else
        printf '  FAIL  %-34s exit=%s\n' "$label" "$rc"
        printf '%s\n' "$out" | head -6 | sed 's/^/          /'
        fails=$((fails + 1))
    fi
}

check_fails_cleanly() {
    local label="$1" want="$2"; shift 2
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ "$rc" -eq "$want" ]; then
        printf '  ok    %-34s exit=%s\n' "$label" "$rc"
    else
        printf '  FAIL  %-34s exit=%s (wanted %s)\n' "$label" "$rc" "$want"
        fails=$((fails + 1))
    fi
    printf '%s\n' "$out" | head -4 | sed 's/^/          /'
}

if [ -n "$ARTIFACT" ]; then
    say "install $ARTIFACT"
    # shellcheck disable=SC2086
    "$PY" -m pip install $PIP_EXTRA --no-deps --force-reinstall "$ARTIFACT" 2>&1 | tail -3
fi

say "what got installed"
"$PY" - <<'EOF'
import importlib.util, os, sys
spec = importlib.util.find_spec("bitsandbytes")
root = list(spec.submodule_search_locations)[0] if spec else None
print("  package      ", root or "<not installed>")
if root:
    for n in sorted(os.listdir(root)):
        if n.startswith("libbitsandbytes"):
            print(f"  native       {n}  {os.path.getsize(os.path.join(root, n)) / 1e6:.2f} MB")
cli = importlib.util.find_spec("bitsandbytes_cpu_cli")
print("  cli module   ", "present" if cli else "MISSING (help will not run without torch)")
EOF

say "torch on this machine"
"$PY" -c "import torch" 2>/dev/null && echo "  torch is installed; the no-torch path is not exercised" \
    || echo "  torch is absent -- this is the case the script is for"

say "commands"
check_ok          "help"            bitsandbytes-cpu help
check_ok          "help 4bit"       bitsandbytes-cpu help 4bit
check_ok          "help 8bitopt"    bnb-cpu help 8bitopt
check_ok          "help --markdown" bitsandbytes-cpu help --markdown
check_ok          "detect"          bitsandbytes-cpu detect
check_ok          "doctor"          bitsandbytes-cpu doctor
check_ok          "version"         bitsandbytes-cpu version
check_fails_cleanly "help <unknown>" 1 bitsandbytes-cpu help nosuchsection

say "selftest"
out="$(bitsandbytes-cpu selftest 2>&1)"; rc=$?
if "$PY" -c "import torch" 2>/dev/null; then
    [ "$rc" -eq 0 ] && echo "  ok    selftest passed" || { echo "  FAIL  selftest exit=$rc"; fails=$((fails+1)); }
else
    # Without torch it must explain how to get torch, not raise a traceback.
    if [ "$rc" -ne 0 ] && ! printf '%s' "$out" | grep -q Traceback; then
        echo "  ok    selftest explains the missing torch instead of raising"
    else
        echo "  FAIL  selftest did not degrade cleanly (exit=$rc)"; fails=$((fails+1))
    fi
fi
printf '%s\n' "$out" | head -4 | sed 's/^/          /'

printf '\n'
if [ "$fails" -eq 0 ]; then
    echo "PASS: the CLI works on this machine"
else
    echo "FAIL: $fails check(s) failed"
fi
exit $((fails > 0))
