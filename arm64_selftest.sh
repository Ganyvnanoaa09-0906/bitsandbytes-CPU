#!/usr/bin/env bash
# Run the ARM64 static (musl) selftest under qemu-aarch64. No root: the qemu-user .deb is extracted
# into /tmp by this script; the test binary is self-contained (static musl).
set -uo pipefail
cd /tmp || exit 1
QDIR=/tmp/qemu_arm64_$$ 
mkdir -p "$QDIR" && cd "$QDIR" || exit 1
echo "downloading qemu-user ..."
apt-get download qemu-user >/dev/null 2>&1
dpkg-deb -x qemu-user_*.deb root >/dev/null 2>&1
QEMU=root/usr/bin/qemu-aarch64
[ -x "$QEMU" ] || { echo "QEMU_AARCH64_MISSING"; exit 1; }
echo "running ARM64 selftest under qemu-aarch64 ..."
"$QEMU" /mnt/d/work/bitsandbytes-CPU/build_arm64/selftest_arm64_musl
rc=$?
echo "exit=$rc"
if [ "$rc" -eq 0 ]; then echo "ARM64_SELFTEST_PASS"; else echo "ARM64_SELFTEST_FAIL"; fi
exit $rc
