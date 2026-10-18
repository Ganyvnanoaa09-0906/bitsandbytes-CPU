#!/usr/bin/env bash
# Run the portable (scalar-only) selftest under an emulated pre-AVX2 CPU.
set -uo pipefail
cd /tmp/qemu_local || exit 1
if [ ! -d root ]; then
  dpkg-deb -x qemu-user_*.deb root || { echo "EXTRACT_FAILED"; exit 1; }
fi
QEMU=root/usr/bin/qemu-x86_64
echo "qemu binary: $QEMU"
[ -x "$QEMU" ] || { echo "QEMU_MISSING"; exit 1; }
if ldd "$QEMU" 2>&1 | grep -q 'not found'; then
  echo "MISSING LIBS:"; ldd "$QEMU" | grep 'not found'
  exit 2
fi
echo "deps ok"

cd /mnt/d/work/bitsandbytes-CPU/bitsandbytes || exit 1
if [ ! -x build_linux/selftest_generic ]; then
  echo "building portable selftest ..."
  g++ -O2 -std=c++17 -fopenmp -march=x86-64 -DBNB_CPU_PORTABLE=1 \
      -DNOMINMAX -DNDEBUG -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 -I csrc \
      -x c++ selftest_cpu.c -x none csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \
      -o build_linux/selftest_generic || { echo "PORTABLE_BUILD_FAILED"; exit 3; }
fi

for cpu in Nehalem core2duo n270; do
  echo "--- qemu-x86_64 -cpu $cpu ---"
  /tmp/qemu_local/root/usr/bin/qemu-x86_64 -cpu "$cpu" ./build_linux/selftest_generic
  rc=$?
  echo "exit=$rc"
  if [ "$rc" -eq 0 ]; then
    echo "QEMU_PRE_AVX2_PASS cpu=$cpu"
    exit 0
  fi
done
echo "QEMU_PRE_AVX2_FAILED"
exit 4
