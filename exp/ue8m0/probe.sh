#!/bin/bash
# Environment probe for the ue8m0 experiment (run inside WSL).
echo "=== g++ ==="
g++ --version | head -2
echo
echo "=== CPU ==="
nproc
grep -m1 'model name' /proc/cpuinfo
echo
echo "=== ISA flags ==="
for f in avx2 fma avx512f avx512vnni avx512bw; do
    if grep -qw "$f" /proc/cpuinfo; then echo "  $f: yes"; else echo "  $f: no"; fi
done
echo
echo "=== uname ==="
uname -a
