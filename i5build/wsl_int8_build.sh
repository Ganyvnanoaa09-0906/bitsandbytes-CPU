#!/bin/bash
# Build and run the int8 GEMM benchmark on Linux, so the panel sizes can be swept and
# the model's own shapes added without needing MSVC on this machine.
#
# The comparison that matters is int8 (this kernel) against fp32 (oneDNN, which is what
# the model runs today). To make sure the Linux run is comparable at all, the first
# shape here is 4096^3, which the Windows build of the same kernel already measured.
set -e
cd /root
rm -rf int8bench && mkdir int8bench
cp /mnt/d/work/bitsandbytes-CPU/i5build/kernels/int8_gemm.c int8bench/
cd int8bench
g++ -O2 -mavx2 -mfma -fopenmp -o int8_gemm int8_gemm.c 2>&1 | head -20
echo "=== built; running the default shape set with 6 threads ==="
OMP_NUM_THREADS=6 ./int8_gemm 6
