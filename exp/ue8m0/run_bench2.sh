#!/bin/bash
# Run the fair head-to-head benchmark (bnb's real kernel vs integer dot product).
set -e
SRC=/mnt/d/work/collab/_ue8m0
OUT=/tmp/bnb_ue8m0
mkdir -p "$OUT"
echo "=== compile ==="
g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 -o "$OUT/bench2" "$SRC/bench2.cpp" 2>&1 | head -30
echo "  ok"
echo
echo "########## single thread (isolates instruction effects) ##########"
for M in 1 2 4 8 64; do
    OMP_NUM_THREADS=1 "$OUT/bench2" $M 4096 4096 256
done
echo "########## 6 threads (realistic) ##########"
for M in 1 4 64; do
    OMP_NUM_THREADS=6 "$OUT/bench2" $M 4096 4096 256
done
echo "########## block size sensitivity (single thread, M=4) ##########"
for BK in 64 128 512 2048; do
    OMP_NUM_THREADS=1 "$OUT/bench2" 4 4096 4096 $BK
done
