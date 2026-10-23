#!/bin/bash
# Compile and run the ue8m0 vs float-dequant microbenchmark inside WSL.
set -e
SRC=/mnt/d/work/collab/_ue8m0
OUT=/tmp/bnb_ue8m0
mkdir -p "$OUT"
echo "=== compile ==="
g++ -O3 -mavx2 -mfma -std=c++17 -o "$OUT/bench" "$SRC/bench.cpp" 2>&1 | head -30
echo "  ok"
echo
for shape in "1 4096 4096 256" "8 4096 4096 256" "64 4096 4096 256" "1 11008 4096 256" "8 11008 4096 256"; do
    "$OUT/bench" $shape
done
