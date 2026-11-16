#!/bin/bash
# Re-check the anomalous 6-thread M=4 case and extend the shape sweep.
set -e
OUT=/tmp/bnb_ue8m0
echo "=== 6 threads, M=4, repeated 5x (noise check) ==="
for i in 1 2 3 4 5; do
    OMP_NUM_THREADS=6 "$OUT/bench2" 4 4096 4096 256 | grep -E 'bnb-u8|INT8 |speedup'
    echo "  --"
done
echo
echo "=== 6 threads, M=8 / M=16 / M=32 ==="
for M in 8 16 32; do
    OMP_NUM_THREADS=6 "$OUT/bench2" $M 4096 4096 256
done
echo
echo "=== 6 threads, decode-shaped MLP  N=11008 ==="
for M in 1 4; do
    OMP_NUM_THREADS=6 "$OUT/bench2" $M 11008 4096 256
done
