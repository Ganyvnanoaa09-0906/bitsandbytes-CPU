#!/bin/bash
# Careful single-thread measurement: pinned to one core, repeated, min reported by the bench.
# Multithreaded runs on this laptop are too noisy to compare (shared with other work), so the
# defensible numbers are the pinned single-thread ones.
set -e
OUT=/tmp/bnb_ue8m0
echo "########## pinned single-thread sweep ##########"
for N in 4096 11008; do
    for M in 1 2 4 8 64; do
        echo "--- M=$M N=$N K=4096 BK=256 ---"
        taskset -c 2 env OMP_NUM_THREADS=1 "$OUT/bench2" $M $N 4096 256 | grep -E 'bnb-u8|INT8 |speedup'
    done
done
echo
echo "########## block size sweep, pinned, M=4 N=4096 ##########"
for BK in 64 128 256 512 1024 2048; do
    echo "--- BK=$BK ---"
    taskset -c 2 env OMP_NUM_THREADS=1 "$OUT/bench2" 4 4096 4096 $BK | grep -E 'bnb-u8|INT8 |speedup'
done
