#!/bin/bash
# Is the int8 kernel's advantage real, or is the benchmark's spread wider than the effect?
#
# The same shapes measured twice today disagreed by 28% (4096^3: 205 GFLOPS in one run,
# 262 in another). Any claim that int8 is 1.55x fp32 has to survive that spread first,
# so the kernel is run three times and the numbers are printed side by side.
cd /root/int8bench || exit 1
for i in 1 2 3; do
    echo "--- run $i ---"
    OMP_NUM_THREADS=6 ./int8_gemm 6 2>&1 | grep -E "M=4096|M=16384|blocked|plain" | sed 's/^/  /'
done
