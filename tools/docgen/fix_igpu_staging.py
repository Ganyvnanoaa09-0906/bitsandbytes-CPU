"""Fix the iGPU kernel's staging loops: they write 4x out of bounds.

Reading igpu_gemm_v4.py while looking for why 0.03% of the output elements are wrong, the cause
is visible in the staging code:

    // stage A: TS*TK = 1024 floats = 256 float4
    for (int p = 0; p < 4; p++) {
        int f4 = p * 256 + lane;      // lane is 0..255, so f4 reaches 1023
        int r = (f4 * 4) / TK;        // r reaches 255
        ...
        As[r][c] = ...;               // As is [64][20] -- rows 64..255 are out of bounds
    }

The tile is 64 rows x 16 columns = 1024 floats = 256 float4, and there are 256 threads. So it is
one iteration per thread, not four. As written, each thread writes four times as many elements as
exist, and the excess lands past the end of As -- into Bs, which is declared next to it. That
corrupts a handful of B values, which is exactly the signature: 914 wrong elements out of 2.8
million, not a crash and not widespread garbage.

The correct index is simply f4 = lane, with no loop. Same for B: TK*TS = 16*64 = 1024 floats =
256 float4, one per thread.

This is the same class of error as everything else this project keeps finding: the code ran, the
output looked mostly right, and the thing being measured was not the thing it claimed to be. A
kernel that computes 99.97% correct results is not a valid basis for measuring how fast a tiled
GEMM is.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\bitsandbytes-CPU\igpu_gemm_v4.py'
s = io.open(P, encoding='utf-8').read()
orig = s

# --- stage A
a_old = """        // stage A: TS*TK = 1024 floats = 256 float4
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f4 = p * 256 + lane;              // 0..1023 float4 index"""
a_new = """        // stage A: TS*TK = 1024 floats = 256 float4, one per thread.
        // The original had a 4-iteration loop with f4 = p*256 + lane, which reaches 1023 and
        // writes As[r][c] with r up to 255 into an array of 64 rows -- four times out of bounds,
        // landing on Bs. That is why 0.03% of outputs were wrong.
        {
            int f4 = lane;                        // 0..255 float4 index"""
if a_old in s:
    s = s.replace(a_old, a_new, 1)
    print('  + stage A 已修')
else:
    print('  ? stage A 锚点没找到')

# --- stage B
b_old = """        // stage B: TK*TS = 1024 floats = 256 float4
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f4 = p * 256 + lane;
            int k = (f4 * 4) / TS;"""
b_new = """        // stage B: same fix -- 1024 floats = 256 float4, one per thread.
        {
            int f4 = lane;
            int k = (f4 * 4) / TS;"""
if b_old in s:
    s = s.replace(b_old, b_new, 1)
    print('  + stage B 已修')
else:
    print('  ? stage B 锚点没找到')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  改动 %d 字节 -> %d 字节' % (len(orig), len(s)))
