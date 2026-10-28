"""measure_lds_throughput.py -- the measurement that was missing.

Seven kernel variants failed to beat 300 GFLOPS against a 1089 GFLOPS pure-compute ceiling,
and two explanations I offered for the gap were both wrong. What I never did is measure the
thing all the variants depend on: the throughput of local memory itself.

The arithmetic that makes it worth checking. GCN's LDS is documented at 128 bytes per cycle
per CU. Six CUs at roughly 0.85 GHz is about 653 GB/s. The real inner loop reads 16 floats
per 64 FMAs, i.e. 0.25 reads per FMA at 4 bytes each, so sustaining the FMA rate needs

    FMA/s * 0.25 * 4 bytes

At the 1089 GFLOPS pure-compute rate that is 545e9 * 1 byte = 545 GB/s of LDS traffic --
83% of the theoretical 653. If the real LDS throughput is meaningfully below its nominal
figure, or if the access pattern costs extra cycles, the loop is LDS-bound and the 300
GFLOPS figure is explained without any further guessing.

Three patterns, all reading the same total bytes:

  strided1   every lane reads consecutive addresses -- the ideal conflict-free case
  broadcast  all lanes read one address -- also conflict-free, but wastes the bus
  strided8   lanes read addresses 8 floats apart -- the pattern the real kernel's B
             operand uses, which touches only 4 of 32 banks

Reported as GB/s and as the FMA rate each could sustain. If strided8 is the slowest, the
B access in the kernel is the problem and the fix is a layout change; if all three are
similar and near 653, then LDS is not the limit and the 300 GFLOPS figure needs another
explanation.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp   # noqa: E402

SRC = r'''
#define TS 64
#define TK 8
#define REPS 2048

// strided reads from local memory, one pattern per kernel; each does READ x REPS reads
__kernel void lds_pattern(const int mode, __global float* out) {
    __local float buf[4096];
    const int lid = get_local_id(0);
    const int nthreads = get_local_size(0);
    for (int i = lid; i < 4096; i += nthreads) buf[i] = (float)(i & 7);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int tx = lid & 7, ty = lid >> 3;
    float acc = 0.f;
    for (int r = 0; r < REPS; r++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int idx;
            if (mode == 0) {
                idx = (lid * 8 + j) & 4095;            // consecutive per lane
            } else if (mode == 1) {
                idx = (r + j) & 4095;                  // every lane the same address
            } else {
                idx = ((tx * 8 + j) * 64 + ty) & 4095; // 8 floats apart across lanes
            }
            acc += buf[idx];
        }
    }
    out[get_global_id(0)] = acc;
}
'''

assert_idle(allow=True, threshold=50.0)
stamp('LDS 吞吐（A/B 在跑；此处负载不是干扰，是背景）')
print()

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'lds_pattern')
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))

G = 6 * 256 * 8          # 48 workgroups of 256, several per CU
out = cl.Buffer(ctx, mf.WRITE_ONLY, G * 4)
REPS = 2048
THREADS = 256
READS = 8

print('每线程 %d 次内层读 × %d 轮 = %d 次读' % (READS, REPS, READS * REPS))
print()
print('%-12s %10s %12s %14s  %s' % ('访问模式', '秒', 'GB/s', '可支撑 FMA/秒', '说明'))
print('-' * 74)
names = {0: ('连续（理想）', '每 lane 连续地址'), 1: ('广播', '全 lane 同址'),
         2: ('跨 8（真实 B 模式）', '8 lane 触及 4 个 bank')}
res = {}
for mode in (0, 1, 2):
    k = kern
    k(queue, (G,), (THREADS,), np.int32(mode), out)
    queue.finish()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        k(queue, (G,), (THREADS,), np.int32(mode), out)
        queue.finish()
        ts.append(time.perf_counter() - t0)
    dt = statistics.median(ts)
    gb = G * READS * REPS * 4 / dt / 1e9
    # the real kernel needs 0.25 reads per FMA, i.e. 1 byte per FMA
    fma = gb * 1e9
    res[mode] = gb
    print('%-12s %10.4f %12.1f %14.3e  %s' % (names[mode][0], dt, gb, fma, names[mode][1]))
print()

THEO_GBS = 6 * 128 * 0.85e9 / 1e9
print('GCN 名义 LDS: 128 B/周期/CU × 6 CU × 0.85 GHz = %.0f GB/s' % THEO_GBS)
best = max(res.values())
print('实测最好（%s）: %.0f GB/s ⇒ 名义值的 %.0f%%'
      % (names[max(res, key=res.get)][0], best, 100 * best / THEO_GBS))
print()
need = 1089e9 / 2 * 0.25 * 4 / 1e9     # to sustain 1089 GFLOPS = 545e9 FMA/s
print('要维持 1089 GFLOPS 的纯计算速率，需要 %.0f GB/s 的 LDS 读' % need)
for mode in (0, 1, 2):
    print('  %-12s %6.0f GB/s ⇒ 只能支撑 %6.0f GFLOPS（纯计算的 %.2f×）'
          % (names[mode][0], res[mode], res[mode] * 2, res[mode] * 2 / 1089))
print()
print('判据: 若三种模式都远低于 545 GB/s ⇒ 这个内核结构【在 LDS 上就注定到不了高 GFLOPS】；')
print('      若都接近或超过 ⇒ LDS 不是限制，300 GFLOPS 另有原因。')
