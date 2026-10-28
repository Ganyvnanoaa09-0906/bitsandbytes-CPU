"""Is it power arbitration, or is it the package being awake? A zero-risk separation.

7.62 measured that the iGPU drops 5.1x the moment one CPU thread works and concluded power arbitration:
the SMU feeds the CPU and starves the GPU out of a shared 15 W budget. That conclusion came from the
SHAPE of the curve -- a cliff, not a slope -- and a cliff fits more than one mechanism.

The alternative is package state. If a core simply being active keeps the package out of the state the
iGPU needs to boost, then how much power the CPU draws is irrelevant, and the remedy is not to negotiate
for watts but to keep the CPU's consumption negligible.

The two differ enormously in what can be done about them. A power limit is a hardware fact; a
package-state effect might be avoidable in software, with no hardware risk and no argument with the EC.
Which matters, because the tempting alternative -- raising the TDP limit on a laptop whose BIOS exposes
neither temperature nor power, on a board whose product line has documented power-stage failures -- is
exactly the kind of blind change this project keeps finding reasons not to make.

Method: run the iGPU GEMM repeatedly while an external process burns the CPU at a chosen duty cycle.

    idle                      nothing running
    1 thread, always busy     one core at 100%, maximum power per core
    1 thread, 5% duty         1 ms busy then 19 ms asleep -- roughly 2-4% of that core's power,
                              while the core stays resident and the scheduler cannot idle the package
    6 threads, always busy    the 7.62 condition
    1 thread, memory-bound    a different power profile: lots of traffic, little FP

The 5% case is the discriminator. If it drags the iGPU down like the fully busy core does, power is not
the variable. If the iGPU stays near its idle rate while the duty cycle is low even though a core is
resident and awake, then the variable is power after all, and 7.62 stands.

Criterion, fixed in advance: if the 5% duty cycle keeps the iGPU within 15% of its idle rate, the
mechanism is power; if it loses more than half the iGPU's throughput like the busy core, the mechanism
is the package being awake.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl
from scripts.lib.measure_guard import assert_idle, stamp

SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
q = cl.CommandQueue(ctx)
mf = cl.mem_flags
M, N, K = 2048, 2736, 512
FLOP = 2.0 * M * N * K
rng = np.random.default_rng(5)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')
gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)


def gpu_call():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    q.finish()


def gpu_steady(reps=6):
    for _ in range(2):
        gpu_call()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu_call()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


assert_idle()
stamp('是功耗仲裁,还是"CPU 醒着"本身')
print()
print('  %-30s %12s %14s' % ('条件', '核显 ms', 'GFLOPS'))
print('  ' + '-' * 60)

cases = [
    ('(a) CPU 完全空闲', None),
    ('(c) 1 线程 5% 占空比', (1, 1000, 19000)),
    ('(c2) 1 线程 1% 占空比', (1, 200, 19800)),
    ('(b) 1 线程满转', (1, 1000000000, 0)),
    ('(d) 6 线程满转', (6, 1000000000, 0)),
]
res = {}
for name, spec in cases:
    proc = None
    if spec:
        nt, on, off = spec
        proc = subprocess.Popen([r'D:\work\bitsandbytes-CPU\cpu_burn.exe',
                                 str(nt), str(on), str(off), '60'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)          # let the state settle before measuring
    t = gpu_steady()
    res[name] = t
    print('  %-30s %12.3f %14.1f' % (name, t * 1e3, FLOP / t / 1e9))
    if proc:
        proc.terminate()
        proc.wait()
        time.sleep(1.5)

print()
idle = res['(a) CPU 完全空闲']
d5 = res['(c) 1 线程 5% 占空比']
d1 = res['(c2) 1 线程 1% 占空比']
busy = res['(b) 1 线程满转']
print('  相对空闲的倍数:')
for k in ('(c2) 1 线程 1% 占空比', '(c) 1 线程 5% 占空比', '(b) 1 线程满转', '(d) 6 线程满转'):
    print('    %-30s %.2f×' % (k, res[k] / idle))
print()
print('  判据（事先固定）:')
print('    5%% 占空比把核显保持在空闲的 15%% 以内 ⇒ 【是功耗】,7.62 成立')
print('    5%% 占空比丢掉了超过一半核显吞吐   ⇒ ★【是"CPU 醒着"本身】,7.62 要改')
print()
if d5 <= idle * 1.15:
    print('  ⇒ 【是功耗】✓ 7.62 的结论成立 —— 而这也意味着它对游戏场景确实是坏策略')
elif d5 >= idle * 1.5:
    print('  ⇒ ★【是"CPU 醒着"本身】✓ —— 而这是好消息:')
    print('     ★【问题不是"用了多少电",而是封装状态 ⇒ 可能【不需要拉功率】就能解决】')
else:
    print('  ⇒ 中间地带,两种机制都有贡献')
