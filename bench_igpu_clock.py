# -*- coding: utf-8 -*-
"""bench_igpu_clock.py — 测核显的【实际持续频率】

为什么需要：
  四个独立实现的 SGEMM 都卡在 250~300 GFLOPS（理论值 1150 的 22~26%）：
    DML 294.5 / CLBlast 278.6 / 自写调优 245.3 / 自写朴素 158.0
  这不正常 —— 好的 BLAS 应该到 60~70%。
  两种可能：
    (a) kernel 效率低（那我们还能优化）
    (b) 核显实际频率远低于 1.5 GHz（功耗墙，任何 kernel 优化都无用）

方法：
  纯 FMA 依赖链 kernel，16 路独立累加器（足以饱和 VALU，无访存）
  6 CU × 64 FMA/周期 = 384 FMA/周期 = 768 FLOP/周期
  ⇒ 反推: 时钟(GHz) = 实测 GFLOPS / 768
"""
import time
import numpy as np
import pyopencl as cl

SRC = r'''
#define ILP 16
__kernel void fma_peak(__global float* out, const int iters) {
    const int gid = get_global_id(0);
    float b = 1.0000001f;
    float c = 1e-12f;
    float a0 = 1.0f + gid * 1e-9f;
    float a1 = a0 + 0.01f,  a2 = a0 + 0.02f,  a3 = a0 + 0.03f;
    float a4 = a0 + 0.04f,  a5 = a0 + 0.05f,  a6 = a0 + 0.06f,  a7 = a0 + 0.07f;
    float a8 = a0 + 0.08f,  a9 = a0 + 0.09f,  aa = a0 + 0.10f,  ab = a0 + 0.11f;
    float ac = a0 + 0.12f,  ad = a0 + 0.13f,  ae = a0 + 0.14f,  af = a0 + 0.15f;
    for (int i = 0; i < iters; i++) {
        a0 = fma(a0, b, c);  a1 = fma(a1, b, c);  a2 = fma(a2, b, c);  a3 = fma(a3, b, c);
        a4 = fma(a4, b, c);  a5 = fma(a5, b, c);  a6 = fma(a6, b, c);  a7 = fma(a7, b, c);
        a8 = fma(a8, b, c);  a9 = fma(a9, b, c);  aa = fma(aa, b, c);  ab = fma(ab, b, c);
        ac = fma(ac, b, c);  ad = fma(ad, b, c);  ae = fma(ae, b, c);  af = fma(af, b, c);
    }
    out[gid] = a0+a1+a2+a3+a4+a5+a6+a7 + a8+a9+aa+ab+ac+ad+ae+af;
}
'''

# 访存受限的对照 kernel（每 FMA 都要读一次内存，用来确认访存墙的位置）
SRC_MEM = r'''
__kernel void mem_bw(__global float* out, __global const float* in, const int iters) {
    const int gid = get_global_id(0);
    const int n = get_global_size(0);
    float acc = 0.0f;
    for (int i = 0; i < iters; i++) acc += in[(gid + i * 7919) % n];
    out[gid] = acc;
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)

print('=' * 78)
print('核显实际持续频率探测')
print('=' * 78)
print('  设备:', dev.name, '|', dev.max_compute_units, 'CU')
print('  最大时钟频率（设备报告）:', dev.max_clock_frequency, 'MHz')
print()

prg = cl.Program(ctx, SRC).build()
prg2 = cl.Program(ctx, SRC_MEM).build()
fma_k = cl.Kernel(prg, 'fma_peak')
mem_k = cl.Kernel(prg2, 'mem_bw')

CU = dev.max_compute_units
LANES_PER_CU = 64
FLOP_PER_CYCLE = CU * LANES_PER_CU * 2      # FMA = 2 FLOP

# 预热核显（必须！否则数据偏低 78%）
print('  预热 20 秒...', flush=True)
N_WARM = 262144
buf_w = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N_WARM * 4)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    for _ in range(8):
        fma_k(queue, (N_WARM,), None, buf_w, np.int32(2000))
    queue.finish()
print('  预热完成\n')

print('  %-30s %12s %12s %10s' % ('测试', 'GFLOPS', '推算时钟', '占理论'))
print('  ' + '-' * 68)


def run_fma(N, iters, label):
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, N * 4)
    fma_k(queue, (N,), None, buf, np.int32(iters))     # 预热一次
    queue.finish()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        fma_k(queue, (N,), None, buf, np.int32(iters))
    queue.finish()
    dt = (time.perf_counter() - t0) / reps
    flops = float(N) * iters * 16 * 2
    gf = flops / dt / 1e9
    clk = gf / FLOP_PER_CYCLE * 1000.0    # MHz
    print('  %-30s %12.1f %10.0f MHz %9.1f%%' % (label, gf, clk, 100 * gf / 1150))
    return gf, clk


print('\n  --- 纯计算（16 路 ILP，无访存）---')
for N, it in ((65536, 20000), (262144, 20000), (262144, 80000), (524288, 40000)):
    run_fma(N, it, 'FMA N=%d iters=%d' % (N, it))

print('\n  --- 对照：理论参考 ---')
print('  %-30s %12.1f %10s MHz %9s' % ('若满频 1.5 GHz', 1150.0, '1500', '100%'))
print('  %-30s %12.1f %10s MHz %9.1f%%' % ('若 1.0 GHz', 768.0, '1000', 66.8))
print('  %-30s %12.1f %10s MHz %9.1f%%' % ('若 0.75 GHz', 576.0, '750', 50.1))
print('  %-30s %12.1f %10s MHz %9.1f%%' % ('若 0.5 GHz', 384.0, '500', 33.4))

print('\n  --- 对照：SGEMM 的实测 ---')
for name, gf in (('DML (AMD 官方)', 294.5), ('CLBlast', 278.6),
                 ('自写调优', 245.3), ('自写朴素', 158.0)):
    print('  %-30s %12.1f %10.0f MHz %9.1f%%' % (name, gf, gf / FLOP_PER_CYCLE * 1000, 100 * gf / 1150))

print()
print('  判读:')
print('    如果纯 FMA 的推算时钟也远低于 1500 MHz')
print('      ⇒ 【核显被功耗墙锁死】，SGEMM 的 250~300 已是真实上限')
print('    如果纯 FMA 能推到接近 1500')
print('      ⇒ 是 SGEMM kernel 的效率问题，还有空间')
print('DONE')
