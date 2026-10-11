"""修微基准：链数扫描，判定这颗 Vega 6 的 packed fp16 到底有没有 2×。

上一版的毛病：fma16 只给了 8 条独立链，而 fp16 的 FMA 延迟可能 4~10 周期 ——
8 条正好卡在"刚好够/不够"的边界，测出来是延迟不是吞吐。

判定标准（用实测的 f32 频率当基准，不靠猜）：
    f32 实测 429 G-FMA/s ⇒ 6 CU × 64 lane × 1 FMA/周期 ⇒ 384 lane-op/周期 ⇒ 1.12 GHz
    · 若 packed 生效（每指令 2 次 fp16 FMA）：上限 = 2 × 429 = 858 G-fp16-FMA/s
    · 若没生效（每指令 1 次）              ：上限 = 429 G-fp16-FMA/s
  所以峰值的**落点**直接把答案给出来。

扫描 8/16/24/32 条链：吞吐随链数上升 ⇒ 之前测的是延迟；到达平台 ⇒ 那就是真吞吐。
"""
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

TEMPLATE = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void fma32_%(n)d(__global float* out, const int iters) {
%(decl32)s
    const float b = 0.99999f, c = 1e-7f;
    for (int i = 0; i < iters; i++) {
%(body32)s
    }
    out[get_global_id(0)] = %(sum32)s;
}
__kernel void fma16_%(n)d(__global float* out, const int iters) {
%(decl16)s
    const half2 b = (half2)(0.999h, 0.999h), c = (half2)(0.001h, 0.001h);
    for (int i = 0; i < iters; i++) {
%(body16)s
    }
    out[get_global_id(0)] = %(sum16)s;
}
'''


def gen(n):
    d32 = '\n'.join(f'    float f32_{i} = {1.0 + i * 0.01:.4f}f;' for i in range(n))
    b32 = '\n'.join(f'        f32_{i} = fma(f32_{i}, b, c);' for i in range(n))
    s32 = '+'.join(f'f32_{i}' for i in range(n))
    d16 = '\n'.join(f'    half2 h16_{i} = (half2)({1.0 + i * 0.01:.3f}h, '
                    f'{1.005 + i * 0.01:.3f}h);' for i in range(n))
    b16 = '\n'.join(f'        h16_{i} = fma(h16_{i}, b, c);' for i in range(n))
    s16 = '+'.join(f'(float)(h16_{i}.x + h16_{i}.y)' for i in range(n))
    return TEMPLATE % dict(n=n, decl32=d32, body32=b32, sum32=s32,
                           decl16=d16, body16=b16, sum16=s16)


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    print(f'device {dev.name} | {dev.max_compute_units} CU（机器空闲）')
    print(f"\n{'链数':>4} {'f32 G-FMA/s':>12} {'f16 G-FMA/s':>12} "
          f"{'f16/f32':>8} {'f16 若 packed':>13} {'f16 若未 packed':>15}")
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * 65536)
    nthreads, iters = 65536, 20000
    base = None
    for chains in (8, 16, 24, 32):
        prog = cl.Program(ctx, gen(chains)).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable'])
        row = []
        for tag, per_iter in (('fma32', chains), ('fma16', 2 * chains)):
            k = cl.Kernel(prog, f'{tag}_{chains}')
            k(q, (nthreads,), None, out, np.int32(iters))
            q.finish()
            t0 = time.perf_counter()
            k(q, (nthreads,), None, out, np.int32(iters))
            q.finish()
            dt = time.perf_counter() - t0
            row.append(per_iter * nthreads * iters / dt / 1e9)
        if base is None:
            base = row[0]
        mhz_p = row[1] / base * 1117 / 2      # 若 packed：每指令 2 次 FMA
        mhz_u = row[1] / base * 1117          # 若未 packed
        print(f'{chains:>4} {row[0]:>12.1f} {row[1]:>12.1f} '
              f'{row[1]/row[0]:>8.2f} {mhz_p:>11.0f} MHz {mhz_u:>12.0f} MHz')
    print(f"\n基准：f32 的 429 G-FMA/s 对应 1.12 GHz（实测）。"
          f"\n判读：f16/f32 ≈ 2.0 ⇒ packed 生效；≈ 1.0 ⇒ 没生效；介于之间 ⇒ 未打满。")


if __name__ == '__main__':
    sys.exit(main())
