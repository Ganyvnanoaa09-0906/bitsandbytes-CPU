"""判定：GEMM 停在 FMA 峰值 35~40%，是【硬件】还是【结构】？

设计一个和 GEMM 内循环【完全同形】的循环，但把加载全删掉：
    · 64 个累加器（8×8，和 TM=TN=8 一样）
    · 16 个操作数寄存器（8 个 a + 8 个 b，只从 global 读一次）
    · 每轮迭代 64 条 FMA，访存为零、LDS 为零、barrier 为零

于是三种可能，各自指向不同结论：
    · 若它也停在 ~180 G-FMA/s（GEMM 的水平）⇒ 480 那个峰值对这种寄存器压力不适用，
      瓶颈在寄存器文件/FMA 端口本身，**换结构没用**
    · 若它能到 ~430（和简单链一样）⇒ 阻塞纯粹来自【喂数据】，GEMM 还有大空间
    · 中间值 ⇒ 部分是寄存器压力、部分是喂数据

对照：之前的纯链探针（12 条链、12 个累加器）是 429~500 G-FMA/s。
"""
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')

SRC = r'''
// 12 条独立链（对照，已知 ~430 G-FMA/s）
__kernel void chain(__global float* out, const int iters) {
    float a0=1.0f,a1=1.01f,a2=1.02f,a3=1.03f,a4=1.04f,a5=1.05f;
    float a6=1.06f,a7=1.07f,a8=1.08f,a9=1.09f,a10=1.1f,a11=1.11f;
    const float b = 0.99999f, c = 1e-7f;
    for (int i = 0; i < iters; i++) {
        a0=fma(a0,b,c);  a1=fma(a1,b,c);  a2=fma(a2,b,c);  a3=fma(a3,b,c);
        a4=fma(a4,b,c);  a5=fma(a5,b,c);  a6=fma(a6,b,c);  a7=fma(a7,b,c);
        a8=fma(a8,b,c);  a9=fma(a9,b,c);  a10=fma(a10,b,c); a11=fma(a11,b,c);
    }
    out[get_global_id(0)] = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11;
}

// GEMM 内循环同形：64 个累加器 + 16 个操作数，每轮 64 条 FMA，零访存
__kernel void gemm_shape(__global float* out, const int iters) {
    float a[8], b[8], acc[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a[i] = 1.0f + i * 0.01f;
        b[i] = 0.99f + i * 0.001f;
    }
    #pragma unroll
    for (int i = 0; i < 64; i++) acc[i] = 0.0f;
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++)
                acc[i*8+j] = fma(a[i], b[j], acc[i*8+j]);
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; i++) s += acc[i];
    out[get_global_id(0)] = s;
}
'''


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, SRC).build(options=['-cl-fast-relaxed-math',
                                               '-cl-mad-enable'])
    out = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 4 * 65536)
    n = 65536
    print(f'device {dev.name} | {dev.max_compute_units} CU（机器空闲）')
    print(f"\n{'内核':<14} {'FMA/轮':>8} {'iters':>10} {'ms':>9} {'G-FMA/s':>10} "
          f"{'占 480 峰值':>11}")
    # 让两个内核的总 FMA 数可比
    for name, per_iter, iters in (('chain', 12, 20000), ('gemm_shape', 64, 3750)):
        k = cl.Kernel(prog, name)
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        t0 = time.perf_counter()
        k(q, (n,), None, out, np.int32(iters))
        q.finish()
        dt = time.perf_counter() - t0
        rate = per_iter * n * iters / dt / 1e9
        print(f'{name:<14} {per_iter:>8} {iters:>10} {dt*1e3:>9.2f} {rate:>10.1f} '
              f'{rate/480*100:>10.1f}%')
    print('\n  chain      = 12 个累加器（寄存器压力极小）')
    print('  gemm_shape = 64 个累加器 + 16 操作数（与 GEMM 内循环同形，零访存）')
    print('  GEMM 实测 = 313~370 GFLOPS = 157~185 G-FMA/s（占 480 的 33~39%）')


if __name__ == '__main__':
    sys.exit(main())
