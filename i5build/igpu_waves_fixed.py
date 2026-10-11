"""修正后的 wave 曲线（收回上一版结论后重测）。

上一版的三处错误：
  ① 网格 `ceil(N/64)*8` 是**按 8 宽局部尺寸**算的；局部尺寸换成 16 后工作组数变成
     4×4=16，只覆盖 4×4 个 tile，**输出只算了 1/4**。
  ② 而 GFLOPS 是按【完整 GEMM 的 FLOPs】除的 ⇒ 虚高最多 4 倍。
  ③ 正确性检查没抓到，因为三个变体**共用同一个 C 缓冲区**：第一个变体（网格正确）
     写入完整正确答案，后面覆盖不足的变体只覆盖一部分，**其余仍是前一个变体的正确值**
     ⇒ 逐位比对通过。**测试测到了自己。**

修正协议：
  · 网格按局部尺寸算：gws = (ceil(N/64)*lsx, ceil(M/64)*lsy)
  · **每个变体独立的 C 缓冲区**，且每次运行前显式清零
  · 断言覆盖完整（把整个输出与参考比，而不是抽样）
"""
import os
import shutil
import sys
import time

import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import igpu_waves as W  # noqa: E402  （复用已验证的内核源码）

BASE = os.path.join(HERE, '_waves_fix')


def build(ctx, tag, src, K, N):
    d = os.path.join(BASE, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(d)
    try:
        return cl.Program(ctx, src).build(
            options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                     f'-DKC={K}', f'-DNC={N}'])
    finally:
        os.chdir(cwd)


VARIANTS = [('v64', W.K64, (8, 8), 1, 7),
            ('v128', W.K128, (16, 8), 2, 14),
            ('v256', W.K256, (16, 16), 4, 28)]


def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    M = N = K = 512
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    ref = a @ b
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
    zero = np.zeros(M * N, np.float32)

    print('=== 正确性（每变体独立缓冲区 + 运行前清零 + 全量比对）===')
    progs, errs = {}, {}
    for name, src, ws, waves, waveN in VARIANTS:
        progs[name] = build(ctx, name, src, K, N)
        cb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=zero)
        gws = (int(np.ceil(N / 64)) * ws[0], int(np.ceil(M / 64)) * ws[1])
        cl.Kernel(progs[name], name)(q, gws, ws, np.int32(M), np.int32(N),
                                     np.int32(K), ab[0], ab[1], cb)
        q.finish()
        o = np.empty((M, N), np.float32)
        cl.enqueue_copy(q, o, cb).wait()
        nz = float((o != 0).mean())
        e = float(np.abs(o - ref).max() / np.abs(ref).max())
        errs[name] = e
        print(f'   {name:<6} ws={ws} gws={gws} 非零 {nz*100:5.1f}%  '
              f'相对误差 {e:.2e}  {"OK" if e < 1e-4 else "❌ 覆盖不足或算错"}')

    print('\n=== 计时（网格正确）===')
    print(f"{'形状':>20} {'7 wave':>9} {'14 wave':>9} {'28 wave':>9}   "
          f"{'7→14':>7} {'14→28':>7}")
    for M2, N2, K2 in ((512, 512, 512), (1024, 1024, 1024), (2048, 512, 512),
                       (2048, 2048, 512), (2048, 512, 2048)):
        a2 = np.random.rand(M2, K2).astype(np.float32)
        b2 = np.random.rand(K2, N2).astype(np.float32)
        fl = 2.0 * M2 * N2 * K2
        ab2 = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2),
               cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b2))
        r, ok = {}, {}
        for name, src, ws, _w, _wn in VARIANTS:
            prog = build(ctx, f'{name}_{M2}_{K2}', src, K2, N2)
            cb = cl.Buffer(ctx, mf.READ_WRITE,
                           size=int(M2) * int(N2) * 4)   # 按实际形状分配，别用切片（上一版切片不够长 ⇒ INVALID_VALUE）
            gws = (int(np.ceil(N2 / 64)) * ws[0], int(np.ceil(M2 / 64)) * ws[1])
            kk = cl.Kernel(prog, name)
            args = (np.int32(M2), np.int32(N2), np.int32(K2), ab2[0], ab2[1], cb)
            kk(q, gws, ws, *args)
            q.finish()
            o = np.empty((M2, N2), np.float32)
            cl.enqueue_copy(q, o, cb).wait()
            ok[name] = float(np.abs(o - a2 @ b2).max() / np.abs(a2 @ b2).max()) < 1e-4
            t0 = time.perf_counter()
            for _ in range(5):
                kk(q, gws, ws, *args)
            q.finish()
            r[name] = fl / ((time.perf_counter() - t0) / 5) / 1e9
        flag = '' if all(ok.values()) else '  ⚠️ 有变体未通过正确性'
        print(f'{M2:>8}×{N2:<5}×{K2:<4} {r["v64"]:>9.1f} {r["v128"]:>9.1f} '
              f'{r["v256"]:>9.1f}   {r["v128"]/r["v64"]:>6.2f}× '
              f'{r["v256"]/r["v128"]:>6.2f}×{flag}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
