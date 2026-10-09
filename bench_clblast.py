# -*- coding: utf-8 -*-
"""bench_clblast.py — 用 CLBlast 测核显的【真实】SGEMM 上限

这是回答"核显到底有没有潜力"的决定性实验。

对照：
  理论峰值                1150 GFLOPS
  我自写 OpenCL 朴素      158.0  (14%)
  我自写 OpenCL 调优      245.3  (21%)
  DML (AMD 自己)          294.5  (26%)
  CPU 真实混合负载        221~241

如果 CLBlast 能到 500~700 GFLOPS，说明【核显有潜力，只是我们没写对 kernel】
如果 CLBlast 也只有 250~300，说明【这块核显就是这么快】，之前的结论成立
"""
import ctypes, os, sys, time
import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
print('=' * 76)
print('CLBlast SGEMM 基准 —— 核显真实上限')
print('=' * 76)
print('  DLL:', DLL)
if not os.path.exists(DLL):
    raise SystemExit('找不到 clblast.dll')

bl = ctypes.CDLL(DLL)
# CLBlastStatusCode CLBlastSgemm(layout, a_trans, b_trans, m, n, k, alpha,
#     a_buf, a_off, a_ld, b_buf, b_off, b_ld, beta, c_buf, c_off, c_ld,
#     queue*, event*)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
LAYOUT_ROW, TRANSPOSE_NO = 1, 111

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
print('  设备:', dev.name, '|', dev.max_compute_units, 'CU')
print()


def bench(m, n, k, iters=6, warm=3, label=''):
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    mf = cl.mem_flags
    a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
    qp = ctypes.c_void_p(int(queue.int_ptr))
    args = (LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO,
            m, n, k, ctypes.c_float(1.0),
            ctypes.c_void_p(int(a.int_ptr)), 0, k,
            ctypes.c_void_p(int(b.int_ptr)), 0, n,
            ctypes.c_float(0.0),
            ctypes.c_void_p(int(c.int_ptr)), 0, n,
            ctypes.byref(qp), None)

    def one():
        rc = bl.CLBlastSgemm(*args)
        if rc != 0:
            raise RuntimeError('CLBlast 错误码 %d' % rc)
        queue.finish()

    for _ in range(warm):
        one()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    dt = (time.perf_counter() - t0) / iters
    gf = 2.0 * m * n * k / dt / 1e9
    print('  %-22s m=%-5d n=%-5d k=%-5d  %8.2f ms  【%7.1f GFLOPS】  %.1f%%'
          % (label or '', m, n, k, dt * 1e3, gf, 100 * gf / 1150))
    return gf


print('  提示：核显必须先预热 20 秒以上，否则数据偏低 78%')
print('  预热中...', flush=True)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    bench(512, 512, 512, iters=4, warm=1)
print('  预热完成')
print()

print('  %-22s %-27s %10s %10s' % ('形状', '', '耗时', 'GFLOPS'))
print('  ' + '-' * 66)
res = {}
for n in (256, 512, 1024, 2048):
    res[n] = bench(n, n, n, iters=6, warm=3, label='正方 %d³' % n)

# FFN 的真实形状：2048x512 @ 512x2048
res['w1'] = bench(2048, 2048, 512, iters=6, warm=3, label='FFN w1 2048x512@512x2048')
res['w2'] = bench(2048, 512, 2048, iters=6, warm=3, label='FFN w2 2048x2048@2048x512')
# 注意力形状
res['attn'] = bench(2048, 2048, 64, iters=6, warm=3, label='attn 2048x64@64x2048')

print()
print('=' * 76)
print('对照表')
print('=' * 76)
print('  理论峰值               1150.0 GFLOPS')
print('  我自写 OpenCL 朴素      158.0 (14%)')
print('  我自写 OpenCL 调优      245.3 (21%)')
print('  DML (AMD 自己)          294.5 (26%)')
print('  CPU 真实混合负载        221~241')
best = max(v for v in res.values() if isinstance(v, float))
print('  【CLBlast 最佳        %8.1f GFLOPS  (%.1f%%)】' % (best, 100 * best / 1150))
print()
if best > 450:
    print('  ⇒ 核显【有潜力】，之前是我们的 kernel 不行')
elif best > 320:
    print('  ⇒ 核显比我们手写的强，但仍远低于理论值')
else:
    print('  ⇒ 核显就是这么快，"没潜力"的结论成立')
print('DONE')
