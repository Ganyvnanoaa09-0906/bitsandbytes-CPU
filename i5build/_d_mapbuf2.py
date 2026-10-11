import os, sys, time
import numpy as np, pyopencl as cl
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\i5build')
import igpu_repro as R

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([dev]); q = cl.CommandQueue(ctx)
M, N, K = 2048, 2048, 512
prog = cl.Program(ctx, R.GEMM_SRC).build(
    options=['-cl-fast-relaxed-math', '-cl-mad-enable', '-DKC=%d' % K, '-DNC=%d' % N])
kern = cl.Kernel(prog, 'sgemm')
gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
mf = cl.mem_flags
a = np.random.rand(M, K).astype(np.float32)
b = np.random.rand(K, N).astype(np.float32)
c = np.empty((M, N), np.float32)
ar = lambda x, y, z: (np.int32(M), np.int32(N), np.int32(K), x, y, z)
FL = 2.0 * M * N * K

def bench(fn, reps=10, warm=2):
    for _ in range(warm): fn()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1e3

def rep(name, ms):
    print('%-36s %7.3f ms   (%6.1f GFLOPS，对 CPU 22.6ms = %.2fx)' % (name, ms, FL / ms / 1e6, 22.6 / ms))

print('device:', dev.name)
print('map_flags 可用属性:', [x for x in dir(cl.map_flags) if not x.startswith('_')])

mA = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=a.nbytes)
mB = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=b.nbytes)
mC = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=c.nbytes)
fl = cl.map_flags.READ | cl.map_flags.WRITE
pA = cl.enqueue_map_buffer(q, mA, fl, 0, a.shape, a.dtype, order='C', is_blocking=True)
pB = cl.enqueue_map_buffer(q, mB, fl, 0, b.shape, b.dtype, order='C', is_blocking=True)
pC = cl.enqueue_map_buffer(q, mC, fl, 0, c.shape, c.dtype, order='C', is_blocking=True)
print('映射对象类型:', type(pA).__name__, '| shape:', getattr(pA, 'shape', '?'))

ok_set = True
try:
    pA[:] = a; pB[:] = b
except Exception as e:
    print('直接赋值失败(%r)，改用 np.asarray' % e)
    ok_set = False
    np.asarray(pA)[:] = a
    np.asarray(pB)[:] = b

kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
try:
    c[:] = pC
except Exception:
    c[:] = np.asarray(pC)
ref = a.astype(np.float64) @ b.astype(np.float64)
err = float(np.abs(c - ref).max() / np.abs(ref).max())
print('正确性: 相对误差 %.2e  %s' % (err, 'OK' if err < 1e-4 else 'FAIL'))

rep('D1. mapped 纯内核(零拷贝)', bench(lambda: (kern(q, gws, (8, 8), *ar(mA, mB, mC)), q.finish())))

if ok_set:
    def d2():
        pA[:] = a; pB[:] = b
        kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
        c[:] = pC
else:
    def d2():
        np.asarray(pA)[:] = a; np.asarray(pB)[:] = b
        kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
        c[:] = np.asarray(pC)
rep('D2. mapped 全周期(CPU拷贝进出)', bench(d2))

t = bench(lambda: (pA.__setitem__(slice(None), a) if ok_set else np.asarray(pA).__setitem__(slice(None), a)), reps=20)
print('CPU->mapped 4MB: %.3f ms (%.1f GB/s)' % (t, a.nbytes / t / 1e6))
t = bench(lambda: (c.__setitem__(slice(None), pC) if ok_set else np.copyto(c, np.asarray(pC))), reps=20)
print('mapped->CPU 4MB: %.3f ms (%.1f GB/s)' % (t, c.nbytes / t / 1e6))
