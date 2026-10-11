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
    print('%-34s %7.3f ms   (%6.1f GFLOPS，对 CPU 22.6ms = %.2fx)' % (name, ms, FL / ms / 1e6, 22.6 / ms))

print('device:', dev.name)

# 参考: 纯设备内核
dA = cl.Buffer(ctx, mf.READ_ONLY, size=a.nbytes)
dB = cl.Buffer(ctx, mf.READ_ONLY, size=b.nbytes)
dC = cl.Buffer(ctx, mf.WRITE_ONLY, size=c.nbytes)
cl.enqueue_copy(q, dA, a); cl.enqueue_copy(q, dB, b); q.finish()
rep('A. 纯内核(设备常驻)', bench(lambda: (kern(q, gws, (8, 8), *ar(dA, dB, dC)), q.finish())))

# B. 权重常驻设备，只搬 A/C
rep('B. 权重常驻+显式拷贝A/C', bench(lambda: (cl.enqueue_copy(q, dA, a), kern(q, gws, (8, 8), *ar(dA, dB, dC)), cl.enqueue_copy(q, c, dC), q.finish())))

# C. 常驻 staging + USE_HOST_PTR
sa = np.empty_like(a); sb = np.empty_like(b); sc = np.empty_like(c)
hA = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=sa)
hB = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=sb)
hC = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=sc)
def c_use():
    sa[:] = a; sb[:] = b
    kern(q, gws, (8, 8), *ar(hA, hB, hC)); q.finish()
    c[:] = sc
rep('C. 常驻staging+USE_HOST_PTR', bench(c_use))

# D. mapped: ALLOC_HOST_PTR + 长期映射
try:
    mA = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=a.nbytes)
    mB = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=b.nbytes)
    mC = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=c.nbytes)
    pA = cl.enqueue_map_buffer(q, mA, cl.map_flags.READ_WRITE, 0, a.shape, a.dtype, order='C', is_blocking=True)
    pB = cl.enqueue_map_buffer(q, mB, cl.map_flags.READ_WRITE, 0, b.shape, b.dtype, order='C', is_blocking=True)
    pC = cl.enqueue_map_buffer(q, mC, cl.map_flags.READ_WRITE, 0, c.shape, c.dtype, order='C', is_blocking=True)
    print('  [映射对象类型] A:%s shape=%s' % (type(pA).__name__, getattr(pA, 'shape', '?')))
    pA[:] = a; pB[:] = b
    rep('D1. mapped 纯内核', bench(lambda: (kern(q, gws, (8, 8), *ar(mA, mB, mC)), q.finish())))
    def d2():
        pA[:] = a; pB[:] = b
        kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
        c[:] = pC
    rep('D2. mapped 全周期(CPU拷贝进出)', bench(d2))
    t = bench(lambda: pA.__setitem__(slice(None), a), reps=20)
    print('  CPU->mapped 4MB: %.3f ms (%.1f GB/s)' % (t, a.nbytes / t / 1e6))
    t = bench(lambda: c.__setitem__(slice(None), pC), reps=20)
    print('  mapped->CPU 4MB: %.3f ms (%.1f GB/s)' % (t, c.nbytes / t / 1e6))
except Exception as e:
    print('  D 映射路线失败: %r' % e)
