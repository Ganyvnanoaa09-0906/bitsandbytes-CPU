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
ar = lambda ab, bb, cb: (np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
FL = 2.0 * M * N * K

def bench(fn, reps=10, warm=2):
    for _ in range(warm): fn()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t0) / reps * 1e3

print('device:', dev.name)
t = bench(lambda: q.finish(), reps=50)
print('空 finish: %.3f ms' % t)

def cyc_use():
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=c)
    kern(q, gws, (8, 8), *ar(ab, bb, cb)); q.finish()
t = bench(cyc_use)
print('A. USE_HOST_PTR 新建+跑      : %7.3f ms  (%6.1f GFLOPS)' % (t, FL / t / 1e6))

def cyc_copy():
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, size=c.nbytes)
    kern(q, gws, (8, 8), *ar(ab, bb, cb))
    cl.enqueue_copy(q, c, cb); q.finish()
t = bench(cyc_copy)
print('B. COPY_HOST_PTR 新建+跑+读回: %7.3f ms  (%6.1f GFLOPS)' % (t, FL / t / 1e6))

ab = cl.Buffer(ctx, mf.READ_ONLY, size=a.nbytes)
bb = cl.Buffer(ctx, mf.READ_ONLY, size=b.nbytes)
cb = cl.Buffer(ctx, mf.WRITE_ONLY, size=c.nbytes)
def cyc_persist():
    cl.enqueue_copy(q, ab, a)
    cl.enqueue_copy(q, bb, b)
    kern(q, gws, (8, 8), *ar(ab, bb, cb))
    cl.enqueue_copy(q, c, cb); q.finish()
t = bench(cyc_persist)
print('C. 常驻+显式拷贝+读回       : %7.3f ms  (%6.1f GFLOPS)' % (t, FL / t / 1e6))

def cyc_pure():
    kern(q, gws, (8, 8), *ar(ab, bb, cb)); q.finish()
t = bench(cyc_pure, reps=20)
print('D. 常驻纯内核+finish        : %7.3f ms  (%6.1f GFLOPS)' % (t, FL / t / 1e6))

# 单独量三种拷贝
t1 = bench(lambda: cl.enqueue_copy(q, ab, a), reps=20)
t2 = bench(lambda: cl.enqueue_copy(q, c, cb), reps=20)
print('\nH2D 4MB: %.3f ms (%.1f GB/s) | D2H 4MB: %.3f ms (%.1f GB/s)' % (
    t1, a.nbytes / t1 / 1e6, t2, c.nbytes / t2 / 1e6))
