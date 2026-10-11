import os, sys, time, ctypes
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

print('device:', dev.name)

# ---------- C 模式拆解 ----------
sa = np.empty_like(a); sb = np.empty_like(b); sc = np.empty_like(c)
hA = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=sa)
hB = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=sb)
hC = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=sc)
t_cin = bench(lambda: (sa.__setitem__(slice(None), a), sb.__setitem__(slice(None), b)))
t_k = bench(lambda: (kern(q, gws, (8, 8), *ar(hA, hB, hC)), q.finish()))
t_cout = bench(lambda: c.__setitem__(slice(None), sc))
print('C 拆解: CPU拷入 2x4MB %.3f ms | 内核(host驻留) %.3f ms | CPU拷出 4MB %.3f ms | 合计 %.3f' % (
    t_cin, t_k, t_cout, t_cin + t_k + t_cout))

# ---------- D 映射探测 ----------
print('\n--- 映射探测 ---')
print('Buffer 的 map 相关属性:', [x for x in dir(hA) if 'map' in x.lower()])
mA = cl.Buffer(ctx, mf.ALLOC_HOST_PTR | mf.READ_WRITE, size=a.nbytes)
fl = cl.map_flags.READ | cl.map_flags.WRITE
res = cl.enqueue_map_buffer(q, mA, fl, 0, a.shape, a.dtype, order='C', is_blocking=True)
print('enqueue_map_buffer 返回:', type(res), '长度', len(res) if isinstance(res, tuple) else '-')
for i, x in enumerate(res if isinstance(res, tuple) else [res]):
    print('  [%d] type=%s shape=%s' % (i, type(x).__name__, getattr(x, 'shape', None)))
    print('      属性样本:', [s for s in dir(x) if not s.startswith('_')][:25])

obj = res[0] if isinstance(res, tuple) else res
view = None
for attempt in ('asarray', 'frombuffer', 'base_ptr'):
    try:
        if attempt == 'asarray':
            view = np.asarray(obj)
        elif attempt == 'frombuffer':
            view = np.frombuffer(obj, dtype=a.dtype, count=a.size).reshape(a.shape)
        else:
            base = getattr(obj, 'base', None)
            if base is None:
                continue
            ct = (ctypes.c_char * a.nbytes).from_address(ctypes.addressof(base))
            view = np.frombuffer(ct, dtype=a.dtype).reshape(a.shape)
        if view is not None and view.size == a.size:
            print('  ⇒ 视图法成功: %s -> shape %s' % (attempt, view.shape))
            break
        else:
            view = None
    except Exception as e:
        print('  %s 失败: %r' % (attempt, e))
        view = None

if view is not None:
    mB = cl.Buffer(ctx, mf.ALLOC_HOST_PTR | mf.READ_WRITE, size=b.nbytes)
    mC = cl.Buffer(ctx, mf.ALLOC_HOST_PTR | mf.READ_WRITE, size=c.nbytes)
    resB = cl.enqueue_map_buffer(q, mB, fl, 0, b.shape, b.dtype, order='C', is_blocking=True)
    resC = cl.enqueue_map_buffer(q, mC, fl, 0, c.shape, c.dtype, order='C', is_blocking=True)
    vB = np.frombuffer(resB[0], dtype=b.dtype, count=b.size).reshape(b.shape) if False else np.asarray(resB[0])
    try:
        vB = np.asarray(resB[0]); vC = np.asarray(resC[0])
    except Exception:
        vB = np.frombuffer(resB[0], dtype=b.dtype, count=b.size).reshape(b.shape)
        vC = np.frombuffer(resC[0], dtype=c.dtype, count=c.size).reshape(c.shape)
    view[:] = a; vB[:] = b
    kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
    c[:] = vC
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = float(np.abs(c - ref).max() / np.abs(ref).max())
    print('  mapped 正确性: %.2e %s' % (err, 'OK' if err < 1e-4 else 'FAIL'))
    print('  mapped 纯内核: %.3f ms (%6.1f GFLOPS, 对CPU %.2fx)' % (
        bench(lambda: (kern(q, gws, (8, 8), *ar(mA, mB, mC)), q.finish())), FL / bench(lambda: (kern(q, gws, (8, 8), *ar(mA, mB, mC)), q.finish())) / 1e6, 22.6 / bench(lambda: (kern(q, gws, (8, 8), *ar(mA, mB, mC)), q.finish()))))
    def d2():
        view[:] = a; vB[:] = b
        kern(q, gws, (8, 8), *ar(mA, mB, mC)); q.finish()
        c.__setitem__(slice(None), vC)
    td = bench(d2)
    print('  mapped 全周期: %.3f ms (%6.1f GFLOPS, 对CPU %.2fx)' % (td, FL / td / 1e6, 22.6 / td))
