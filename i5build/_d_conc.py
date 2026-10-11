# _d_conc.py — CPU vs iGPU vs 并发，交流电下，修正网格的 64 线程内核
import os, re, subprocess, sys, time
import numpy as np
import pyopencl as cl

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import igpu_repro as R

CPU_WORKER = r'''
import sys, time
import torch
torch.set_num_threads(6)
M, K, N, T = (int(x) for x in sys.argv[1:5])
a = torch.randn(M, K); b = torch.randn(K, N)
for _ in range(2): a @ b
print('CPU_READY', flush=True)
t0 = time.perf_counter(); n = 0
while time.perf_counter() - t0 < T:
    c = a @ b; n += 1
el = time.perf_counter() - t0
print('CPU_RESULT %.6f %d %.3f' % (el, n, 2.0*M*K*N*n/el/1e9), flush=True)
'''

def launch_cpu(M, K, N, secs):
    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': '6', 'MKL_NUM_THREADS': '6',
                'OMP_WAIT_POLICY': 'active', 'OMP_DYNAMIC': 'FALSE'})
    p = subprocess.Popen([sys.executable, '-c', CPU_WORKER, str(M), str(K), str(N), str(secs)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    line = p.stdout.readline()
    if 'CPU_READY' not in line:
        rest = p.stdout.read()
        raise RuntimeError('worker 未就绪: %r %r' % (line, rest))
    return p

def cpu_finish(p):
    out = p.stdout.read()
    p.wait(timeout=60)
    m = re.search(r'CPU_RESULT\s+([\d.]+)\s+(\d+)\s+([\d.]+)', out)
    if not m:
        raise RuntimeError('解析失败: %r' % out)
    return float(m.group(3))

def build_kernel(ctx, M, N, K):
    prog = cl.Program(ctx, R.GEMM_SRC).build(
        options=['-cl-fast-relaxed-math', '-cl-mad-enable', '-DKC=%d' % K, '-DNC=%d' % N])
    return cl.Kernel(prog, 'sgemm')

def make_bufs(ctx, M, N, K):
    mf = cl.mem_flags
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    ab = (cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
          cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    return ab, cb, a, b

def check(ctx, q, kern, M, N, K):
    ab, cb, a, b = make_bufs(ctx, M, N, K)
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    kern(q, gws, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cb)
    q.finish()
    o = np.empty((M, N), np.float32)
    cl.enqueue_copy(q, o, cb).wait()
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = float(np.abs(o - ref).max() / np.abs(ref).max())
    return err

def gpu_rate(q, kern, ab, cb, M, N, K, secs, batch=4):
    gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
    args = (np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cb)
    kern(q, gws, (8, 8), *args); q.finish()
    fl = 2.0 * M * N * K
    n = 0
    t0 = time.perf_counter()
    while True:
        for _ in range(batch):
            kern(q, gws, (8, 8), *args)
        q.finish()
        n += batch
        if time.perf_counter() - t0 >= secs:
            break
    el = time.perf_counter() - t0
    return fl * n / el / 1e9

def main():
    dev = cl.get_platforms()[0].get_devices()[0]
    ctx = cl.Context([dev]); q = cl.CommandQueue(ctx)
    ac, cpu = R.gpu_state()
    print('device: %s | 电源: %s | CPU%%: %s' % (dev.name, ac, cpu))
    T = 8.0
    for (M, K, N) in [(2048, 512, 2048), (2048, 2048, 512)]:
        print('\n########## M=%d K=%d N=%d ##########' % (M, K, N))
        kern = build_kernel(ctx, M, N, K)
        err = check(ctx, q, kern, M, N, K)
        print('正确性: 相对误差 %.2e  %s' % (err, 'OK' if err < 1e-4 else 'FAIL'))
        ab, cb, _, _ = make_bufs(ctx, M, N, K)
        for r in range(2):
            p = launch_cpu(M, K, N, int(T)); c_solo = cpu_finish(p)
            g_solo = gpu_rate(q, kern, ab, cb, M, N, K, T)
            p = launch_cpu(M, K, N, int(T) + 2)
            g_conc = gpu_rate(q, kern, ab, cb, M, N, K, T)
            c_conc = cpu_finish(p)
            comb = c_conc + g_conc
            print('  轮%d  CPU单独 %6.1f | 核显单独 %6.1f | 并发: CPU %6.1f + 核显 %6.1f = %6.1f'
                  % (r + 1, c_solo, g_solo, c_conc, g_conc, comb))
            print('        => 并发/CPU单独 = %.2fx | 并发/串行和(%.1f) = %.2f'
                  % (comb / c_solo, c_solo + g_solo, comb / (c_solo + g_solo)))

if __name__ == '__main__':
    main()
