"""How big a fake number can this machine produce? The recipe, measured.

The user asked when a measurement will be wrong enough to brag about 900 TFLOPS. The answer is that it
already was: the enqueue-time mistake produced 46787 GFLOPS, which is 40x the iGPU's rated peak. The
question is only how much bigger it can get.

The mechanism: an OpenCL kernel call is asynchronous. It returns once the command is enqueued, in
microseconds, while the kernel itself runs for milliseconds. Timing the call without waiting measures
the enqueue. Divide a GEMM's FLOPs by that and the arithmetic is real, the numerator is real, and the
denominator belongs to a different event.

This measures both numbers for the same kernels, so the ratio is visible as a ratio rather than
asserted. It uses the hand-written kernel rather than CLBlast for one reason: CLBlast returns -1011 on
the probe shapes and this is meant to be a working demonstration, not another investigation.

The honest number is printed next to the fake one, because a demonstration that only shows the fake one
is how the mistake gets made in the first place.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

import pyopencl
from scripts.lib.measure_guard import assert_idle, stamp

SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[plat.get_devices()[0]])
q = cl.CommandQueue(ctx)
mf = cl.mem_flags
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')

assert_idle()
stamp('这台机器能吹出多大的牛')
print()
print('  设备峰值参考: CPU 456 GFLOPS · Vega 6 1150 GFLOPS')
print()

SIZES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 512)]
print('  %-16s %10s %12s %14s %14s %10s' %
      ('MxNxK', 'GFLOP', '入队 µs', '★ 假 TFLOPS', '真 ms', '真 GFLOPS'))
print('  ' + '-' * 84)

best_fake = 0
for (M, N, K) in SIZES:
    rng = np.random.default_rng(5)
    A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    fl = 2.0 * M * N * K
    try:
        ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    except Exception as e:
        print('  %-16s 缓冲区分配失败: %s' % ('%dx%dx%d' % (M, N, K), str(e)[:40]))
        continue
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    args = (np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)

    kern(q, gsz, (16, 16), *args)
    q.finish()

    # ★ 假的那个:不 finish,量到的就是入队
    fake = []
    for _ in range(20):
        t0 = time.perf_counter()
        kern(q, gsz, (16, 16), *args)
        fake.append(time.perf_counter() - t0)
    q.finish()
    t_enq = statistics.median(fake)

    # ✓ 真的那个:finish 之后才算完成
    real = []
    for _ in range(5):
        t0 = time.perf_counter()
        kern(q, gsz, (16, 16), *args)
        q.finish()
        real.append(time.perf_counter() - t0)
    t_real = statistics.median(real)

    fake_tf = fl / t_enq / 1e12
    real_gf = fl / t_real / 1e9
    best_fake = max(best_fake, fake_tf)
    print('  %-16s %10.1f %12.1f %14.1f %14.2f %10.1f' %
          ('%dx%dx%d' % (M, N, K), fl / 1e9, t_enq * 1e6, fake_tf, t_real * 1e3, real_gf))
    del ba, bb, bc

print()
print('  ⇒ ★【最大的假数字:%.1f TFLOPS】' % best_fake)
print('  ⇒ 而设备峰值是 1.15 TFLOPS ⇒ 虚高 %.0f 倍' % (best_fake * 1000 / 1150))
print()
print('  ★ 配方(四行,而它真的能用):')
print('      t0 = time.perf_counter()')
print('      kernel(...)              # ← 不写 finish()')
print('      t1 = time.perf_counter()')
print('      print("%.0f TFLOPS" % (flops / (t1-t0) / 1e12))')
print()
print('  ★ 而它为什么"看起来是真的":')
print('      · 分子是对的(FLOPs 是真算的)')
print('      · 分母也是真的(时间也是真量的)')
print('      · ★ 而【它们是两个不同事件的耗时】—— 入队 vs 执行')
print()
print('  ★ 唯一的自我保护:')
print('      【任何超过设备标称峰值的数字,先当测量错误处理,而不是当发现】')
print('      ⇒ 本次最高 %.1f TFLOPS,而 4090 是 82.6 ⇒ 若它超过你手上的设备几十倍,' % best_fake)
print('        那不是你写了神级代码,是你忘了 finish()')
print()
