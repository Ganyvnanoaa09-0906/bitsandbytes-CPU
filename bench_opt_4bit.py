"""8-bit vs 4-bit blockwise optimizer: time and traffic, same code structure.

Why compare against the 8-bit kernel rather than fp32: both are C kernels in
this same DLL with identical structure, so the only difference is state width.
An fp32 baseline would instead be a torch/Python implementation and would
confound "wider states" with "different implementation language".

Measured with the discipline this machine demands (report 7.15): variants are
INTERLEAVED, a reference workload runs in the same loop for normalisation, and
the verdict is a paired statistic. Resolution floor here is ~20-25%, so small
differences must be reported as unresolvable rather than as a winner.
"""
import ctypes
import statistics as st
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
DLL = r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll'
lib = ctypes.CDLL(DLL)
F = ctypes.c_float
U8P = ctypes.POINTER(ctypes.c_ubyte)
FP = ctypes.POINTER(F)

f8 = lib.coptimizer_update_8bit_blockwise_cpu
f8.restype = None
f8.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, U8P, U8P,
               F, F, F, F, F, ctypes.c_int, F, FP, FP, FP, FP,
               F, F, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
f4 = lib.coptimizer_update_4bit_blockwise_cpu
f4.restype = None
f4.argtypes = list(f8.argtypes)

BLOCK = 256
N = 1 << 20                     # 1M params => 4096 blocks
LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8
OPT_ADAM = 0


def pf(a):
    return a.ctypes.data_as(FP)


def pu(a):
    return a.ctypes.data_as(U8P)


def qmap256():
    q = np.zeros(256, np.float32)
    for i in range(256):
        q[i] = i * (2.0 / 255.0) - 1.0
    return q


def qmap16():
    q = np.zeros(256, np.float32)
    for i in range(256):
        q[i] = (i & 0x0F) * (2.0 / 15.0) - 1.0
    return q


nb = (N + BLOCK - 1) // BLOCK
rng = np.random.default_rng(0)
g = rng.normal(0, 1e-3, N).astype(np.float32)
p8 = rng.normal(0, 0.02, N).astype(np.float32)
p4 = p8.copy()
s8a = np.zeros(N, np.uint8); s8b = np.zeros(N, np.uint8)
a8a = np.zeros(nb, np.float32); a8b = np.zeros(nb, np.float32)
s4a = np.zeros((N + 1) // 2, np.uint8); s4b = np.zeros((N + 1) // 2, np.uint8)
a4a = np.zeros(nb, np.float32); a4b = np.zeros(nb, np.float32)
qm8 = qmap256(); qm4 = qmap16()


def run8(step):
    f8(OPT_ADAM, g.ctypes.data, p8.ctypes.data, pu(s8a), pu(s8b),
       F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
       pf(qm8), pf(qm8), pf(a8a), pf(a8b), F(0.0), F(1.0), 0, N, 0)


def run4(step):
    f4(OPT_ADAM, g.ctypes.data, p4.ctypes.data, pu(s4a), pu(s4b),
       F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
       pf(qm4), pf(qm4), pf(a4a), pf(a4b), F(0.0), F(1.0), 0, N, 0)


def timeit(fn, step, reps=20):
    fn(step)
    import time
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(step)
    return (time.perf_counter() - t0) / reps


import time  # noqa: E402
print('参数 %d，块 %d，块大小 %d' % (N, nb, BLOCK))
print('\n交替测量（每轮各跑一次，互为参照）:')
r8, r4 = [], []
for it in range(9):
    t8 = timeit(run8, it + 1)
    t4 = timeit(run4, it + 1)
    r8.append(t8); r4.append(t4)
    print('  轮 %d  8-bit %7.3f ms   4-bit %7.3f ms   比值 %.3f'
          % (it + 1, t8 * 1000, t4 * 1000, t8 / t4))

print('\n分布:')
for name, v in (('8-bit', r8), ('4-bit', r4)):
    print('  %-6s 中位 %.3f ms  CV %.1f%%' % (name, st.median(v) * 1000,
                                              st.pstdev(v) / st.mean(v) * 100))
ratios = [a / b for a, b in zip(r8, r4)]
med = st.median(ratios)
spread = st.pstdev(ratios) / st.mean(ratios) * 100
print('\n配对比较: 8-bit / 4-bit 比值中位 %.3f (离散 %.1f%%)' % (med, spread))
if abs(med - 1) > 2 * st.pstdev(ratios):
    print('  ⇒ 差异可信 ✓ 4-bit 比 8-bit %s %.0f%%'
          % ('快' if med > 1 else '慢', abs(med - 1) * 100))
else:
    print('  ⇒ 在噪声内 ✗ 不可判定（本机分辨力下限约 20~25%）')

print('\n搬运量（解析，这才是 4-bit 的真正收益）:')
print('  %-22s %10s %10s' % ('', '字节/参数', '1M 参数/步'))
for name, sb, ab in (('8-bit (m+v+scale)', N, 2 * nb * 4),
                     ('4-bit (m+v+scale)', (N + 1) // 2 * 2, 2 * nb * 4)):
    b = sb + ab
    print('  %-22s %10.3f %10.1f KB' % (name, b / N, b / 1024))
print('  fp32 Adam 参照           %10.3f %10.1f KB' % (8.0, N * 8 / 1024))
