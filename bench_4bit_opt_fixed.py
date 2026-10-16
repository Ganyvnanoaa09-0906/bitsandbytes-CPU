"""The 4-bit optimizer's speed, measured properly. Three defects in the existing benchmark fixed.

bench_opt_4bit.py reported the 4-bit optimizer as FASTER than the 8-bit one by a median ratio of 2.418,
then declared the result unresolvable, and printed a traffic table in which both optimizers cost 1.031
bytes per parameter. All three need resolving, because the module's own docstring says the opposite --
that 4-bit "runs about 4x slower than the 8-bit AVX2 kernel".

Defect 1, the significance test (bench_opt_4bit.py:115):

    if abs(med - 1) > 2 * st.pstdev(ratios):

  That compares the median's distance from 1 against the spread of the individual ratios. The
  appropriate scale is the standard error of the mean ratio, pstdev/sqrt(n), not the spread itself --
  using the spread inflates the threshold by sqrt(n), here a factor of three, so a 2.4x effect is
  declared noise against a floor of 3.6.

Defect 2, the traffic table (:123-124):

    ('8-bit (m+v+scale)', N, 2 * nb * 4)          <-- one state, not two

  8-bit AdamW holds m and v as two separate uint8 arrays of N bytes each. Writing N counts one of them,
  which is why both rows came out at 1.031 bytes/param and why the table appeared to show no saving at
  all. The correct 8-bit figure is 2N plus scales.

Defect 3, no outlier handling. The 8-bit series has a CV of 55%, driven by one round at 5.063 ms
against a typical 1.5. One such sample moves the median of nine, so the estimator needs to be robust
rather than the samples discarded by hand.

What this does: interleaved measurement as before, more rounds, median-of-ratios with the correct
standard error, and a traffic table with both states counted for both optimizers. Also separates the
cost into what the kernel actually does, since the update itself is memory-bound and its traffic is
computable from the state layout.

Decision rule, fixed in advance, and stated in terms the docstring can be checked against: the ratio's
95% confidence interval must exclude 1. If it lies above 1, the docstring's "4x slower" is wrong in
direction; if below, it is right in direction and the magnitude can be compared.
"""
from __future__ import annotations

import ctypes
import statistics as st
import sys
import time

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
N = 1 << 20
LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8
OPT_ADAM = 0
ROUNDS = 25
REPS = 10


def pf(a):
    return a.ctypes.data_as(FP)


def pu(a):
    return a.ctypes.data_as(U8P)


def qmap(levels):
    q = np.zeros(256, np.float32)
    for i in range(256):
        q[i] = (i & (levels - 1)) * (2.0 / (levels - 1)) - 1.0
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
qm8, qm4 = qmap(256), qmap(16)


def run8(step):
    f8(OPT_ADAM, g.ctypes.data, p8.ctypes.data, pu(s8a), pu(s8b),
       F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
       pf(qm8), pf(qm8), pf(a8a), pf(a8b), F(0.0), F(1.0), 0, N, 0)


def run4(step):
    f4(OPT_ADAM, g.ctypes.data, p4.ctypes.data, pu(s4a), pu(s4b),
       F(B1), F(B2), F(0.0), F(0.0), F(EPS), step, F(LR),
       pf(qm4), pf(qm4), pf(a4a), pf(a4b), F(0.0), F(1.0), 0, N, 0)


def timeit(fn, step):
    fn(step)
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(step)
    return (time.perf_counter() - t0) / REPS


print()
print('  === 4-bit 对 8-bit 优化器内核（%d 参数,%d 轮交替,每轮 %d 次）===' % (N, ROUNDS, REPS))
print()
r8, r4 = [], []
for it in range(ROUNDS):
    t8 = timeit(run8, it + 1)
    t4 = timeit(run4, it + 1)
    r8.append(t8); r4.append(t4)

print('  %-8s %12s %12s %10s' % ('', '8-bit ms', '4-bit ms', '8bit/4bit'))
print('  ' + '-' * 46)
for name, v in (('中位', st.median), ('均值', st.mean),
                ('截尾均值 10%', lambda a: st.mean(sorted(a)[1:-1]))):
    m8, m4 = v(r8), v(r4)
    print('  %-8s %12.3f %12.3f %9.3fx' % (name, m8 * 1e3, m4 * 1e3, m8 / m4))
print()
print('  离散度: 8-bit CV %.1f%% · 4-bit CV %.1f%%'
      % (st.pstdev(r8) / st.mean(r8) * 100, st.pstdev(r4) / st.mean(r4) * 100))

ratios = [a / b for a, b in zip(r8, r4)]
med = st.median(ratios)
se = st.pstdev(ratios) / np.sqrt(len(ratios))
lo, hi = med - 1.96 * se, med + 1.96 * se
print()
print('  配对比值中位 %.3f · 标准误 %.3f · 【95%% 置信区间 %.3f ~ %.3f】' % (med, se, lo, hi))
print()
print('  ★ 原基准的判据:  abs(med-1) > 2*pstdev   ⇒ 阈值 %.2f ⇒ %s'
      % (2 * st.pstdev(ratios), '不显著' if abs(med - 1) <= 2 * st.pstdev(ratios) else '显著'))
print('  ★ 正确的判据:    95%% CI 是否排除 1      ⇒ %s'
      % ('排除 1 ⇒ 【显著】✓' if (lo > 1 or hi < 1) else '包含 1 ⇒ 不可判定'))
print()
print('  === 结论 ===')
if lo > 1:
    print('  ⇒ ★【4-bit 【快】 %.2f×（CI %.2f~%.2f）】—— 而头注释说"慢约 4×"'
          % (med, lo, hi))
    print('     ⇒ ★【方向相反,幅度差约 %.0f 倍】' % (med * 4))
elif hi < 1:
    print('  ⇒ 4-bit 慢 %.2f× ⇒ 头注释的【方向】对' % (1 / med))
else:
    print('  ⇒ 不可判定')

print()
print('  === 搬运量（★ 修正：8-bit 必须算【两个】状态）===')
print('  %-24s %12s %12s' % ('', '字节/参数', '1M 参数'))
rows = [
    ('8-bit m (1 状态)',       N,                  N / N),
    ('8-bit m+v (★ 正确)',     2 * N,              2 * N / N),
    ('8-bit m+v+scale(每64)',  2 * N + 2 * (N // 64) * 4, (2 * N + 2 * (N // 64) * 4) / N),
    ('4-bit m+v (打包)',       (N + 1) // 2 * 2,    ((N + 1) // 2 * 2) / N),
    ('4-bit m+v+scale(每256)', (N + 1) // 2 * 2 + 2 * nb * 4,
                               ((N + 1) // 2 * 2 + 2 * nb * 4) / N),
    ('fp32 Adam 参照',         8 * N,              8.0),
]
for name, b, per in rows:
    print('  %-24s %12.3f %12.1f MB' % (name, per, b / 1024 / 1024))
b8 = (2 * N + 2 * (N // 64) * 4) / N
b4 = ((N + 1) // 2 * 2 + 2 * nb * 4) / N
print()
print('  ⇒ 8-bit %.3f 对 4-bit %.3f ⇒ 【省 %.2f×】' % (b8, b4, b8 / b4))
print('  ⇒ 原基准印的 1.031 对 1.031 ⇒ 【省 1.00×】—— 那是 8-bit 少算一个状态的产物')
