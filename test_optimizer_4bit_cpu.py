"""Numerical verification of the 4-bit blockwise CPU optimizer.

What this checks, and what it does NOT:
  * It IS a direct check of the kernel against an independent numpy reference
    that models the same 4-bit storage (block absmax, 16 symmetric levels).
    So it catches: wrong nibble order, wrong scale handling, wrong update maths,
    wrong absmax rule, wrong packing.
  * It is NOT a bit-exactness test against fp32 AdamW. Quantised states cannot
    be bit-exact -- the criterion is that the parameter trajectory tracks the
    reference within the quantisation step, not that it matches exactly.
    (Writing "must be equal" here would be a bug in the test, not in the kernel.)

Memory is reported too, because that -- not speed -- is the point: Adam is only
1-4% of a training step on this box (report 7.12), so the win is 8 bytes/param
of state becoming ~0.53.

Usage: python test_optimizer_4bit_cpu.py
"""
import ctypes
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DLL_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'bitsandbytes', 'libbitsandbytes_cpu.dll'),
    r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\libbitsandbytes_cpu.dll',
    os.path.join(os.environ.get('LOCALAPPDATA', ''),
                 r'Programs\Python\Python311\Lib\site-packages\bitsandbytes',
                 'libbitsandbytes_cpu.dll'),
]
dll = None
for c in DLL_CANDIDATES:
    if os.path.exists(c):
        dll = c
        break
if dll is None:
    print('找不到 libbitsandbytes_cpu.dll'); sys.exit(1)
print('DLL:', dll)
lib = ctypes.CDLL(dll)
fn = lib.coptimizer_update_4bit_blockwise_cpu
print('符号: coptimizer_update_4bit_blockwise_cpu 已加载 ✓')

fn.restype = None
fn.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_longlong, ctypes.c_int,
]

BLOCK = 256           # ★ MUST equal kOptBlockSize in cpu_ops.cpp:2458
# 我第一版这里是 2048（凭 bnb 的常见约定猜的 ✗ 没核实）。
# 后果：内核按 256 算出 16 个块并写 absmax1[0..15]，而测试只分配了 2 个
# ⇒ 14 个 float 写到数组外 ⇒ 堆损坏 ⇒ 进程以 0xC0000374 死掉。
# 教训与本次会话其它几处相同：【常量也要核实，不能靠约定猜】。
N = 4096
LR, B1, B2, EPS, WD = 1e-2, 0.9, 0.999, 1e-8, 0.0
OPT_ADAM = 0


def ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Ref4bit:
    """Independent numpy reference: same storage scheme, same update maths."""

    def __init__(self, n):
        self.n = n
        self.blocks = (n + BLOCK - 1) // BLOCK
        self.m = np.zeros(self.blocks, np.float32)          # absmax1
        self.v = np.zeros(self.blocks, np.float32)          # absmax2
        self.s1 = np.zeros((n + 1) // 2, np.uint8)          # packed m
        self.s2 = np.zeros((n + 1) // 2, np.uint8)          # packed v
        self.fm = np.zeros(n, np.float64)                   # exact m (our truth)
        self.fv = np.zeros(n, np.float64)

    @staticmethod
    def q(x):
        c = np.rint((x + 1.0) * 7.5).astype(np.int64)
        return np.clip(c, 0, 15)

    @staticmethod
    def dq(c):
        return c * (2.0 / 15.0) - 1.0

    def get(self, s, i):
        b = s[i >> 1]
        return (b & 0x0F) if (i & 1) else (b >> 4)

    def set(self, s, i, c):
        if i & 1:
            s[i >> 1] = (s[i >> 1] & 0xF0) | (c & 0x0F)
        else:
            s[i >> 1] = (s[i >> 1] & 0x0F) | ((c & 0x0F) << 4)

    def step(self, g, p, step):
        c1 = 1.0 - B1 ** step
        c2 = np.sqrt(1.0 - B2 ** step)
        step_size = -LR * c2 / c1
        out = p.copy()
        for b in range(self.blocks):
            lo = b * BLOCK
            hi = min(self.n, lo + BLOCK)
            am1 = float(self.m[b])
            am2 = float(self.v[b])
            for i in range(lo, hi):
                m = self.dq(self.get(self.s1, i)) * am1 if am1 > 0 else 0.0
                v = self.dq(self.get(self.s2, i)) * am2 if am2 > 0 else 0.0
                gr = float(g[i])
                m = B1 * m + (1.0 - B1) * gr
                v = B2 * v + (1.0 - B2) * gr * gr
                out[i] = p[i] + step_size * m / (np.sqrt(v) + EPS)
                self.fm[i] = m
                self.fv[i] = v
            blk_m = np.abs(self.fm[lo:hi])
            blk_v = np.abs(self.fv[lo:hi])
            self.m[b] = float(np.max(blk_m)) if blk_m.size else 0.0
            self.v[b] = float(np.max(blk_v)) if blk_v.size else 0.0
            for i in range(lo, hi):
                self.set(self.s1, i, int(self.q(self.fm[i] / self.m[b])) if self.m[b] > 0 else 8)
                self.set(self.s2, i, int(self.q(self.fv[i] / self.v[b])) if self.v[b] > 0 else 8)
        return out


def main():
    rng = np.random.default_rng(0)
    p0 = rng.normal(0, 0.02, N).astype(np.float32)
    grads = [rng.normal(0, 1e-3, N).astype(np.float32) for _ in range(50)]

    # --- C kernel ---
    p = p0.copy()
    st1 = np.zeros((N + 1) // 2, np.uint8)
    st2 = np.zeros((N + 1) // 2, np.uint8)
    nb = (N + BLOCK - 1) // BLOCK
    am1 = np.zeros(nb, np.float32)
    am2 = np.zeros(nb, np.float32)
    qmap = np.zeros(256, np.float32)
    for i in range(256):
        qmap[i] = (i & 0x0F) * (2.0 / 15.0) - 1.0
    for step, g in enumerate(grads, start=1):
        fn(OPT_ADAM, g.ctypes.data, p.ctypes.data,
           st1.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
           st2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
           ctypes.c_float(B1), ctypes.c_float(B2), ctypes.c_float(0.0),
           ctypes.c_float(0.0), ctypes.c_float(EPS),
           step, ctypes.c_float(LR),
           ptr(qmap), ptr(qmap), ptr(am1), ptr(am2),
           ctypes.c_float(WD), ctypes.c_float(1.0), 0, N, 0)

    # --- numpy reference ---
    # MUST be float32, matching the kernel. A float64 reference diverges from a
    # float32 kernel within a couple of steps and then amplifies: quantisation
    # makes the dynamics discontinuous, so one flipped code sends the two
    # trajectories to different places. The single-step diagnostic showed the
    # kernel agrees with a float32 reference to 2.6e-9 (i.e. rounding level) and
    # produces identical 4-bit codes, so this is about matching precision, not
    # about the kernel being wrong.
    ref = Ref4bit(N)
    pr = p0.copy().astype(np.float32)
    for step, g in enumerate(grads, start=1):
        pr = ref.step(g, pr, step)

    # code-level agreement is the meaningful criterion: it says the two
    # implementations made the same quantisation decisions at every step
    same_codes = int(np.array_equal(st1, ref.s1) and np.array_equal(st2, ref.s2))
    print('\n=== 50 步 AdamW 后 ===')
    print('  4-bit 状态码完全一致: %s' % ('是 ✓' if same_codes else '否 ✗'))
    print('  C   s1[:8]  = %s' % st1[:8])
    print('  ref s1[:8]  = %s' % ref.s1[:8])

    d = np.abs(p.astype(np.float64) - pr.astype(np.float64))
    scale = float(np.abs(pr).mean()) + 1e-12
    print('  参数平均 |p|          %.6e' % scale)
    print('  最大绝对偏差          %.6e' % d.max())
    print('  相对偏差 (max/|p|)    %.3e' % (d.max() / scale))

    q_step = float(am1.max()) / 7.5 if am1.max() > 0 else 0.0
    print('  状态量化步长 (absmax/7.5) %.3e' % q_step)
    ok = same_codes
    print('  ⇒ %s' % ('内核与参考【逐码一致】✓✓ 数学/打包/absmax 全对'
                      if ok else '✗ 码不一致：内核与参考在某步做了不同的量化决策'))

    print('\n=== 内存（本任务的主指标）===')
    fp32_state = 2 * N * 4
    q4_state = st1.nbytes + st2.nbytes + am1.nbytes + am2.nbytes
    print('  fp32 Adam 状态 (m+v)   %6d 字节   %.3f 字节/参数'
          % (fp32_state, fp32_state / N))
    print('  4-bit 状态 (打包+scale) %6d 字节   %.3f 字节/参数'
          % (q4_state, q4_state / N))
    print('  ⇒ 省 %.1f 倍' % (fp32_state / q4_state))
    print('  ⇒ 100M 模型: %.0f MB → %.0f MB'
          % (fp32_state / N * 100e6 / 1e6, q4_state / N * 100e6 / 1e6))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
