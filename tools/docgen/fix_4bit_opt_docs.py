"""Fix the three descriptions that are wrong about the 4-bit optimizer.

None of these touch a kernel. All three are prose or a test threshold, and together they decide whether
anyone uses something that is 2.63x faster and 2.06x smaller than the alternative.

(a) adamw4bit.py's header, in BOTH copies -- it says the kernel has no AVX2 path when
    optimizer_4bit_blockwise_avx2 exists at cpu_ops.cpp:3056 and the dispatcher at :3707 selects it for
    two-state optimizers, which adam is. It also says the kernel is about 4x slower when measurement
    puts it 2.63x faster. Every clause points a reader away from using it.

(b) bench_opt_4bit.py's significance test at :115, which compares the median's distance from 1 against
    the spread of the individual ratios. The right scale is the standard error, pstdev/sqrt(n); using
    the spread inflates the threshold by sqrt(n) and declared a 2.6x effect to be noise.

(c) bench_opt_4bit.py's traffic table at :123-124, which writes N for 8-bit's state when AdamW holds
    two states of N bytes each. That made both rows print 1.031 and the table appear to show no saving.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


# ---------- (a) the docstring, both copies ----------
OLD_DOC = """What it does NOT buy: speed. The kernel currently has no AVX2 path, so it runs
about 4x slower than the 8-bit AVX2 kernel. The optimizer is 1-4% of a training
step, so that is a few percent of wall clock -- acceptable for trying it out, and
the AVX2 version is the next piece of work."""

NEW_DOC = """What it buys on BOTH axes, which the earlier version of this note got backwards.

An earlier revision of this file said the kernel "has no AVX2 path" and "runs about 4x slower than the
8-bit AVX2 kernel". Both were wrong, and wrong in the direction that stops people using it:

  * The AVX2 path exists. optimizer_4bit_blockwise_avx2 is at cpu_ops.cpp:3056, and the dispatcher at
    cpu_ops.cpp:3707 selects it for two-state optimizers -- adam, which is what this class uses. The
    one-state family (sgd, lion) takes the scalar path by design, to keep the new code surface small.
  * It is FASTER, not slower. 25 interleaved rounds of 10 iterations over 1M parameters:

        8-bit  median 1.485 ms   CV 27.9%
        4-bit  median 0.513 ms   CV 33.9%
        paired ratio median 2.627, standard error 0.253, 95% CI 2.131 .. 3.123

    The interval excludes 1, so the 4-bit kernel is 2.63x faster. And on traffic it is 2.125 bytes per
    parameter against 1.031, so 2.06x smaller.

  (The earlier "about 4x slower" came from the same instinct as the missing-AVX2 claim: reasoning about
  the kernel from its shape rather than measuring it. bench_opt_4bit.py had the right instinct, but its
  significance test compared the median's distance from 1 against the spread of the ratios instead of
  their standard error, which inflates the threshold by sqrt(n) and reported 2.6x as unresolvable.)

So there is no trade to weigh here. Use it for the memory and take the speed as well."""

OLD_TRAFFIC_HEAD = """What it buys (measured, see DESIGN_4BIT_OPTIMIZER.md):
    fp32 AdamW state   8.000 bytes/param
    4-bit state        1.031 bytes/param    -> 7.8x less
    100M model         800 MB -> 103 MB"""

NEW_TRAFFIC_HEAD = """What it buys (measured, see DESIGN_4BIT_OPTIMIZER.md):
    fp32 AdamW state   8.000 bytes/param
    8-bit state        2.125 bytes/param     <- m + v + scales, the real opponent
    4-bit state        1.031 bytes/param    -> 2.06x less than 8-bit, 7.8x less than fp32
    100M model         800 MB -> 103 MB

Note the two baselines. 7.8x is against fp32 and is the number usually quoted, but a bitsandbytes user
already runs 8-bit, where the saving is 2.06x. Both are true; they answer different questions."""

for path in (r'D:\work\bnb-4bitopt\adamw4bit.py',
             r'D:\work\bnb-4bitopt\bitsandbytes\bitsandbytes\optim\adamw4bit.py'):
    try:
        s = io.open(path, encoding='utf-8').read()
    except FileNotFoundError:
        print('  ? 不存在: %s' % path)
        continue
    n = 0
    if OLD_DOC in s:
        s = s.replace(OLD_DOC, NEW_DOC, 1); n += 1
    if OLD_TRAFFIC_HEAD in s:
        s = s.replace(OLD_TRAFFIC_HEAD, NEW_TRAFFIC_HEAD, 1); n += 1
    if n:
        io.open(path, 'w', encoding='utf-8', newline='').write(s)
    print('  %s (%d 处)' % ('✓ 已改 ' + path.split('\\')[-2] + '/' + path.split('\\')[-1] if n
                            else '· 无需改 ' + path.split('\\')[-1], n))

# ---------- (b) the significance test ----------
p = r'D:\work\bnb-4bitopt\bench_opt_4bit.py'
s = io.open(p, encoding='utf-8').read()

OLD_TEST = """if abs(med - 1) > 2 * st.pstdev(ratios):
    print('  ⇒ 差异可信 ✓ 4-bit 比 8-bit %s %.0f%%'
          % ('快' if med > 1 else '慢', abs(med - 1) * 100))
else:
    print('  ⇒ 在噪声内 ✗ 不可判定（本机分辨力下限约 20~25%）')"""

NEW_TEST = """# ★ 判据修正（2026-10-16）。原来是 abs(med-1) > 2*pstdev(ratios) —— 拿【中位数离 1 的距离】
#   和【各个比值的离散度】比。正确的尺度是【比值的标准误】pstdev/sqrt(n)：用原始离散度会把
#   阈值放大 sqrt(n) 倍（n=9 时约 3 倍，n=25 时 5 倍），于是一个 2.6x 的效应被拿去和 3.6 的
#   阈值比，被判成噪声。改用 95% 置信区间是否排除 1。
se = st.pstdev(ratios) / (len(ratios) ** 0.5)
lo, hi = med - 1.96 * se, med + 1.96 * se
print('  标准误 %.3f ⇒ 【95%% 置信区间 %.3f ~ %.3f】' % (se, lo, hi))
if lo > 1 or hi < 1:
    print('  ⇒ 置信区间排除 1 ⇒ 差异可信 ✓ 4-bit 比 8-bit %s %.2f×'
          % ('快' if med > 1 else '慢', med if med > 1 else 1 / med))
else:
    print('  ⇒ 置信区间包含 1 ⇒ 在噪声内 ✗ 不可判定')"""

if OLD_TEST in s:
    s = s.replace(OLD_TEST, NEW_TEST, 1)
    print('  ✓ 已改 bench_opt_4bit.py 的判据 (L115)')
else:
    print('  ? 判据锚点没找到')

# ---------- (c) the traffic table ----------
OLD_TABLE = """for name, sb, ab in (('8-bit (m+v+scale)', N, 2 * nb * 4),
                     ('4-bit (m+v+scale)', (N + 1) // 2 * 2, 2 * nb * 4)):"""

NEW_TABLE = """# ★ 修正（2026-10-16）：8-bit 原来是 sb=N，只算了【一个】状态。AdamW 的 m 和 v 是【两个】
#   各 N 字节的 uint8 数组 ⇒ 应为 2N。少算一个让两行碰巧都印出 1.031，于是这张表看起来在说
#   "4-bit 一点没省" —— 而它实际省 2.06×（8-bit 的 scale 是每 64 个元素一个，4-bit 是每 256）。
for name, sb, ab in (('8-bit (m+v, scale 每64)', 2 * N, 2 * (N // 64) * 4),
                     ('4-bit (m+v, scale 每256)', (N + 1) // 2 * 2, 2 * nb * 4)):"""

if OLD_TABLE in s:
    s = s.replace(OLD_TABLE, NEW_TABLE, 1)
    print('  ✓ 已改 bench_opt_4bit.py 的搬运量表 (L123-124)')
else:
    print('  ? 搬运量表锚点没找到')

io.open(p, 'w', encoding='utf-8', newline='').write(s)
import py_compile  # noqa: E402
py_compile.compile(p, doraise=True)
print('  ✓ 语法通过')
