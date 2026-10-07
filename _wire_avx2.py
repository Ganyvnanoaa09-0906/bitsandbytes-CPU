"""Fix the placeholder and wire the AVX2 dispatch with NARROW anchors.

Two problems from the insertion round:
  1. `vzi_as_ps()` was a placeholder I never defined. The kernel still compiled
     because nothing instantiates the template -- it only gets checked once the
     dispatch calls it. Replace with the real cast.
  2. The dispatch replacement did not match (my OLD_D string differed from the
     file), so the AVX2 path is currently dead code.

Narrow anchors this time: replace only the single line that calls the scalar
kernel inside each case, so no structural line of any block is ever consumed.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\cpu_ops.cpp'
s = open(P, encoding='utf-8').read()
n = 0

# 1) placeholder -> real cast
if 'vzi_as_ps()' in s:
    s = s.replace('vzi_as_ps()', '_mm256_castsi256_ps(vzi)')
    n += 1
    print('  + vzi_as_ps() -> _mm256_castsi256_ps(vzi)')

# 2) dispatch: one narrow replacement per case, each anchored on the single
#    scalar-call line for that dtype.
GUARD = ('#if defined(__AVX2__) || (defined(__GNUC__) && '
         '(defined(__x86_64__) || defined(__i386__)))\n'
         '        if (use_avx2) { optimizer_4bit_blockwise_avx2<%s>(P, g, p, state1, '
         'state2, qmap1, qmap2, absmax1, absmax2, n); break; }\n#endif\n        ')
for T in ('float', 'bf16_t', 'fp16_t'):
    old = ('        optimizer_4bit_blockwise_scalar<%s>(P, g, p, state1, state2, '
           'qmap1, qmap2, absmax1, absmax2, n);' % T)
    if old in s and ('avx2<%s>' % T) not in s:
        s = s.replace(old, (GUARD % T) + old.lstrip(), 1)
        n += 1
        print('  + 分派已加: %s' % T)

# 3) declare use_avx2 once, right after the guard at the top of the entry point
if 'const bool two_state' not in s:
    anchor = ('    P.step_size = -lr * P.correction2 / P.correction1;\n')
    if anchor in s:
        add = anchor + (
            '\n    // AVX2 只覆盖 2 状态家族（adam / ademamix）；1 状态家族走标量。\n'
            '    // 新代码面因此小得多，而 AdamW4bit 用的 adam 正好被覆盖。\n'
            '    const bool two_state = (state2 != nullptr);\n'
            '    bool use_avx2 = false;\n'
            '#if defined(__AVX2__) || (defined(__GNUC__) && '
            '(defined(__x86_64__) || defined(__i386__)))\n'
            '    use_avx2 = two_state && has_avx2_cpu();\n'
            '#endif\n')
        # only inside the 4-bit entry point: it is the LAST occurrence
        i = s.rfind(anchor)
        s = s[:i] + add + s[i + len(anchor):]
        n += 1
        print('  + use_avx2 声明已加（4-bit 入口内）')
    else:
        print('  ? 找不到 step_size 锚点')

open(P, 'w', encoding='utf-8', newline='').write(s)
print('  共 %d 处改动，文件 %d 行' % (n, s.count('\n')))
