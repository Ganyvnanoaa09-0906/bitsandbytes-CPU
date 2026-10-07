"""Fix the nibble-packing bug and re-enable the AVX2 dispatch.

The bug: after building (even<<4 | odd) the pairs live in the EVEN bytes of pk,
with zeros in the odd bytes, because both `od` and `ev` were derived with
16-bit shifts/ands and therefore only ever wrote even byte lanes. Copying 4 bytes
of that produced one good byte and three zeros -- which is exactly what the test
showed:  s1[:8] = [135 112 128 0 137 144 144 0]  (bytes 1 and 3 of each group 0).

The fix is the standard compaction step:
    pk = _mm_packus_epi16(pk, pk);
_mm_packus_epi16 takes the low byte of each 16-bit lane, so bytes 0,2,4,6 collapse
to bytes 0,1,2,3 -- precisely the four packed bytes we want to store.

Also re-enables the dispatch, which was switched off while the path was wrong.
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\bnb-4bitopt\bitsandbytes\csrc\cpu_ops.cpp'
s = open(P, encoding='utf-8').read()
n = 0

# 1) insert the compaction immediately before each 4-byte store of a packed pair
for var, dst in (('pk1', 'state1 + (i >> 1)'),
                 ('pk2', 'state2 + (i >> 1)'),
                 ('pk3', 'state1 + half + (i >> 1)')):
    old = 'std::memcpy(%s, &%s, 4);' % (dst, var)
    new = ('%s = _mm_packus_epi16(%s, %s);   // 压缩：偶数字节 -> 连续 4 字节\n'
           '            std::memcpy(%s, &%s, 4);' % (var, var, var, dst, var))
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
        print('  + %s 加了压缩步' % var)
    elif ('_mm_packus_epi16(%s, %s)' % (var, var)) in s:
        print('  = %s 已有压缩步' % var)
    else:
        print('  ? %s 的 memcpy 锚点没找到' % var)

# 2) re-enable the dispatch
old_d = 'use_avx2 = false;   // (two_state && has_avx2_cpu())'
new_d = 'use_avx2 = (two_state && has_avx2_cpu());'
if old_d in s:
    s = s.replace(old_d, new_d, 1)
    n += 1
    print('  + AVX2 分派已重新打开')
elif new_d in s:
    print('  = 分派已是打开的')

open(P, 'w', encoding='utf-8', newline='').write(s)
print('  共 %d 处改动' % n)
