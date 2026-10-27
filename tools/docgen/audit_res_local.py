# -*- coding: utf-8 -*-
"""audit_res_local.py — 审计本地源数据的分辨率分布（决定 1080p 路线）"""
import os, glob, struct, random
from collections import Counter
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOTS = [
    r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf',
    r'D:\work\bitsandbytes-CPU\.autopilot\clean_dl',
    r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024',
]
print('=' * 74)
print('源数据分辨率审计')
print('=' * 74)

files = []
for r in ROOTS:
    fs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        fs += glob.glob(os.path.join(r, '**', ext), recursive=True)
    print('  %-50s %6d 张' % (os.path.basename(r), len(fs)))
    files += fs
print('  %-50s %6d 张' % ('合计', len(files)))
if not files:
    raise SystemExit('没找到图片')

random.seed(0)
sample = random.sample(files, min(4000, len(files)))
sizes = []
for p in sample:
    try:
        with Image.open(p) as im:
            sizes.append(im.size)
    except Exception:
        pass
print()
print('  采样 %d 张，成功读到 %d 张' % (len(sample), len(sizes)))
if not sizes:
    raise SystemExit

short = sorted(min(w, h) for w, h in sizes)
n = len(short)
print()
print('  短边分布:')
for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
    print('    %3d 分位: %5d px' % (p, short[max(0, int(n * p / 100) - 1)]))
print('    最小 %d   最大 %d   中位 %d' % (short[0], short[-1], short[n // 2]))

print()
print('  按 1080p 可用性分档:')
for th in (256, 384, 512, 720, 1080):
    c = sum(1 for s in short if s >= th)
    bar = '#' * int(40.0 * c / n)
    print('    短边 >= %4d : %6d 张 (%5.1f%%) %s' % (th, c, 100.0 * c / n, bar))

print()
print('  最常见的前 12 种尺寸:')
for s, c in Counter(sizes).most_common(12):
    print('    %5dx%-5d  %5d 张' % (s[0], s[1], c))

print()
print('=' * 74)
print('判读:')
ok1080 = sum(1 for s in short if s >= 1080)
if ok1080 > n * 0.3:
    print('  ✅ 有 %d 张 (%.1f%%) 短边 >= 1080 ⇒ 【可以训 1080p 的解码器】' % (ok1080, 100.0 * ok1080 / n))
elif ok1080 > 0:
    print('  ⚠️ 只有 %d 张 (%.1f%%) 短边 >= 1080 ⇒ 数据偏少' % (ok1080, 100.0 * ok1080 / n))
else:
    print('  ❌ 没有短边 >= 1080 的图 ⇒ 【1080p 必须走超分路线，不能靠重训解码器】')
    ok512 = sum(1 for s in short if s >= 512)
    print('     短边 >= 512 的有 %d 张 (%.1f%%)' % (ok512, 100.0 * ok512 / n))
print('DONE')
