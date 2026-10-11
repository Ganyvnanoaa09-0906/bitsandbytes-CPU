# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.stdout.reconfigure(encoding='utf-8')
D = r"D:\work\bitsandbytes-CPU\i5build\loop_ab_eval"
arms = ['L1', 'random']; Ls = [1, 2, 3]
imgs = {}
for a in arms:
    for L in Ls:
        p = os.path.join(D, '%s_L%d.png' % (a, L))
        if os.path.exists(p):
            imgs[(a, L)] = Image.open(p).convert('RGB')
W = 512
tiles = []
for a in arms:
    for L in Ls:
        im = imgs.get((a, L))
        if im is None: continue
        r = W / im.width
        tiles.append(((a, L), im.resize((W, max(1, int(im.height * r))))))
th = max(t[1].height for t in tiles) + 30
sheet = Image.new('RGB', (W * 3, th * 2), 'white')
dr = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
except Exception:
    font = ImageFont.load_default()
for i, ((a, L), im) in enumerate(tiles):
    x = (i % 3) * W; y = (i // 3) * th
    sheet.paste(im, (x, y + 30))
    dr.rectangle([x, y, x + W - 1, y + 29], fill=(20, 20, 20))
    dr.text((x + 8, y + 4), 'Arm %s  |  infer L=%d' % (a, L), fill=(255, 255, 0), font=font)
sheet.save(os.path.join(D, 'contact_sheet.png'))
print('=== 基础统计（crude proxies）===')
for a in arms:
    for L in Ls:
        im = imgs.get((a, L))
        if im is None: continue
        g = im.convert('L')
        arr = np.asarray(g, dtype=np.float32)
        lap = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        print('%s L=%d | mean %.1f | std %.1f | edge_var %.1f' % (a, L, arr.mean(), arr.std(), lap.var()))
print('saved', os.path.join(D, 'contact_sheet.png'))
