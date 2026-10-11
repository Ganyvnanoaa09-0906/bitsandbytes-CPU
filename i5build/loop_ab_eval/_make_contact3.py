# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.stdout.reconfigure(encoding='utf-8')
D = r"D:\work\bitsandbytes-CPU\i5build\loop_ab_eval"
rows = [('base(24000)', 'base'), ('Arm L1', 'L1'), ('Arm random', 'random')]
Ls = [1, 2, 3]
W = 512
tiles = []
for rname, rtag in rows:
    for L in Ls:
        p = os.path.join(D, '%s_L%d.png' % (rtag, L))
        if not os.path.exists(p): 
            print('missing', p); continue
        im = Image.open(p).convert('RGB')
        t = im.resize((W, max(1, int(im.height * W / im.width))))
        tiles.append(((rname, L), t, p))
th = max(t[1].height for t in tiles) + 30
sheet = Image.new('RGB', (W * 3, th * 3), 'white')
dr = ImageDraw.Draw(sheet)
try: font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
except Exception: font = ImageFont.load_default()
for i, ((rname, L), im, p) in enumerate(tiles):
    x = (i % 3) * W; y = (i // 3) * th
    sheet.paste(im, (x, y + 30))
    dr.rectangle([x, y, x + W - 1, y + 29], fill=(20, 20, 20))
    dr.text((x + 8, y + 4), '%s  |  infer L=%d' % (rname, L), fill=(255, 255, 0), font=font)
sheet.save(os.path.join(D, 'contact_sheet3.png'))
print('saved', os.path.join(D, 'contact_sheet3.png'))
print('\n=== 平坦块占比（越低=越不糊）===')
for rname, rtag in rows:
    line = []
    for L in Ls:
        p = os.path.join(D, '%s_L%d.png' % (rtag, L))
        if not os.path.exists(p): line.append('  -   '); continue
        a = np.asarray(Image.open(p).convert('L'), dtype=np.float32)
        stds = []
        for y in range(0, a.shape[0]-15, 16):
            for x in range(0, a.shape[1]-15, 16):
                stds.append(float(a[y:y+16, x:x+16].std()))
        stds = np.array(stds)
        line.append('%.3f' % (stds < 8).mean())
    print('%-14s | %s' % (rname, ' | '.join(line)))
