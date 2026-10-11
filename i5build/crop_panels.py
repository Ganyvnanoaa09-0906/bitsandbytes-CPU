"""Crop the panels out of a sampler contact sheet and upscale them.

Why: the contact sheets are 3 panels wide; at preview size I dismissed the
cold-start (frac=0) panels as "mush", but the user spotted a human figure in
one. Read them at 2x before judging.
"""
import os
import sys

from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
HERE = r'D:\work\bitsandbytes-CPU\i5build\ar3_test'


def crop_strip(path, out_prefix, scale=2):
    im = Image.open(path)
    w, h = im.size
    n = round(w / h)                      # square panels side by side
    pw = w // n
    print('%s  %dx%d -> %d panels of %dx%d' % (os.path.basename(path), w, h, n, pw, h))
    outs = []
    for i in range(n):
        p = im.crop((i * pw, 0, (i + 1) * pw, h))
        p = p.resize((pw * scale, h * scale), Image.LANCZOS)
        o = os.path.join(HERE, '%s_p%d.png' % (out_prefix, i))
        p.save(o)
        outs.append(o)
        print('   -> %s' % os.path.basename(o))
    # also a single stacked sheet at 2x so one look covers all panels
    sheet = Image.new('RGB', (pw * scale * n, h * scale), 'white')
    for i, o in enumerate(outs):
        sheet.paste(Image.open(o), (i * pw * scale, 0))
    so = os.path.join(HERE, '%s_2x.png' % out_prefix)
    sheet.save(so)
    print('   sheet -> %s' % os.path.basename(so))


for f, pre in (('f75_f00.png', 'cold0'), ('ar3_uncond.png', 'uncond')):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        crop_strip(p, pre)
    else:
        print('missing:', p)
