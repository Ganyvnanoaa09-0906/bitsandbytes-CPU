"""Compare the two A/B arms: same prompt, same seed, only --lora differs.

The question this answers: the LoRA was trained on cosplay PHOTOS, and the first
generation came out as an anime portrait. Those two facts are consistent with
either (a) the LoRA is active and the base model's anime prior dominates, or
(b) the LoRA never loaded and the base model produced everything. A single run
cannot tell them apart, so the comparison puts the two arms side by side.

Reads the frame directories the driver moved each arm into, builds one sheet per
arm and a combined sheet, and reports the per-arm frame-difference statistic the
sampler itself prints (its "VALID" oracle: frames must actually change).
"""
import glob
import os
import sys

from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\work\cloud_results\anime_e2e'
OUT = r'D:\work\bitsandbytes-CPU\i5build\ab_lora_compare.png'


def sheet(dirname, cols=4):
    d = os.path.join(ROOT, dirname)
    fs = sorted(glob.glob(os.path.join(d, '*.png')))
    if not fs:
        print('  %s: 没有帧' % dirname)
        return None
    step = max(1, len(fs) // 8)
    sel = fs[::step][:8]
    ims = [Image.open(f).convert('RGB') for f in sel]
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    s = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * w, (i // cols) * h))
    print('  %s: %d 帧，取 %d 帧拼图' % (dirname, len(fs), len(ims)))
    return s


a = sheet('lr_frames_nolora')
b = sheet('lr_frames_lora')
if a is None or b is None:
    print()
    print('  ⇒ 两臂还没都跑完，稍后再试')
    sys.exit(0)

W = max(a.width, b.width)
out = Image.new('RGB', (W, a.height + b.height + 8), (255, 255, 255))
out.paste(a, (0, 0))
out.paste(b, (0, a.height + 8))
out.save(OUT)
print()
print('  上排 = 不带 LoRA，下排 = 带训好的 LoRA')
print('  写出 %s  %s' % (OUT, out.size))
