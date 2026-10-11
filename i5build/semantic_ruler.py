"""Semantic ruler: CLIP zero-shot, because hand-crafted statistics failed.

Why this exists: the previous ruler (edge / high-frequency / Laplacian variance)
PASSED a case the eye rejects -- GRN+L=3 scored edge 0.0253 vs the known-good
75%-prefix reference at 0.0186, yet the images are still abstract paintings, not
subjects. Edge density is high in any painterly texture. So statistics can only
catch collapse/diversity; "is there a subject" needs semantics.

This asks CLIP directly: for each sample, which text description fits best?
   person / anime character  -> "has a subject"
   abstract painting / blur / solid colour -> "no subject"

Discipline: validate the ruler on the two calibration sets BEFORE using it.
   GOOD = 75% prefix continuations (subjects were visible)
   BAD  = naive 0% cold start (blank / black)
If it cannot rank those correctly, its verdicts on the variants are worthless.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)
OUT = r'D:\work\bitsandbytes-CPU\i5build\cold_metric'

SUBJECT = ['a photo of a person', 'an anime character', 'a portrait',
           'a drawing of a girl', 'a person wearing clothes']
NO_SUBJECT = ['an abstract painting', 'a blurry image', 'a plain solid color',
              'a texture', 'an empty background', 'random noise']
PROMPTS = SUBJECT + NO_SUBJECT


def panels(path):
    """Split a contact sheet into square panels (handles portrait sheets)."""
    im = Image.open(path).convert('RGB')
    w, h = im.size
    for n in (8, 4, 6, 3, 2, 1):
        for cols in range(1, n + 1):
            if n % cols:
                continue
            rows = n // cols
            if w % cols == 0 and h % rows == 0 and w // cols == h // rows:
                pw, ph = w // cols, h // rows
                return [im.crop(((i % cols) * pw, (i // cols) * ph,
                                 (i % cols) * pw + pw, (i // cols) * ph + ph))
                        for i in range(n)]
    raise ValueError('cannot infer grid for %dx%d' % (w, h))


def main():
    print('loading CLIP ...')
    name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(name)
    proc = CLIPProcessor.from_pretrained(name)
    model.eval()

    with torch.no_grad():
        tin = proc(text=PROMPTS, return_tensors='pt', padding=True)
        tf = model.get_text_features(**tin)
        # transformers 5.x returns a ModelOutput here, not a tensor (4.x returned
        # a tensor). Unwrap if needed -- this cost one run of debugging.
        if hasattr(tf, 'pooler_output'):
            tf = tf.pooler_output
        elif not torch.is_tensor(tf):
            tf = tf[0]
        tf = tf / tf.norm(dim=-1, keepdim=True)

    sets = [
        ('GOOD 75%前缀', 'calib_good.png'),
        ('BAD  0%朴素', 'calib_bad.png'),
        ('GRN  L=3', 'grnL3.png'),
        ('GRN  L=8', 'grnL8.png'),
        ('朴素 L=1', 'loopL1.png'),
        ('朴素 L=5', 'loopL5.png'),
        ('GRN 冷启动(旧)', '../ar3_test/cold_grn.png'),
        ('朴素冷启动(旧)', '../ar3_test/f75_f00.png'),
    ]
    print('\n%-18s %8s %8s %8s   %s' % ('样本集', '主体分', '无主体分', '主体占比', '最匹配的描述'))
    print('-' * 88)
    calib = {}
    for label, rel in sets:
        p = os.path.join(OUT, rel)
        if not os.path.exists(p):
            print('%-18s （没有这张图，跳过）' % label)
            continue
        ims = panels(p)
        with torch.no_grad():
            inf = proc(images=ims, return_tensors='pt')['pixel_values']
            f = model.get_image_features(pixel_values=inf)
            if hasattr(f, 'pooler_output'):
                f = f.pooler_output
            elif not torch.is_tensor(f):
                f = f[0]
            f = f / f.norm(dim=-1, keepdim=True)
            # CLIP needs its learned temperature: similarity * logit_scale (~100)
            # then softmax. Without it the softmax comes out uniform -- the tell
            # was that every set scored exactly 5/11 = 0.4545 vs 6/11 = 0.5455,
            # i.e. the prompt-count ratio, meaning zero discrimination. The
            # calibration step is what caught this.
            scale = float(model.logit_scale.exp()) if hasattr(model, 'logit_scale') \
                else 100.0
            sim = ((f @ tf.T) * scale).softmax(dim=-1)
        probs = sim.mean(0).numpy()
        subj = float(probs[:len(SUBJECT)].sum())
        nosub = float(probs[len(SUBJECT):].sum())
        best = PROMPTS[int(np.argmax(probs))]
        # per-sample: fraction that prefers a subject prompt
        per = (sim[:, :len(SUBJECT)].sum(1) > sim[:, len(SUBJECT):].sum(1))
        frac = float(per.float().mean())
        calib[label] = (subj, nosub, frac)
        print('%-18s %8.3f %8.3f %7.0f%%   %s'
              % (label, subj, nosub, frac * 100, best))

    print('\n' + '=' * 88)
    g = calib.get('GOOD 75%前缀')
    b = calib.get('BAD  0%朴素')
    if g and b:
        print('尺子自检（先验已知好/坏）：')
        print('  GOOD 主体占比 %.0f%%  /  BAD 主体占比 %.0f%%' % (g[2] * 100, b[2] * 100))
        if g[2] > b[2] and g[0] > b[0]:
            print('  ⇒ 尺子可用 ✓ 方向正确（好样本更偏"有主体"）')
        else:
            print('  ⇒ ⚠️ 尺子方向不对或区分不出，别信下面的结论 ✗')
    print('\n判读：主体占比明显高于 BAD 的变体 ⇒ 真的出现了主体；'
          '\n      只是在 BAD 附近打转 ⇒ 仍然是没有主体的抽象图。')


if __name__ == '__main__':
    main()
