"""encode_dataset.py -- turn the image folder into AR training tokens.

Why this exists: the 16x16 token file used by the AR was produced by a script
that is no longer in the tree (only its consumers remain), which is why the
image-to-row mapping had to be guessed. This one writes the mapping next to the
tokens so the next person does not have to guess.

Two deliberate choices:

  * Preprocessing matches ImageFolderLite exactly -- Resize(size), ToTensor,
    Normalize([0.5]*3) -- minus the random flip, applied here as an explicit
    second pass. The original file had 29113 rows for 14432 images, i.e. almost
    exactly 2x, which is what image + horizontal flip gives (28864).

  * Grayscale output is deliberately NOT used; the original .jpg files are RGB
    and the tokenizer was trained on RGB.

Usage:
  python encode_dataset.py --ckpt i5build/vqgan32 --data <imgdir> \
      --out tokens_32x32_full.pt --size 256 --flip
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from small_vqvae import vqvae_from_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='tokenizer checkpoint or run dir')
    ap.add_argument('--data', required=True, help='image directory')
    ap.add_argument('--out', required=True, help='output .pt of tokens')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--flip', action='store_true', help='also encode the horizontal flip')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ck = a.ckpt
    if os.path.isdir(ck):
        ck = os.path.join(ck, 'ckpt_last.pt')
    model, vk = vqvae_from_ckpt(ck)
    model.eval()
    grid = a.size // (2 ** len(model.ch))
    print('tokenizer %s' % ck)
    print('ch=%s => grid %dx%d = %d token/图' % (tuple(model.ch), grid, grid, grid * grid))

    files = sorted(glob.glob(os.path.join(a.data, '*.jpg')) +
                   glob.glob(os.path.join(a.data, '*.png')))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit('no images in %s' % a.data)
    print('图片 %d 张，flip=%s => 预计 %d 行' %
          (len(files), a.flip, len(files) * (2 if a.flip else 1)))

    toks, paths, flips = [], [], []
    t0 = time.time()
    buf, bufmeta = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf)                       # (B,3,H,W) in [-1,1]
        with torch.no_grad():
            _, idx, _ = model(x)                   # idx: (B,H,W) or (B, N)
        idx = idx.reshape(idx.shape[0], -1).to(torch.int32)
        toks.append(idx)
        paths.extend(bufmeta)
        buf.clear()
        bufmeta.clear()

    with torch.no_grad():
        for i, f in enumerate(files):
            with Image.open(f) as im:
                im = im.convert('RGB').resize((a.size, a.size), Image.BICUBIC)
                arr = np.asarray(im).copy()        # copy: torch rejects read-only
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            t = (t - 0.5) / 0.5
            buf.append(t)
            bufmeta.append({'file': os.path.basename(f), 'flip': False})
            if a.flip:
                buf.append(torch.flip(t, dims=[2]))
                bufmeta.append({'file': os.path.basename(f), 'flip': True})
            if len(buf) >= a.batch:
                flush()
            if (i + 1) % 1000 == 0:
                el = time.time() - t0
                done = (i + 1) * (2 if a.flip else 1)
                print('  %d/%d 张  %d 行  %.1f 分钟  预计还要 %.1f 分钟'
                      % (i + 1, len(files), done, el / 60,
                         el / (i + 1) * (len(files) - i - 1) / 60), flush=True)
        flush()

    tokens = torch.cat(toks, 0)
    torch.save(tokens, a.out)
    meta_path = a.out + '.paths.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(paths, f, ensure_ascii=False)
    print()
    print('写出 %s  %s  dtype=%s' % (a.out, tuple(tokens.shape), tokens.dtype))
    print('写出 %s  （image->row 的映射，别再让它丢失）' % meta_path)
    used = int(tokens.unique().numel())
    print('用到的码: %d/%d （%.1f%%）'
          % (used, model.vq.codebook.weight.shape[0],
             100.0 * used / model.vq.codebook.weight.shape[0]))


if __name__ == '__main__':
    main()
