"""vq_recon_compare.py -- originals on top, VQ reconstruction below.

This is the gate for the 32x32 path. The whole reason for going to 1024 tokens is
that 256 tokens cannot hold enough detail to be watchable, and whether 1024 is
enough is a question about images, not about a loss number. So: show them.

Two things make the comparison honest:

  * The originals on the top row are resized to the SAME resolution the tokenizer
    sees, not shown at full size. Comparing a 512px original against a 256px
    reconstruction would make the tokenizer look worse than it is, and that is
    the kind of confound this project keeps tripping over.

  * Preprocessing matches ImageFolderLite exactly (resize, ToTensor,
    Normalize([0.5]*3)) minus the random flip, which does not belong at eval
    time. Loading the model goes through vqvae_from_ckpt so the architecture comes
    from the checkpoint rather than from an assumption about ch.

Usage:
  python vq_recon_compare.py --ckpt i5build/vq32 --data <imgdir> --out cmp.png --n 4
"""
import argparse
import glob
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from small_vqvae import vqvae_from_ckpt  # noqa: E402


def pick_ckpt(path: str) -> str:
    if os.path.isfile(path):
        return path
    last = os.path.join(path, 'ckpt_last.pt')
    if os.path.exists(last):
        return last
    cands = sorted(glob.glob(os.path.join(path, 'ckpt_*.pt')),
                   key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    if not cands:
        raise SystemExit('no checkpoint under %s' % path)
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint file or the run directory')
    ap.add_argument('--data', required=True, help='image directory')
    ap.add_argument('--out', default='vq_recon_compare.png')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--size', type=int, default=0, help='0 = infer from the checkpoint')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    ck = pick_ckpt(a.ckpt)
    model, vk = vqvae_from_ckpt(ck)
    ch = tuple(model.ch)
    down = 2 ** len(ch)
    size = a.size or int(vk.get('size') or 0)
    if not size:
        # size is not recorded in older checkpoints; the grid is what matters and
        # the user passes --size in that case
        size = 256
    grid = size // down
    print('checkpoint %s' % ck)
    print('ch=%s ⇒ %dx downsample ⇒ 网格 %dx%d ⇒ %d token（每 token %d 像素）'
          % (ch, down, grid, grid, grid * grid, (size * size) // (grid * grid)))

    files = sorted(glob.glob(os.path.join(a.data, '*.jpg')) +
                   glob.glob(os.path.join(a.data, '*.png')))
    if not files:
        raise SystemExit('no images in %s' % a.data)
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]

    model.eval()
    origs, recons = [], []
    with torch.no_grad():
        for f in files:
            with Image.open(f) as im:
                im = im.convert('RGB').resize((size, size), Image.BICUBIC)
                t = torch.from_numpy(
                    __import__('numpy').asarray(im)).permute(2, 0, 1).float() / 255.0
            t = (t - 0.5) / 0.5                      # Normalize([0.5]*3,[0.5]*3)
            x = t.unsqueeze(0)
            xr, idx, _ = model(x)
            origs.append(t)
            recons.append(xr[0].clamp(-1, 1))
            used = int(idx.unique().numel())
            print('  %-28s 用了 %4d/%d 个码' % (os.path.basename(f), used, model.vq.codebook.weight.shape[0]))

    def to_img(t):
        t = ((t + 1) / 2).clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(arr)

    W = size * a.n + 8 * (a.n - 1)
    out = Image.new('RGB', (W, size * 2 + 24), (255, 255, 255))
    for i, (o, r) in enumerate(zip(origs, recons)):
        out.paste(to_img(o), (i * (size + 8), 0))
        out.paste(to_img(r), (i * (size + 8), size + 24))
    out.save(a.out)
    print()
    print('上排 = 原图（已缩到 tokenizer 看到的分辨率 %d，避免不公平的对比）' % size)
    print('下排 = VQ 重建')
    print('写出 %s  %s' % (a.out, out.size))


if __name__ == '__main__':
    main()
