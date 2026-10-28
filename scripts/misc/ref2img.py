"""ref2img.py -- reference image in, new image out, on the CPU-only AR model.

Why this needs no change to sample_ar.py: CLIP places text and image embeddings
in the same 512-d space, and the sampler already accepts a precomputed
(N, text_dim) condition tensor via --cond-pt. So an image's CLIP embedding can be
handed to it exactly as a text embedding would be. The model never learns where
the vector came from.

What this driver does:
  1. CLIP-encode the reference image (image tower, pooled output, L2-normalised
     to match the training conditions -- cond_clip_29113.pt has mean norm 1.0000)
  2. write that single row out as a condition tensor
  3. invoke sample_ar.py with --cond-pt and --cond-row 0, so every sample uses
     the reference's embedding
  4. compose reference and samples side by side

Honest expectation: adaln conditioning on a *pooled* embedding steers semantics,
not pixels. The output should be related in subject/style, not a pixel-level
variant of the reference. If identity preservation is needed later, that calls
for patch-level conditioning (cond_mode 'cross' over image tokens), which is a
different training run.

Usage:
  python ref2img.py --ref photo.png --ar <cond ckpt> --vqvae <vqvae ckpt> \
                    --out compare.png --n 4
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import argparse
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def clip_image_embed(path: str, model_id: str = 'openai/clip-vit-base-patch32') -> torch.Tensor:
    """(1, 512) L2-normalised CLIP image embedding for one file.

    Mirrors clip_score_eval.py:279-281 exactly rather than trusting memory: there,
    `proc(images=..., return_tensors='pt')` returns a BatchFeature that must be
    splatted into get_image_features(**ii). My first version treated the processor
    as returning a tensor and called .unsqueeze on it, which raised
    KeyError: 'unsqueeze' -- the processor is dict-like in transformers 5.x.
    """
    from PIL import Image
    import torch.nn.functional

    sys.path.insert(0, HERE)
    from scripts.lib.clip_score_eval import load_clip, clip_feat

    model, proc = load_clip(model_id)
    img = Image.open(path).convert('RGB')
    ii = proc(images=[img], return_tensors='pt')
    with torch.no_grad():
        out = model.get_image_features(**ii)
    f = clip_feat(model, out, 'image')      # unpacks the transformers-5 ModelOutput
    f = F.normalize(f.float(), dim=-1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='reference image')
    ap.add_argument('--ar', required=True, help='conditional AR checkpoint')
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--out', default='ref2img.png')
    ap.add_argument('--n', type=int, default=4, help='how many samples from this reference')
    ap.add_argument('--guidance', type=float, default=3.0,
                    help='CFG scale; only used when the checkpoint has cond_mode != none')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loop_L', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()

    print('[1/4] CLIP 编码参考图 …')
    emb = clip_image_embed(a.ref)
    print('      形状 %s  范数 %.4f' % (tuple(emb.shape), float(emb.norm())))

    cond_path = os.path.join(HERE, '_ref_cond.pt')
    torch.save(emb, cond_path)
    print('      条件张量已写: %s' % cond_path)

    grid = os.path.join(HERE, '_ref_grid.png')
    cmd = [sys.executable, os.path.join(HERE, 'sample_ar.py'),
           '--ar', a.ar, '--vqvae', a.vqvae, '--out', grid,
           '--n', str(a.n), '--cond-pt', cond_path, '--cond-row', '0',
           '--guidance', str(a.guidance), '--temperature', str(a.temperature),
           '--top-k', str(a.top_k), '--seed', str(a.seed),
           '--threads', str(a.threads)]
    if a.loop_L:
        cmd += ['--loop_L', str(a.loop_L)]
    print('[2/4] 采样: %s' % ' '.join(cmd[1:]))
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print('      采样失败，退出码 %d' % r.returncode)
        return r.returncode
    if not os.path.exists(grid):
        print('      采样没产出 %s' % grid)
        return 1

    print('[3/4] 拼对照图 …')
    from PIL import Image
    ref = Image.open(a.ref).convert('RGB')
    g = Image.open(grid).convert('RGB')
    h = max(ref.height, g.height)
    ref2 = ref.resize((int(ref.width * h / ref.height), h))
    g2 = g.resize((int(g.width * h / g.height), h))
    out = Image.new('RGB', (ref2.width + g2.width + 12, h), (255, 255, 255))
    out.paste(ref2, (0, 0))
    out.paste(g2, (ref2.width + 12, 0))
    out.save(a.out)
    print('[4/4] 已写出 %s   （左 = 参考图，右 = %d 张生成）' % (a.out, a.n))
    print()
    print('★ 判据是【你的眼睛】：生成的图与参考图在主体/风格上有没有关系。')
    print('  adaln 作用在 pooled 嵌入上，管的是语义不是像素 ⇒ 不要期待像素级相似。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
