"""Build a 100-image subset for the memorisation test, plus the reference it should reproduce.

The question this answers has never been asked: can the model fit at all? Every hypothesis so far
has been on the generalisation side -- more data, more steps, a smaller model -- and none of them
checked the other side, whether the architecture and the sampling procedure can reproduce images
they have seen.

The design uses the AR objective deliberately, because an AR run on all 14432 images already
produced flat colour fields at 6000 steps. Running the same objective on 100 images changes exactly
one variable: data volume. If the samples become recognisable versions of those 100, the architecture
is fine and the problem is generalisation, and the fix is more data or a smaller model. If they stay
flat, something structural is wrong with the model or the sampler, and no amount of data will help.

Rows come in pairs -- original and horizontal flip -- so 100 images is 200 rows. The training batch
is 2, so an epoch is 100 steps, and 2000 steps is 20 epochs. A 30M-parameter model can memorise 100
images in far less than that, so if it has not by 2000 steps it never will.

Also written: a sheet of those 100 images' VQGAN reconstructions, which is the target. 7.48
established that the tokenizer can represent these images faithfully, so this sheet is what a
successful fit should look like.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
OUT_TOK = r'D:\work\bitsandbytes-CPU\tokens_overfit100.pt'
OUT_REF = r'D:\work\bitsandbytes-CPU\overfit100_reference.png'
N_IMAGES = 100

t = torch.load(TOK, map_location='cpu')
print('  完整 tokens: %s' % (tuple(t.shape),))

# pairs: row 2i and 2i+1 are the same image and its flip
idx = torch.arange(0, N_IMAGES * 2)
sub = t[idx].contiguous()
torch.save(sub, OUT_TOK)
print('  子集已写: %s  %s  (%.1f MB)' % (OUT_TOK, tuple(sub.shape), os.path.getsize(OUT_TOK) / 1e6))
print('  ⇒ 100 张图 = %d 行 · batch 2 ⇒ 每个 epoch %d 步' % (sub.shape[0], sub.shape[0] // 2))

# reference sheet: decode the same rows through the VQGAN, which is what a successful fit
# should reproduce. Uses the decoder only, one image at a time, so it does not disturb training.
try:
    import numpy
    from PIL import Image

    ck = torch.load(r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt',
                    map_location='cpu', weights_only=False)
    print('  VQ ckpt: ch=%s n_codes=%s size=%s grid=%s' %
          (ck.get('ch'), ck.get('n_codes'), ck.get('size'), ck.get('grid')))

    # find the loader the project already uses rather than assuming an architecture
    sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
    vqvae = None
    for modname, fname in (('small_vqvae', 'vqvae_from_ckpt'),
                           ('train_vqvae_gan', 'vqvae_from_ckpt'),
                           ('ms_vq', 'vqvae_from_ckpt')):
        try:
            mod = __import__(modname)
            if hasattr(mod, fname):
                got = getattr(mod, fname)(ck)
                # vqvae_from_ckpt returns (model, cfg) or just the model depending on the version
                if isinstance(got, (tuple, list)):
                    print('  %s.%s 返回 %d 项,取第一项' % (modname, fname, len(got)))
                    for cand in got:
                        if hasattr(cand, 'eval') and hasattr(cand, 'decode'):
                            vqvae = cand
                            break
                else:
                    vqvae = got
                if vqvae is not None:
                    print('  用 %s.%s 加载 VQGAN' % (modname, fname))
                    break
        except Exception:
            continue
    if vqvae is None:
        print('  (找不到 vqvae_from_ckpt,跳过参考图)')
    else:
        vqvae.eval()
        torch.set_num_threads(1)
        picks = list(range(0, N_IMAGES * 2, 2))[:16]   # 16 originals
        imgs = []
        with torch.no_grad():
            for r in picks:
                z = vqvae.quantize(sub[r:r + 1].long()) if hasattr(vqvae, 'quantize') else None
                if z is None:
                    break
                rec = vqvae.decode(z)
                arr = rec[0].clamp(-1, 1).add(1).div(2).mul(255).byte()
                arr = arr.permute(1, 2, 0).numpy()
                imgs.append(Image.fromarray(arr))
        if imgs:
            w, h = imgs[0].size
            sheet = Image.new('RGB', (4 * w, 4 * h))
            for i, im in enumerate(imgs[:16]):
                sheet.paste(im, ((i % 4) * w, (i // 4) * h))
            sheet.save(OUT_REF)
            print('  参考图已写: %s  (%dx%d)' % (OUT_REF, sheet.size[0], sheet.size[1]))
except Exception as e:
    print('  参考图生成失败（不影响子集）: %s' % str(e)[:120])
