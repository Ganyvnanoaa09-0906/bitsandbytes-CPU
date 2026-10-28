# -*- coding: utf-8 -*-
"""probe_clip_variants.py — 到底哪种取法才能得到 CLIP 对比空间里的嵌入？

背景：transformers 5.x 下 `CLIPModel.get_text_features()` 返回
`BaseModelOutputWithPooling`（只有 last_hidden_state / pooler_output，没有 text_embeds）。
按"取 pooler_output 再手工过投影层"的做法，图文余弦只有 0.02~0.08，
远低于正常匹配的 0.2~0.3 ⇒ 说明取法仍不对。

本脚本把候选取法两两组合，直接看哪一种能把"匹配对"的余弦抬到正常量级。
判据：**正确取法的匹配余弦应在 0.2~0.35**（CLIP ViT-B/32 的典型值）。
"""
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch as T
import torch.nn.functional as F
from PIL import Image

from scripts.lib.clip_score_eval import load_clip

TXT = "a photo of a woman in a blue maid outfit with twintails"


def variants(out, model, kind):
    """返回 {名字: 张量} 的候选嵌入取法。"""
    import torch as T
    d = {}
    if T.is_tensor(out):
        d['tensor'] = out
        return d
    lhs = getattr(out, 'last_hidden_state', None)
    pool = getattr(out, 'pooler_output', None)
    proj = getattr(model, 'text_projection' if kind == 'text' else 'image_projection', None)

    if pool is not None:
        d['pool'] = pool
        if proj is not None:
            d['proj(pool)'] = proj(pool)
    if lhs is not None:
        if lhs.dim() == 3:                       # 文本: (B, L, C)
            d['lhs[:,0]'] = lhs[:, 0]
            d['lhs.mean'] = lhs.mean(1)
            if proj is not None:
                d['proj(lhs[:,0])'] = proj(lhs[:, 0])
                d['proj(lhs.mean)'] = proj(lhs.mean(1))
        else:                                    # 图像: (B, C, H, W)
            d['lhs.mean_hw'] = lhs.mean([2, 3])
            if proj is not None:
                d['proj(lhs.mean_hw)'] = proj(lhs.mean([2, 3]))
    return d


def main():
    model, proc = load_clip('openai/clip-vit-base-patch32')
    model.eval()

    d = r'.autopilot\clean_tu\images'
    p = sorted(os.listdir(d))[0]
    with Image.open(os.path.join(d, p)) as im:
        img = im.convert('RGB')

    with torch.no_grad():
        ti = proc(text=[TXT], return_tensors='pt', padding=True, truncation=True)
        ii = proc(images=[img], return_tensors='pt')
        tout = model.get_text_features(**ti)
        iout = model.get_image_features(**ii)

    tv = variants(tout, model, 'text')
    iv = variants(iout, model, 'image')
    print('text variants:', list(tv.keys()))
    print('image variants:', list(iv.keys()))
    print()
    print('%-18s %-18s %8s' % ('text', 'image', 'cos'))
    print('-' * 48)
    best = (None, None, -9)
    for tn, t in tv.items():
        for inn, i in iv.items():
            if t.shape[-1] != i.shape[-1]:
                continue
            c = float(F.normalize(t.float(), dim=-1) @ F.normalize(i.float(), dim=-1).T)
            print('%-18s %-18s %8.4f' % (tn, inn, c))
            if c > best[2]:
                best = (tn, inn, c)
    print()
    print('最大: text=%s image=%s cos=%.4f' % best)
    print('⇒ 正常 CLIP 匹配对应在 0.2~0.35。取那个组合。')


if __name__ == '__main__':
    main()
