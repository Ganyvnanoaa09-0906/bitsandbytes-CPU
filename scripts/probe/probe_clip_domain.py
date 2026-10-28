# -*- coding: utf-8 -*-
"""probe_clip_domain.py — 诊断 CLIP 在本数据集上到底能不能打标

若所有标签相似度都在 0.18~0.25 的窄带里 → CLIP 对这个域（动漫/写实混合、
带水印残留、COS 人像）没有判别力，zero-shot 打标不可用。
那就得换方案：要么用真实 captioning 模型（BLIP），要么放弃文本条件。
"""
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
import os, sys, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.clip_score_eval import load_clip
from scripts.lib.auto_caption import build_text_bank, list_images

torch.set_num_threads(4)
root = r'.autopilot\clean_nf\images'
paths = list_images(root)
print('图片池 %d 张' % len(paths))

model, proc = load_clip('openai/clip-vit-base-patch32')
bank = build_text_bank(model, proc)
print('标签库就绪\n')

sel = [paths[i] for i in np.linspace(0, len(paths) - 1, 6).astype(int)]
imgs = []
for p in sel:
    with Image.open(p) as im:
        imgs.append(im.convert('RGB'))

with torch.no_grad():
    fe = model.get_image_features(**proc(images=imgs, return_tensors='pt'))
    fe = torch.nn.functional.normalize(fe.float(), dim=-1)

print('=== 每组内 top3 相似度（看是否有区分度）===')
for grp, (labels, emb) in bank.items():
    sims = (fe @ emb.t())          # (n_img, n_label)
    print('\n[%s]' % grp)
    for k, lab in enumerate(labels):
        v = sims[:, k]
        print('   %-44s 均值 %.3f  范围 %.3f~%.3f  跨度 %.3f'
              % (lab[:44], float(v.mean()), float(v.min()), float(v.max()),
                 float(v.max() - v.min())))

# 关键判据：同一图内，组内 第一名 vs 第二名 的差距
print('\n\n=== 判别力判据（组内 top1 - top2 的差距）===')
gaps = []
for grp, (labels, emb) in bank.items():
    if len(labels) < 2:
        continue
    sims = (fe @ emb.t())
    top2 = sims.topk(2, dim=1).values
    g = (top2[:, 0] - top2[:, 1])
    gaps.append(float(g.mean()))
    print('   %-14s 平均差距 %.4f   最小 %.4f' % (grp, float(g.mean()), float(g.min())))
print('\n平均组内差距 %.4f' % (sum(gaps) / len(gaps)))
print('参照: auto_caption 的 margin 默认 0.012')
print('  ⇒ 差距远小于 0.012 = 完全无判别力，argmax 是随机选')
print('  ⇒ 差距明显大于 margin = 有判别力，打标可用')
