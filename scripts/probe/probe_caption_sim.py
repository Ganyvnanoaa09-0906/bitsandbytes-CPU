# -*- coding: utf-8 -*-
"""probe_caption_sim.py — 量一下 caption 用的 CLIP 相似度到底落在什么区间。

为什么要先量：threshold 模式在 0.24/0.26 下【一个标签都不出】，
而正常 CLIP 图文匹配的余弦相似度通常在 0.2~0.3。
如果实测最大值只有 0.1 量级，那说明【嵌入本身不对】（例如少了投影层），
此时无论怎么调阈值都是在错误的尺度上拧螺丝。
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
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.lib.clip_score_eval import load_clip, clip_feat
from scripts.lib.auto_caption import TAG_GROUPS, TEMPLATES


def main():
    model, proc = load_clip('openai/clip-vit-base-patch32')

    # 构造标签库（与 auto_caption 同口径）
    bank = {}
    with torch.no_grad():
        for grp, labels in TAG_GROUPS.items():
            embs = []
            for lab in labels:
                e = None
                for t in TEMPLATES:
                    inp = proc(text=[t.format(lab)], return_tensors='pt',
                               padding=True, truncation=True)
                    f = clip_feat(model, model.get_text_features(**inp), 'text')
                    f = F.normalize(f.float(), dim=-1)
                    e = f if e is None else e + f
                embs.append(F.normalize(e / len(TEMPLATES), dim=-1))
            bank[grp] = (labels, torch.cat(embs, 0))

    # 取一张图
    d = r'.autopilot\clean_tu\images'
    p = sorted(os.listdir(d))[0]
    with Image.open(os.path.join(d, p)) as im:
        img = im.convert('RGB')
    with torch.no_grad():
        inp = proc(images=[img], return_tensors='pt')
        fe = clip_feat(model, model.get_image_features(**inp), 'image')
        fe = F.normalize(fe.float(), dim=-1)

    print('image:', p)
    print('%-14s %8s %8s   %s' % ('group', 'max', '2nd', 'best label'))
    print('-' * 74)
    allmax = []
    for grp, (labels, E) in bank.items():
        sim = (fe @ E.T)[0]
        v, i = sim.topk(min(3, len(labels)))
        allmax.append(float(v[0]))
        print('%-14s %8.4f %8.4f   %s' % (grp, v[0], v[1] if len(v) > 1 else float('nan'),
                                          labels[int(i[0])]))
    print()
    print('所有组的 max 相似度: 最大 %.4f  最小 %.4f  中位 %.4f'
          % (max(allmax), min(allmax), sorted(allmax)[len(allmax) // 2]))
    print('⇒ 若这些值在 0.15~0.30，说明嵌入正常，把 --thr 设成"中位数附近"即可；')
    print('   若只有 0.0x，说明嵌入仍不对（别调阈值，先修嵌入）。')


if __name__ == '__main__':
    main()
