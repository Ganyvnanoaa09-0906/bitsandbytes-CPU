# -*- coding: utf-8 -*-
"""auto_caption.py - 用本地 CLIP 做 zero-shot 自动打标，为 AR 模型提供真实文本条件。

为什么需要
----------
v1 训练时 `text = torch.zeros(...)`，模型其实是无条件生成。用户的评审点 1
（条件注入太弱）前提就不成立 —— 得先有每张图的文本描述。

做法
----
1. 准备一个面向 cos/人像的标签词表
2. 对每个标签构造若干模板句，用 CLIP 文本编码器得到文本嵌入（只需算一次）
3. 每张图取 CLIP 图像嵌入，与全部标签嵌入点积
4. 分数超过阈值的标签收集起来，按固定顺序拼成 caption

输出 jsonl：{path, tags:[...], caption:"..."}
caption 交给 CLIP 文本编码器 → (L,512) 序列 → 喂给 v2 的 cross-attention；
或取平均后喂 adaLN。

用法
----
  python auto_caption.py --data <图片目录> --out captions.jsonl --topk 8 --thr 0.24
  python auto_caption.py --data <目录> --out c.jsonl --report-only   # 只统计标签分布
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_score_eval import find_local_snapshot, load_clip, clip_feat

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# 标签按「组」组织：同组内互斥（取分最高的一个），组间独立保留。
# 这样 caption 更干净，不会出现 "long hair, short hair" 这种矛盾。
TAG_GROUPS = {
    'count':     ['solo, one person', 'two people', 'a group of people'],
    'shot':      ['a close-up portrait', 'an upper body shot', 'a full body shot',
                  'a wide shot with scenery'],
    'pose':      ['standing', 'sitting', 'lying down', 'kneeling',
                  'arms raised up', 'looking at the viewer', 'seen from the side',
                  'seen from above', 'seen from below', 'back view'],
    'hair_len':  ['short hair', 'long hair', 'very long hair', 'twintails'],
    'hair_col':  ['black hair', 'blonde hair', 'silver or white hair', 'red hair',
                  'blue hair', 'pink hair', 'brown hair', 'purple hair', 'green hair'],
    'outfit':    ['a dress', 'a swimsuit', 'a school uniform', 'armor',
                  'a kimono', 'a bunny suit', 'a maid outfit', 'a witch costume',
                  'a suit and tie', 'a hoodie and casual clothes'],
    'outfit_col':['a white outfit', 'a black outfit', 'a red outfit',
                  'a blue outfit', 'a pink outfit', 'a purple outfit'],
    'acc':       ['wearing glasses', 'wearing a hat', 'wearing cat ears',
                  'holding a prop or weapon'],
    'place':     ['indoors', 'outdoors', 'in a studio with plain background',
                  'at night', 'on a beach', 'in a forest', 'in a city street',
                  'on a stage'],
    'kind':      ['a photograph of a real person', 'an anime illustration',
                  'a screenshot of a website or app', 'a page of a comic or manga'],
    'quality':   ['a blurry low quality image', 'a sharp high quality image',
                  'a heavily watermarked image'],
}
TEMPLATES = ['a photo of {}', '{}', 'a picture of {}']


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


@torch.no_grad()
def build_text_bank(model, proc, device='cpu'):
    """返回 {group: (labels, emb[T,D])}，嵌入已归一化。"""
    bank = {}
    for grp, labels in TAG_GROUPS.items():
        embs = []
        for lab in labels:
            e = None
            for t in TEMPLATES:
                inp = proc(text=[t.format(lab)], return_tensors='pt', padding=True, truncation=True)
                f = clip_feat(model, model.get_text_features(**inp), 'text')
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                e = f if e is None else e + f
            embs.append(torch.nn.functional.normalize(e / len(TEMPLATES), dim=-1))
        bank[grp] = (labels, torch.cat(embs, 0))
    return bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--topk', type=int, default=3, help='threshold 模式下每组最多保留几个')
    ap.add_argument('--thr', type=float, default=0.24, help='threshold 模式的余弦相似度阈值')
    ap.add_argument('--mode', default='argmax', choices=['argmax', 'threshold'],
                    help='argmax = 每组取最匹配的一个（相对排序仍携带信息，推荐）；'
                         'threshold = 只用绝对分数过阈值的标签')
    ap.add_argument('--margin', type=float, default=0.012,
                    help='argmax 模式下，第二名与第一名差距小于此值时也一并保留')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = list_images(a.data)
    if a.limit:
        paths = paths[:a.limit]
    print('[caption] 图片 %d 张' % len(paths), flush=True)

    t0 = time.perf_counter()
    model, proc = load_clip('openai/clip-vit-base-patch32')
    bank = build_text_bank(model, proc)
    n_tag = sum(len(v[0]) for v in bank.values())
    print('[caption] CLIP 就绪，标签 %d 个（%d 组），耗时 %.1fs' % (
        n_tag, len(bank), time.perf_counter() - t0), flush=True)

    rows = []
    tagcount = {}
    t0 = time.perf_counter()
    for s in range(0, len(paths), a.batch):
        chunk = paths[s:s + a.batch]
        imgs, ok = [], []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert('RGB')); ok.append(True)
            except Exception:
                imgs.append(Image.new('RGB', (224, 224))); ok.append(False)
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors='pt')
            fe = clip_feat(model, model.get_image_features(**inp), 'image')
            fe = torch.nn.functional.normalize(fe.float(), dim=-1)
        for i, p in enumerate(chunk):
            if not ok[i]:
                rows.append(dict(path=p, tags=[], caption='', ok=False))
                continue
            chosen = []
            for grp, (labels, emb) in bank.items():
                sims = (fe[i:i + 1] @ emb.t())[0]
                order = torch.argsort(sims, descending=True)
                if a.mode == 'argmax':
                    # 组内相对排序：绝对分数在域外普遍偏低，但排名仍有信息量。
                    chosen.append(labels[order[0].item()])
                    tagcount[labels[order[0].item()]] = tagcount.get(labels[order[0].item()], 0) + 1
                    if a.topk > 1:
                        for j in order[1:min(a.topk, len(order))].tolist():
                            if sims[order[0]].item() - sims[j].item() <= a.margin:
                                chosen.append(labels[j])
                                tagcount[labels[j]] = tagcount.get(labels[j], 0) + 1
                else:
                    picked = 0
                    for j in order.tolist():
                        if sims[j].item() < a.thr or picked >= a.topk:
                            break
                        chosen.append(labels[j]); picked += 1
                        tagcount[labels[j]] = tagcount.get(labels[j], 0) + 1
            cap = ', '.join(chosen) if chosen else 'a person'
            rows.append(dict(path=p, tags=chosen, caption=cap, ok=True))
        if (s // a.batch) % 20 == 0:
            dt = time.perf_counter() - t0
            done = min(s + a.batch, len(paths))
            print('[caption] %d/%d  %.1fs  %.0f ms/张' % (done, len(paths), dt, dt / max(1, done) * 1000),
                  flush=True)

    dt = time.perf_counter() - t0
    okrows = [r for r in rows if r['ok']]
    print()
    print('[caption] 完成 %.1fs (%.0f ms/张)  成功 %d/%d' % (
        dt, dt / max(1, len(rows)) * 1000, len(okrows), len(rows)))
    if okrows:
        lens = [len(r['tags']) for r in okrows]
        print('[caption] 每张标签数: 中位 %d  平均 %.1f' % (
            sorted(lens)[len(lens) // 2], sum(lens) / len(lens)))
        print('[caption] 最常用标签 top20:')
        for k, v in sorted(tagcount.items(), key=lambda x: -x[1])[:20]:
            print('    %-46s %5d (%.1f%%)' % (k, v, v / len(okrows) * 100))
        print()
        print('[caption] 样例 caption:')
        for r in okrows[:5]:
            print('    %s' % r['caption'][:110])
    if a.out and not a.report_only:
        with open(a.out, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print('[caption] -> %s' % a.out)


if __name__ == '__main__':
    main()
