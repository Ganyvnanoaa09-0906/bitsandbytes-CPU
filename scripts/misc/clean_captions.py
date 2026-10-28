# -*- coding: utf-8 -*-
"""clean_captions.py — 清理字幕里的 QC 噪声与自相矛盾标签。

诊断依据（2026-09-23 实测）：
  - 66 个标签、每条中位 20 个
  - `a heavily watermarked image` 92.8%、`a blurry low quality image` 57.2%
  - `an anime illustration` 90.0% 而图其实是真人 cosplay 照片
  - `two people`(87.1%) 与 `solo`(78.8%) 在 66.4% 的条目里同时出现
  - 分组矛盾率：人数 78.8%、镜头 61.1%、姿势 58.8%
  - 字幕信号 Spearman = +0.0543（弱但非零）

清理策略：
  1) 丢掉 QC/元标签（watermarked / blurry / sharp / screenshot / comic page）
  2) 丢掉互斥的域标签（anime illustration 与 photograph of a real person 都丢）
  3) 丢掉完全重复的标签（solo 与 one person 只留一个）
  4) 互斥组内只保留一个（按固定优先级），避免"同时 sitting 和 lying down"
  5) 保留可共现的内容标签（发色/发长之外的服饰、道具、配饰、视角）

★ 输出到【新文件】，不覆盖原数据。

用法:
  python clean_captions.py                 # 清理并打印前后统计
  python clean_captions.py --encode        # 顺便重新编码成新 cond 张量
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
PATHS = os.path.join(ROOT, '.autopilot', 'tokens_16x16_full.pt.paths.json')
CAPS = os.path.join(ROOT, '.autopilot', 'captions', 'captions_merged.jsonl')
OUT_JSONL = os.path.join(ROOT, '.autopilot', 'captions', 'captions_clean.jsonl')
OUT_COND = os.path.join(ROOT, '.autopilot', 'cond_clip_29113_clean.pt')

# 1) 直接丢掉的 QC / 元标签
DROP = {
    'a heavily watermarked image',
    'a blurry low quality image',
    'a sharp high quality image',
    'a screenshot of a website or app',
    'a page of a comic or manga',
    # 2) 互斥的域标签，两个都丢（都接近常数，且其中一个是错的）
    'an anime illustration',
    'a photograph of a real person',
    # 3) 与 'solo' 完全重复
    'one person',
}

# 4) 互斥组：组内只保留【第一个命中】的标签（按列表顺序为优先级）
EXCLUSIVE = [
    # 人数：★ 优先 solo。理由：直接看图为单人 cosplay 照片，而 `two people` 占 87.1%
    #     与 `solo` 78.8% 在 66.4% 的条目里同时出现 ⇒ 前者多半是标注器猜错的。
    #     而且 solo(79/21) 比 two(87/13) 分布更均衡、信息量更大。
    ['a group of people', 'solo', 'two people'],
    # 镜头：优先更具体的
    ['a close-up portrait', 'an upper body shot', 'a full body shot', 'a wide shot with scenery'],
    # 姿势：互斥
    ['lying down', 'sitting', 'kneeling', 'standing'],
    # 发长：very long 是 long 的子集，short 与 long 互斥
    ['short hair', 'very long hair', 'long hair'],
    # 发色：一个人一种主色
    ['black hair', 'silver or white hair', 'blonde hair', 'brown hair',
     'blue hair', 'pink hair', 'purple hair', 'red hair', 'green hair'],
    # 服装主色
    ['a white outfit', 'a black outfit', 'a blue outfit', 'a pink outfit',
     'a red outfit', 'a purple outfit'],
    # 视角
    ['seen from above', 'seen from below', 'seen from the side', 'back view'],
    # 室内外
    ['outdoors', 'indoors'],
    # 具体场地
    ['in a studio with plain background', 'on a stage', 'in a forest',
     'in a city street', 'on a beach'],
]


def clean_one(text):
    tags = [t.strip().lower() for t in re.split(r'[,\n]', text) if t.strip()]
    seen = set()
    kept = []
    for t in tags:
        if t in DROP or t in seen:
            continue
        seen.add(t)
        kept.append(t)
    # 互斥组：每组只留第一个命中的
    for grp in EXCLUSIVE:
        present = [g for g in grp if g in seen]
        if len(present) > 1:
            for g in present[1:]:
                kept.remove(g)
                seen.discard(g)
    return ', '.join(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encode', action='store_true', help='顺便编码成新的 cond 张量')
    a = ap.parse_args()

    paths = json.load(open(PATHS, encoding='utf-8'))
    caps = [json.loads(l) for l in open(CAPS, encoding='utf-8') if l.strip()]
    by = {os.path.abspath(c['path']): c for c in caps}

    rows = []
    orig_texts, new_texts = [], []
    for p in paths:
        c = by.get(os.path.abspath(p), {})
        o = c.get('caption', '')
        n = clean_one(o)
        orig_texts.append(o)
        new_texts.append(n)
        rows.append({'path': p, 'caption': n, 'caption_raw': o})

    def stats(texts, name):
        cnt = Counter()
        for t in texts:
            for tag in re.split(r'[,\n]', t):
                tag = tag.strip().lower()
                if tag:
                    cnt[tag] += 1
        lens = [len([x for x in re.split(r'[,\n]', t) if x.strip()]) for t in texts]
        print("  【%s】不同标签 %d 个；每条标签数 中位 %.0f 均值 %.1f"
              % (name, len(cnt), sorted(lens)[len(lens) // 2], sum(lens) / len(lens)))
        return cnt

    print("=" * 72)
    print("清理前后对比")
    print("=" * 72)
    c0 = stats(orig_texts, '清理前')
    c1 = stats(new_texts, '清理后')

    print()
    print("  被删掉的标签（按删掉的数量）:")
    for tag, n in c0.most_common():
        d = n - c1.get(tag, 0)
        if d > 0:
            print("    %-46s -%6d  (剩 %d)" % (tag[:46], d, c1.get(tag, 0)))
    print()
    print("  新增出现的标签（不该有）:", [t for t in c1 if t not in c0] or "无")

    # 矛盾率复检
    def conflicts(texts):
        n = len(texts)
        out = {}
        for gname, grp in (('人数', EXCLUSIVE[0]), ('镜头', EXCLUSIVE[1]),
                           ('姿势', EXCLUSIVE[2]), ('发色', EXCLUSIVE[4])):
            m = sum(1 for t in texts if sum(1 for g in grp if g in t) > 1)
            out[gname] = 100 * m / n
        return out

    cf0, cf1 = conflicts(orig_texts), conflicts(new_texts)
    print()
    print("  互斥组同时出现率:")
    for k in cf0:
        print("    %-6s %.1f%%  ->  %.1f%%" % (k, cf0[k], cf1[k]))

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print()
    print("  -> %s" % OUT_JSONL)

    print()
    print("  清理后样例:")
    for i in (0, 2, 5):
        print("    原: %s" % orig_texts[i][:88])
        print("    新: %s" % new_texts[i][:88])
        print()

    if a.encode:
        print("=" * 72)
        print("重新编码（用与原 cond 完全相同的口径）")
        print("=" * 72)
        import torch
        sys.path.insert(0, ROOT)
        from scripts.lib.prompt_cond import encode
        e = encode(new_texts)
        torch.save(e.half(), OUT_COND)
        print("  -> %s  形状=%s  范数 mean=%.4f"
              % (OUT_COND, tuple(e.shape), float(e.norm(dim=1).mean())))
        old = torch.load(os.path.join(ROOT, '.autopilot', 'cond_clip_29113.pt'),
                         map_location='cpu', weights_only=False).float()
        idx = torch.randperm(e.shape[0])[:400]
        so = (old[idx] @ old[idx].T)
        sn = (e[idx] @ e[idx].T)
        m = ~torch.eye(len(idx), dtype=bool)
        print("  两两余弦  原: mean=%.3f std=%.3f   清理后: mean=%.3f std=%.3f"
              % (float(so[m].mean()), float(so[m].std()),
                 float(sn[m].mean()), float(sn[m].std())))
        print("  （std 变大 => 条件更有区分度）")


if __name__ == '__main__':
    main()
