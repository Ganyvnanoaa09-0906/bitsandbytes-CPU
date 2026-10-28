# -*- coding: utf-8 -*-
"""prompt_cond.py — 把任意提示词编码成模型能吃的条件张量。

背景：`cond_clip_29113.pt` 是数据集的字幕用 **openai/clip-vit-base-patch32**
编码出来的 (29113, 512) 单位范数嵌入。训练好的条件模型只认这个空间，
所以要出任意提示词的图，必须用【同一个编码器 + 同一套后处理】。

⚠️ 这里有两个已经踩过的坑，必须原样复刻 `train_ar_v2.load_caption_cond`：
  1. transformers 5.x 的 `get_text_features()` **不返回张量**，而是
     `BaseModelOutputWithPooling`。直接 `.float()` 抛 AttributeError，
     而旧代码把异常 try/except 吞掉返回 None ⇒ **条件训练静默退化成无条件**。
     必须走 `clip_feat()` 解包。
  2. **不要再过一遍投影层**。实测 pooler_output 已经是投影后的嵌入：
        直接 pooler_output            余弦 0.2888  ✅
        text_projection(pooler_output) 余弦 0.0434  ❌ 重复投影
  另外必须 `F.normalize(..., dim=-1)`，与训练时一致。

用法:
  python prompt_cond.py --prompt "long hair, twintails, maid outfit"
  python prompt_cond.py --prompt "..." --out my_cond.pt      # 存成可喂给 --cond-pt
  python prompt_cond.py --verify                             # 与原 cond 对照（关键验证）
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
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MODEL_ID = 'openai/clip-vit-base-patch32'
COND = os.path.join(ROOT, '.autopilot', 'cond_clip_29113.pt')
PATHS = os.path.join(ROOT, '.autopilot', 'tokens_16x16_full.pt.paths.json')
CAPS = os.path.join(ROOT, '.autopilot', 'captions', 'captions_merged.jsonl')


def encode(texts, batch=64):
    """文本 -> (N, 512) 单位范数嵌入。完全复刻训练时的口径。"""
    from clip_score_eval import load_clip, clip_feat
    m, p = load_clip(MODEL_ID)
    m.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            inp = p(text=texts[s:s + batch], return_tensors='pt',
                    padding=True, truncation=True)
            f = clip_feat(m, m.get_text_features(**inp), 'text')
            feats.append(F.normalize(f.float(), dim=-1))
    return torch.cat(feats, 0)


def verify(n=8):
    """关键验证：拿原字幕重新编码，和 cond_clip_29113.pt 对应行比。
    余弦 ≈ 1.0 才说明编码器/后处理完全一致，任意提示词才落在同一空间。
    """
    print("=" * 74)
    print("验证：原字幕重新编码 vs cond_clip_29113.pt")
    print("=" * 74)
    C = torch.load(COND, map_location='cpu', weights_only=False).float()
    paths = json.load(open(PATHS, encoding='utf-8'))
    caps = [json.loads(l) for l in open(CAPS, encoding='utf-8') if l.strip()]
    by = {os.path.abspath(c['path']): c.get('caption', '') for c in caps}

    # 取若干行，含开头/中间/结尾，避免只看一段
    idx = [0, 1, 2, 3] + [len(paths) // 2 + i for i in range(2)] + [len(paths) - 2, len(paths) - 1]
    idx = [i for i in idx if 0 <= i < len(paths)][:n]
    texts = [by.get(os.path.abspath(paths[i]), 'a person') for i in idx]
    print("  取 %d 行：%s" % (len(idx), idx))
    print()
    got = encode(texts)
    print("  %-8s %-12s %-12s %s" % ("行号", "与原 cond 余弦", "范数", "字幕前 40 字"))
    print("  " + "-" * 78)
    worst = 1.0
    for j, i in enumerate(idx):
        cos = float(F.cosine_similarity(got[j:j + 1], C[i:i + 1]).item())
        worst = min(worst, cos)
        print("  %-8d %-12.4f %-12.4f %s" % (i, cos, float(got[j].norm()), texts[j][:40]))
    print()
    print("  最差余弦 = %.4f" % worst)
    if worst > 0.999:
        print("  ✅ 完全一致 —— 任意提示词可以用")
    elif worst > 0.99:
        print("  ✅ 一致（数值误差级）")
    else:
        print("  ❌ 不一致！说明编码口径不同，任意提示词会落在错误的子空间")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default=None, nargs='+', help='一个或多个提示词')
    ap.add_argument('--out', default=None, help='存成 .pt（可喂给 sample 的 --cond-pt）')
    ap.add_argument('--verify', action='store_true', help='与原 cond 对照验证')
    a = ap.parse_args()

    if a.verify and a.prompt is None:
        verify()
        return
    if a.prompt is None:
        ap.error('要么给 --prompt，要么给 --verify')

    texts = a.prompt if len(a.prompt) > 1 else [a.prompt[0]]
    e = encode(texts)
    print("[prompt] %d 条提示词 -> %s  范数 mean=%.4f" % (len(texts), tuple(e.shape), float(e.norm(dim=1).mean())))
    for i, t in enumerate(texts):
        print("    [%d] %s" % (i, t[:90]))
    if len(texts) > 1:
        sim = e @ e.T
        off = sim[~torch.eye(len(texts), dtype=bool)]
        print("[prompt] 提示词两两余弦 mean=%.3f min=%.3f max=%.3f"
              % (float(off.mean()), float(off.min()), float(off.max())))
    out = a.out or os.path.join(ROOT, '.autopilot', 'cond_prompt.pt')
    torch.save(e.half(), out)
    print("[prompt] -> %s" % out)


if __name__ == '__main__':
    main()
