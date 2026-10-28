# -*- coding: utf-8 -*-
"""make_cond_pt.py — 把 caption 编码成 (N, text_dim) 条件张量存盘（供云端 --cond-pt 使用）。

为什么要预计算：
  条件来自 CLIP 文本编码器；云端 HF 不可达、也没有 CLIP 权重。
  预计算后只需传 ~30MB(fp16) 的张量，而不是 605MB 的模型快照，
  而且训练时完全不依赖 CLIP，云端更稳。
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
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib import train_ar_v2

PATHS = r'.autopilot\tokens_16x16_full.pt.paths.json'
CAPS = r'.autopilot\captions\captions_merged.jsonl'
OUT = r'.autopilot\cond_clip_29113.pt'


def main():
    paths = json.load(open(PATHS, encoding='utf-8'))
    c = T.load_caption_cond(CAPS, paths, 512)
    if c is None:
        raise SystemExit('编码失败')
    print('[cond] shape=%s  dtype=%s' % (tuple(c.shape), c.dtype))
    print('[cond] 行范数 mean=%.4f min=%.4f max=%.4f'
          % (float(c.norm(dim=1).mean()), float(c.norm(dim=1).min()), float(c.norm(dim=1).max())))
    sim = c[:300] @ c[:300].T
    off = sim[~torch.eye(300, dtype=bool)]
    print('[cond] 两两余弦 mean=%.3f min=%.3f max=%.3f （接近 1.000 就说明等于无条件）'
          % (float(off.mean()), float(off.min()), float(off.max())))
    torch.save(c.half(), OUT)
    print('[cond] -> %s  %.1f MB' % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
