# -*- coding: utf-8 -*-
"""precompute_clip_feats2.py — 按【显式路径清单】预计算 CLIP 特征（REPA 教师）

与 precompute_clip_feats.py 的区别：那个绑定 `tokens_16x16_full.pt.paths.json`；
这个接受任意 paths.json，用于在 29k 全量之外做小规模 REPA A/B（子集上数据更少，
REPA 的效果应该更明显 ⇒ 结论有指示性）。

⚠️ 必须保证【特征行序 == token 行序】。所以调用方要先写出配对的 paths.json。

CLIP 载入的坑（已踩过）：不能用 from_pretrained('openai/clip-vit-base-patch32') ——
HF 不可达时 transformers 5.x 会因 .no_exist 标记报"找不到缓存文件"，
即使 config.json / pytorch_model.bin 都在快照目录里。必须给本地快照绝对路径。
"""
from __future__ import annotations
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
import os, sys, json, time, argparse
import numpy as np
import torch
from PIL import Image

ROOT = r'D:\work\bitsandbytes-CPU'
os.environ.setdefault('HF_HOME', os.path.join(ROOT, '.hf_home'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
_CACHE = os.path.join(ROOT, '.hf_home', 'hub',
                      'models--openai--clip-vit-base-patch32', 'snapshots')
SNAP = None
if os.path.isdir(_CACHE):
    _d = sorted(os.listdir(_CACHE))
    if _d:
        SNAP = os.path.join(_CACHE, _d[0])

SIZE = 256
LAYER = 8          # REPA 的 encoder-depth=8（第 8 个 block 的输出，不是最后一层）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', required=True, help='图片路径清单 json')
    ap.add_argument('--out', required=True, help='输出特征 .pt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = json.load(open(a.paths, encoding='utf-8'))
    n = len(paths)
    print('[clip] 图片 %d 张  ->  %s' % (n, a.out), flush=True)
    from transformers import CLIPVisionModel
    src = SNAP if SNAP else 'openai/clip-vit-base-patch32'
    print('[clip] 从 %s 载入' % src, flush=True)
    model = CLIPVisionModel.from_pretrained(src).eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    feats, ok = [], []
    t0 = time.time()
    for s in range(0, n, a.batch):
        e = min(n, s + a.batch)
        xs = []
        for i in range(s, e):
            try:
                with Image.open(paths[i]) as im:
                    im = im.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)
                    xs.append(torch.from_numpy(
                        np.asarray(im, dtype='uint8').copy()).float().permute(2, 0, 1) / 255.)
                ok.append(True)
            except Exception:
                xs.append(torch.zeros(3, SIZE, SIZE))
                ok.append(False)
        x = (torch.stack(xs) - mean) / std
        with torch.no_grad():
            o = model(pixel_values=x, interpolate_pos_encoding=True,
                      output_hidden_states=True)
        feats.append(o.hidden_states[LAYER][:, 1:].half().cpu())   # 去 CLS
        if (s // a.batch) % 20 == 0:
            el = time.time() - t0
            rate = max(e, 1) / max(el, 1e-9)
            print('[clip] %d/%d  %.0fs  %.1f img/s  ETA %.0f min'
                  % (e, n, el, rate, (n - e) / max(rate, 1e-9) / 60), flush=True)
    F = torch.cat(feats, 0)
    torch.save(F, a.out)
    meta = dict(n=n, shape=list(F.shape), layer=LAYER, size=SIZE,
                model='clip-vit-base-patch32', n_patch=int(F.shape[1]),
                dim=int(F.shape[2]), ok_ratio=float(np.mean(ok)),
                paths=a.paths)
    json.dump(meta, open(a.out + '.meta.json', 'w'), indent=1)
    print('[clip] 完成 %s  有效 %.4f  耗时 %.1f min'
          % (tuple(F.shape), meta['ok_ratio'], (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
