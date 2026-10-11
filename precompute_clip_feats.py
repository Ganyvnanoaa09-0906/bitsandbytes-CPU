# -*- coding: utf-8 -*-
"""precompute_clip_feats.py — 离线预计算 CLIP 视觉特征（REPA 的教师）

## 依据（Seedream 3.0 技术报告 2504.11346 §2.2.2 + REPA 官方代码 2410.06940）
- REPA 的损失 = 模型中间层特征与教师特征的【余弦距离】，权重 lambda = 0.5
- 教师原本是 DINOv2-L；REPA 官方代码列出 `clip-vit-L` 也是受支持的教师，
  超参相同（--proj-coeff=0.5 --encoder-depth=8 --projector-embed-dims=768）
- ⇒ **用我们本地已有的 CLIP 是方法学上被认可的**（我们拿不到 DINOv2：HF 不可达）
- `encoder-depth=8` 指教师的【第 8 个 block 输出】，不是最后一层

## 为什么需要离线预计算
教师只在训练时【前向】一次，且不需要梯度 ⇒ 特征可以预先算好存盘，
训练时反向只走我们自己的模型。这是 REPA"不依赖 GPU"的关键。
如果没有这一步，每个训练步都要额外跑一遍 CLIP，成本翻倍以上。

## 分辨率选择
CLIP-B/32 的 patch = 32。取 256x256 输入 ⇒ 8x8 = 64 个 patch（配合
interpolate_pos_encoding 把位置嵌入从 7x7 插到 8x8）。
64 -> 我们 16x16=256 的网格是干净的 2x 上采样。
（取 224 只有 7x7=49，上采样 3.65x，太粗。）

## 分片保存
每 SHARD 张存一个文件，最后再合并。否则跑到 90% 崩了就全丢。
"""
from __future__ import annotations
import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = r'D:\work\bitsandbytes-CPU'
os.environ.setdefault('HF_HOME', os.path.join(ROOT, '.hf_home'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# ⚠️ 不要用 from_pretrained('openai/clip-vit-base-patch32')：HF 不可达，
# 而 transformers 5.x 的离线缓存查找会因为 .no_exist 标记而报
# "couldn't find them in the cached files"（即使 config.json/pytorch_model.bin
# 都在快照目录里）。直接给【本地快照目录的绝对路径】绕开整个缓存机制。
_CACHE = os.path.join(ROOT, '.hf_home', 'hub',
                      'models--openai--clip-vit-base-patch32', 'snapshots')
SNAP = None
if os.path.isdir(_CACHE):
    _d = sorted(os.listdir(_CACHE))
    if _d:
        SNAP = os.path.join(_CACHE, _d[0])

TOKENS = os.path.join(ROOT, r'.autopilot\tokens_16x16_full.pt')
PATHS = TOKENS + '.paths.json'
OUTDIR = os.path.join(ROOT, r'.autopilot\clip_feats')
SIZE = 256
LAYER = 8            # REPA 的 encoder-depth=8
SHARD = 2000
BATCH = 16


def main():
    torch.set_num_threads(int(os.environ.get('THREADS', '6')))
    os.makedirs(OUTDIR, exist_ok=True)
    paths = json.load(open(PATHS, encoding='utf-8'))
    n = len(paths)
    print('[clip] 图片 %d 张  输出目录 %s' % (n, OUTDIR), flush=True)

    from transformers import CLIPVisionModel
    src = SNAP if SNAP else 'openai/clip-vit-base-patch32'
    print('[clip] 从 %s 载入' % src, flush=True)
    model = CLIPVisionModel.from_pretrained(src)
    model.eval()
    print('[clip] 模型载入 OK  隐藏维度 %d  层数 %d'
          % (model.config.hidden_size, model.config.num_hidden_layers), flush=True)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    t0 = time.time()
    done = 0
    for s0 in range(0, n, SHARD):
        s1 = min(n, s0 + SHARD)
        out_p = os.path.join(OUTDIR, 'shard_%06d_%06d.pt' % (s0, s1))
        if os.path.exists(out_p):
            done = s1
            print('[clip] 跳过已有分片 %s' % os.path.basename(out_p), flush=True)
            continue
        feats, ok = [], []
        for b0 in range(s0, s1, BATCH):
            b1 = min(s1, b0 + BATCH)
            xs = []
            for i in range(b0, b1):
                try:
                    with Image.open(paths[i]) as im:
                        im = im.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)
                        xs.append(torch.from_numpy(
                            np.asarray(im, dtype='uint8').copy()).float().permute(2, 0, 1) / 255.0)
                    ok.append(True)
                except Exception:
                    xs.append(torch.zeros(3, SIZE, SIZE))
                    ok.append(False)
            x = (torch.stack(xs) - mean) / std
            with torch.no_grad():
                o = model(pixel_values=x, interpolate_pos_encoding=True,
                          output_hidden_states=True)
            h = o.hidden_states[LAYER]            # (B, 1+pH*pW, d)
            patch = h[:, 1:]                      # 去掉 CLS
            feats.append(patch.half().cpu())
            done = b1
            if (done - s0) % (BATCH * 10) == 0:
                el = time.time() - t0
                rate = max(done - s0, 1) / max(el, 1e-9)
                eta = (n - done) / max(rate, 1e-9)
                print('[clip] %d/%d  %.1fs  %.1f img/s  ETA %.0f min'
                      % (done, n, el, rate, eta / 60), flush=True)
        f = torch.cat(feats, 0)
        torch.save({'feats': f, 'ok': torch.tensor(ok, dtype=torch.bool),
                    'layer': LAYER, 'size': SIZE, 'model': 'clip-vit-base-patch32'},
                   out_p)
        print('[clip] 分片完成 %s  %s' % (os.path.basename(out_p), tuple(f.shape)), flush=True)

    # 合并
    shards = sorted(glob.glob(os.path.join(OUTDIR, 'shard_*.pt')))
    print('[clip] 合并 %d 个分片' % len(shards), flush=True)
    fs, oks, layers = [], [], set()
    for p in shards:
        d = torch.load(p, map_location='cpu')
        fs.append(d['feats']); oks.append(d['ok']); layers.add(d['layer'])
    F_all = torch.cat(fs, 0)
    OK = torch.cat(oks, 0)
    assert F_all.shape[0] == n, '特征数 %d != 图片数 %d' % (F_all.shape[0], n)
    meta = dict(layer=sorted(layers)[0], size=SIZE, n=n,
                n_patch=int(F_all.shape[1]), dim=int(F_all.shape[2]),
                model='clip-vit-base-patch32',
                ok_ratio=float(OK.float().mean()))
    torch.save(F_all, os.path.join(OUTDIR, 'clip_feats.pt'))
    json.dump(meta, open(os.path.join(OUTDIR, 'clip_feats.meta.json'), 'w'), indent=1)
    print('[clip] 完成 %s  meta=%s' % (tuple(F_all.shape), meta), flush=True)
    print('[clip] 有效图比例 %.4f  总耗时 %.1f min' % (meta['ok_ratio'], (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
