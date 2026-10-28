# -*- coding: utf-8 -*-
"""diagnose_quality2.py — 定位出图精度瓶颈：分词器 还是 AR

三方对比：
  A) 原图
  B) 分词器重建（给正确的 token）—— 这是分词器的上限
  C) AR 生成（token 由模型生成）

如果 B 也糊 ⇒ 瓶颈在分词器；如果 B 清晰而 C 糊 ⇒ 瓶颈在 AR。
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
import sys, glob, os, random, time
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 80)
print('出图精度瓶颈定位')
print('=' * 80)

from scripts.lib.small_vqvae import vqvae_from_ckpt

N = 8
SIZE = 256


def load_real(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
        a = np.asarray(im, dtype='uint8').copy()
        out.append(torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(out)


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def save_grid(tensors, path, cols=2):
    n = len(tensors)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 256, rows * 256), (20, 20, 20))
    for i, t in enumerate(tensors):
        t = t.clamp(-1, 1)
        x = ((t + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        canvas.paste(Image.fromarray(x), ((i % cols) * 256, (i // cols) * 256))
    canvas.save(path)
    return path


def decode_from_idx(vq, idx, side):
    """与 sample_ar.py 完全一致的解码路径"""
    with torch.no_grad():
        emb = vq.vq.codebook(idx)                  # (B,side,side,C)
        z = emb.permute(0, 3, 1, 2).contiguous()   # (B,C,side,side)
        img = vq.dec(z)                            # (B,3,256,256)
    return torch.tanh(img)


# ---------- 数据 ----------
files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(2024)
paths = random.sample(files, N)
real = load_real(paths)
save_grid(real, r'D:\work\cloud_results\DQ_A_real.png')
print('  A 原图        -> DQ_A_real.png  (%d 张)' % N)

results = {}

# ---------- 分词器重建 ----------
for label, ck, side, out_name in (
    ('8x8  (ar_swiglu 用)', r'D:\work\cloud_results\vqvae_8x8.pt', 8, 'DQ_B_rec8x8.png'),
    ('16x16 (hier 用)', r'D:\work\cloud_results\ckpt_C_mse.pt', 16, 'DQ_C_rec16x16.png'),
):
    print()
    print('[%s]' % label)
    vq, meta = vqvae_from_ckpt(ck)
    vq.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        rec, idx, _ = vq(real)
    dt = time.perf_counter() - t0
    save_grid(rec, os.path.join(r'D:\work\cloud_results', out_name))
    p = psnr(real, rec)
    results[label] = (p, dt / N, meta.get('ch'))
    print('  重建 PSNR 【%.2f dB】   %.1fs (%.3f s/张)' % (p, dt, dt / N))
    print('  ch=%s  ->  %s' % (meta.get('ch'), out_name))

# ---------- AR 生成 ----------
print()
print('[AR 生成]')


def sample_tokens(ckpt, L=0):
    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight'):
                mx = v.shape[0]
                break
        use = L if L > 0 else max(1, mx - 1)
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=use, loop_L_max=max(1, mx - 1), loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        toks = m.generate(None, steps=cfg.total_tokens, top_k=1024, temperature=1.0,
                          batch=N, seed=2024)
    return toks, cfg, time.perf_counter() - t0


for label, ck, ckname, side, out_name in (
    ('ar_swiglu T=64', r'D:\work\cloud_results\swiglu_last.pt',
     r'D:\work\cloud_results\vqvae_8x8.pt', 8, 'DQ_D_gen8x8.png'),
    ('loop1 T=256', r'D:\work\cloud_results\loop1_26500.pt',
     r'D:\work\cloud_results\ckpt_C_mse.pt', 16, 'DQ_E_gen16x16.png'),
):
    toks, cfg, dt = sample_tokens(ck)
    vq, _ = vqvae_from_ckpt(ckname)
    vq.eval()
    idx = toks[:, :side * side].view(-1, side, side)
    img = decode_from_idx(vq, idx, side)
    save_grid(img, os.path.join(r'D:\work\cloud_results', out_name))
    print('  %-16s 生成 %.1fs (%.3f s/张)  -> %s' % (label, dt, dt / N, out_name))

print()
print('=' * 80)
print('[结论]')
for k, (p, s, ch) in results.items():
    print('  %-22s 重建 PSNR %5.2f dB   %.3f s/张' % (k, p, s))
print()
print('  判读:')
print('    重建 PSNR 低 ⇒ 分词器容量不足（瓶颈在分词器）')
print('    重建好但生成糊 ⇒ 瓶颈在 AR')
print('  看图: DQ_A_real.png(原图) / DQ_B_rec8x8.png / DQ_C_rec16x16.png')
print('        DQ_D_gen8x8.png(生成) / DQ_E_gen16x16.png(生成)')
print('DONE')
