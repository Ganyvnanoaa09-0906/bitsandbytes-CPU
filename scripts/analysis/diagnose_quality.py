# -*- coding: utf-8 -*-
"""diagnose_quality.py — 定位出图精度的瓶颈：分词器 还是 AR

对比三方：
  A) 原图（256×256）
  B) 分词器重建（encode → decode，给定正确 token 时的上限）
  C) AR 生成（token 由模型生成）

如果 B 也糊 ⇒ 瓶颈在分词器；如果 B 清晰而 C 糊 ⇒ 瓶颈在 AR。
同时输出两两 PSNR。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, glob, os, random, time
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 80)
print('出图精度瓶颈定位')
print('=' * 80)

from small_vqvae import vqvae_from_ckpt
import fused_cpu

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
    """a,b in [-1,1]"""
    mse = (a - b).pow(2).mean().item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


def grid(tensors, path, cols=2):
    """存成网格图（每张放大到 256）"""
    n = len(tensors)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * 256, rows * 256))
    for i, t in enumerate(tensors):
        x = ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(x).resize((256, 256), Image.NEAREST)
        canvas.paste(im, ((i % cols) * 256, (i // cols) * 256))
    canvas.save(path)
    return path


# 数据
files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
print('  源图 %d 张' % len(files))
random.seed(2024)
paths = random.sample(files, N)
real = load_real(paths)
grid(real, r'D:\work\cloud_results\DQ_A_real.png')
print('  A 原图   -> DQ_A_real.png')

# ---------- 8x8 分词器 ----------
print()
print('[1] 8×8 分词器（ar_swiglu 用的）')
vq8, _ = vqvae_from_ckpt(r'D:\work\cloud_results\vqvae_8x8.pt')
vq8.eval()
t0 = time.perf_counter()
with torch.no_grad():
    rec8, idx8, _ = vq8(real)
t_rec8 = time.perf_counter() - t0
grid(rec8, r'D:\work\cloud_results\DQ_B_rec8x8.png')
p8 = psnr(real, rec8)
print('  重建 PSNR %.2f dB   耗时 %.1fs (%.3f s/张)' % (p8, t_rec8, t_rec8 / N))

# ---------- 16x16 分词器 ----------
print()
print('[2] 16×16 分词器（hier 用的）')
vq16, _ = vqvae_from_ckpt(r'D:\work\cloud_results\ckpt_C_mse.pt')
vq16.eval()
t0 = time.perf_counter()
with torch.no_grad():
    rec16, idx16, _ = vq16(real)
t_rec16 = time.perf_counter() - t0
grid(rec16, r'D:\work\cloud_results\DQ_C_rec16x16.png')
p16 = psnr(real, rec16)
print('  重建 PSNR %.2f dB   耗时 %.1fs (%.3f s/张)' % (p16, t_rec16, t_rec16 / N))

# ---------- 生成图（用 AR 的 token）----------
print()
print('[3] AR 生成（token 由模型生成）后解码')


def ar_tokens(ckpt, grid_n):
    """用 sample_ar 的逻辑生成 token"""
    from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight'):
                mx = v.shape[0]; break
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1), loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        tok = m.generate(None, steps=cfg.total_tokens, top_k=1024, temperature=1.0,
                         batch=N, seed=2024)
    return tok


for label, ck, vq, name in (
    ('ar_swiglu (T=64)', r'D:\work\cloud_results\swiglu_last.pt', vq8, 'DQ_D_gen8x8.png'),
    ('loop1 (T=256)', r'D:\work\cloud_results\loop1_26500.pt', vq16, 'DQ_E_gen16x16.png'),
):
    print('  --- %s ---' % label)
    t0 = time.perf_counter()
    tok = ar_tokens(ck, 0)
    t_gen = time.perf_counter() - t0
    with torch.no_grad():
        out = vq.decode_from_indices(tok) if hasattr(vq, 'decode_from_indices') else None
        if out is None:
            # 退回：用 vq 的 quantize + decode
            z = vq.vq.codebook[tok.reshape(-1)].reshape(
                tok.shape[0], -1, *vq.vq.codebook.shape[1:])
            try:
                out = vq.dec(vq.vq.project_out(vq.vq.pre(z))) if hasattr(vq.vq, 'project_out') else vq.dec(z)
            except Exception:
                out = vq.dec(z)
    out = torch.tanh(out) if out.abs().max() > 1.5 else out
    grid(out, os.path.join(r'D:\work\cloud_results', name))
    print('    生成 %d token/图，用时 %.1fs (%.3f s/张) -> %s'
          % (tok.shape[1], t_gen, t_gen / N, name))

print()
print('=' * 80)
print('判读:')
print('  8x8  重建 PSNR %.2f dB' % p8)
print('  16x16 重建 PSNR %.2f dB' % p16)
print('  ⇒ 若 8x8 重建就明显差 ⇒ 瓶颈在 8x8 分词器（提升点：更好的分词器）')
print('  ⇒ 若重建好但生成差 ⇒ 瓶颈在 AR')
print('DONE')
