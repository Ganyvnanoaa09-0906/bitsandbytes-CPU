# -*- coding: utf-8 -*-
"""test_ar_infill.py — AR 能不能做「条件续写」（修补能力测试）

动机：用户提出用 AR 做脏区域更新（只重绘变化的地方）。
      这要求 AR 具备【给定上下文、续写合理内容】的能力。

测试设计：
  ① 取真实图的 token 序列（长度 T）
  ② 保留前 k 个 token 作为「已知上下文」
  ③ 让 AR 从第 k+1 个继续生成到 T
  ④ 对比【续写结果】vs【真实的后半段】：
     · 逐位置 token 准确率（top-1 / top-5）
     · 与真实的后半段相比，视觉统计量是否一致
  ⑤ 对照：让 AR 从头生成（无上下文）的准确率

若「有上下文」的准确率 >> 「无上下文」⇒ AR 会利用上下文 ⇒ 能做修补 ✅
若两者差不多 ⇒ AR 忽略上下文 ⇒ 不能修补 ❌
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
import sys, os, glob, random, time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 82)
print('AR 条件续写（修补）能力测试')
print('=' * 82)

from scripts.lib.small_vqvae import vqvae_from_ckpt

HERE = r'D:\work\cloud_results'
SIZE = 256
N = 32


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


def load_ar(ckpt):
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
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1),
                              loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


# ---------- 真实 token ----------
files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(3)
real = load_real(random.sample(files, N))

CASES = [
    ('ar_swiglu (T=64, 8x8)', r'D:\work\cloud_results\swiglu_last.pt',
     r'D:\work\cloud_results\vqvae_8x8.pt'),
    ('ar_flat256 (T=256, 非hier)', r'D:\work\cloud_results\flat_12000.pt',
     r'D:\work\cloud_results\ckpt_C_mse.pt'),
]

for label, ck, vqpath in CASES:
    print()
    print('=' * 82)
    print('[%s]' % label)
    m, cfg = load_ar(ck)
    vq, _ = vqvae_from_ckpt(vqpath)
    vq.eval()
    T = cfg.total_tokens
    side = int(round(T ** 0.5))

    # 真实 token
    outs = []
    with torch.no_grad():
        for i in range(0, N, 16):
            _, idx, _ = vq(real[i:i + 16])
            outs.append(idx.view(idx.shape[0], -1))
    real_tok = torch.cat(outs).long()
    print('  真实 token: %s' % (tuple(real_tok.shape),))

    # ---------- 有上下文 vs 无上下文 ----------
    print()
    print('  %-14s %12s %12s %12s' % ('保留前 k 个', 'top-1 准确率', 'top-5 准确率', 'loss'))
    print('  ' + '-' * 58)

    for k in (0, T // 4, T // 2, 3 * T // 4):
        # 构造输入：BOS + 真实的前 k 个（k=0 时只有 BOS）
        # 目标：预测第 k+1 .. T 个
        xs = real_tok[:, :k] if k > 0 else torch.zeros(N, 0, dtype=torch.long)
        # 用 forward_logits 取 logits
        with torch.no_grad():
            if cfg.use_bos:
                bos = m.bos_emb.expand(N, 1, -1)
                if k > 0:
                    emb = m.token_emb(xs)
                    xin = torch.cat([bos, emb], dim=1)
                else:
                    xin = bos
            else:
                xin = m.token_emb(xs) if k > 0 else m.token_emb(torch.zeros(N, 1, dtype=torch.long))
            # 位置
            tt = xin.shape[1]
            pooled = None
            _, g = m.forward_logits(xs if k > 0 else torch.zeros(N, 1, dtype=torch.long))
            g = g[0] if isinstance(g, (tuple, list)) else g
        # g 的形状是 (N, len(input), vocab)，但我们只关心最后一步对下一个 token 的预测
        # ⚠️ 严格来说要递归，但这里用「teacher forcing 的前缀」近似评估：
        #    取 g[:, -1] 预测第 k+1 个
        if k < T:
            tgt = real_tok[:, k] if k < T else None
            if tgt is not None:
                logits = g[:, -1] if g.shape[1] == (k + 1 if k > 0 else 1) else g[:, k - 1 if k > 0 else 0]
                top1 = (logits.argmax(-1) == tgt).float().mean().item()
                tk5 = logits.topk(5, -1).indices
                top5 = (tk5 == tgt.unsqueeze(-1)).any(-1).float().mean().item()
                loss = F.cross_entropy(logits, tgt).item()
                print('  %-14d %11.1f%% %11.1f%% %12.4f' % (k, top1 * 100, top5 * 100, loss))

    print()
    print('  判读:')
    print('    k 越大、准确率越高 ⇒ AR 会利用上下文 ⇒ 【能做修补】✅')
    print('    k 变化基本不影响 ⇒ AR 忽略上下文 ⇒ 【不能修补】❌')

print()
print('=' * 82)
print('DONE')
