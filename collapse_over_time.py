# -*- coding: utf-8 -*-
"""collapse_over_time.py — T=256 的 token 坍塌是训练不足还是结构性？

方法：取不同训练步的 checkpoint，算生成 token 的相邻同率。
  · 若随步数下降 ⇒ 训练不足，继续训就行
  · 若一直 ~30%   ⇒ 结构性，必须改架构/范式

已知：
  真实 16x16 相邻同率 4.47%(H)
  loop1 @26500 步: 31.83%(H)  ← 坍塌

对比对象：hier 的多个 checkpoint（60000 步训完，有 97 个 ckpt）
"""
import sys, os, time
import torch
import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 80)
print('T=256 token 坍塌的时间演化')
print('=' * 80)

from small_vqvae import vqvae_from_ckpt

HERE = r'D:\work\cloud_results'
SIDE = 16
N = 64          # 每次生成 64 张
REAL_H = 0.0447  # 实测真实图的相邻同率


def load_ar(ckpt):
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
                mx = v.shape[0]
                break
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1),
                              loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    return m, cfg, ck.get('step', -1)


def stats(toks, side):
    B = toks.shape[0]
    g = toks.view(B, side, side)
    h = (g[:, :, :-1] == g[:, :, 1:]).float().mean().item()
    v = (g[:, :-1, :] == g[:, 1:, :]).float().mean().item()
    flat = toks.reshape(-1)
    perm = flat[torch.randperm(flat.numel())]
    rnd = (flat == perm).float().mean().item()
    return h, v, rnd


CKPTS = []
for name in ('hier_ckpt_10000.pt', 'hier_ckpt_20000.pt', 'hier_ckpt_30000.pt',
             'hier_ckpt_40000.pt', 'hier_ckpt_50000.pt', 'hier_last.pt',
             'loop1_26500.pt'):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        CKPTS.append((name, p))

if not CKPTS:
    print('  本地没有多步数的 checkpoint，需要先下载')
    print('  期望文件: hier_ckpt_{10000,20000,30000,40000,50000}.pt / hier_last.pt')
    sys.exit(0)

print()
print('  %-22s %8s %12s %12s %10s' % ('checkpoint', 'step', '相邻同H', '相邻同V', '相对真实'))
print('  ' + '-' * 68)
for name, p in CKPTS:
    m, cfg, step = load_ar(p)
    with torch.no_grad():
        toks = m.generate(None, steps=cfg.total_tokens, top_k=1024, temperature=1.0,
                          batch=N, seed=777).long()
    h, v, rnd = stats(toks, SIDE)
    print('  %-22s %8d %11.2f%% %11.2f%% %9.2fx'
          % (name, step, h * 100, v * 100, h / REAL_H))

print()
print('  真实 16x16 基准: 相邻同 H 4.47%%  V 5.68%%')
print()
print('  判读:')
print('    若随 step 单调下降 ⇒ 训练不足，继续训')
print('    若一直 ~30%% ⇒ 结构性坍塌，必须改架构')
print('DONE')
