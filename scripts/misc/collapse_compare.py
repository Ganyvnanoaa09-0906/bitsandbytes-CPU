# -*- coding: utf-8 -*-
"""collapse_compare.py — 对比不同架构的 T=256 坍塌程度

问题：坍塌是 T=256 的通用现象，还是 hier 特有的？
对比：
  · hier      (16x16, 两阶段 coarse→fine)
  · ar_flat   (16x16, 单阶段)
  · ar_l3     (16x16, L³ 层)
  · loop1     (16x16, 循环)
  · ar_swiglu (8x8,  单阶段)  ← 对照组，已知不坍塌
  · 真实图     ← 基准
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys, os, time
import torch
import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 84)
print('不同架构的 T=256 坍塌程度对比')
print('=' * 84)

HERE = r'D:\work\cloud_results'
N = 64


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
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    return m, cfg, ck.get('step', -1), len(missing), len(unexpected)


def stats(toks, side):
    B = toks.shape[0]
    g = toks.view(B, side, side)
    h = (g[:, :, :-1] == g[:, :, 1:]).float().mean().item()
    v = (g[:, :-1, :] == g[:, 1:, :]).float().mean().item()
    flat = toks.reshape(-1)
    perm = flat[torch.randperm(flat.numel())]
    rnd = (flat == perm).float().mean().item()
    # 最长同色游程
    runs = []
    for b in range(B):
        gg = g[b]
        r = 1
        for i in range(1, side * side):
            if gg.reshape(-1)[i] == gg.reshape(-1)[i - 1]:
                r += 1
            else:
                runs.append(r); r = 1
        runs.append(r)
    return h, v, rnd, float(np.mean(runs))


CASES = [
    ('ar_swiglu T=64  (8x8)', 'swiglu_last.pt', 8, 0),
    ('hier  T=256 (16x16)', 'hier_last.pt', 16, 0),
    ('ar_flat T=256(16x16)', 'flat_12000.pt', 16, 0),
    ('ar_l3  T=256 (16x16)', 'l3_12000.pt', 16, 0),
    ('loop1  T=256 (16x16)', 'loop1_26500.pt', 16, 3),
]

print()
print('  %-24s %8s %10s %10s %8s %10s %10s' % (
    '模型', 'step', '相邻同H', '相邻同V', '随机', '平均游程', '相对真实'))
print('  ' + '-' * 84)

REAL = {8: 0.0279, 16: 0.0447}
rows = []
for label, fn, side, L in CASES:
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        print('  %-24s （缺文件）' % label)
        continue
    m, cfg, step, nm, nu = load_ar(p)
    if cfg.total_tokens != side * side:
        side = int(round(cfg.total_tokens ** 0.5))
    t0 = time.perf_counter()
    with torch.no_grad():
        toks = m.generate(None, steps=cfg.total_tokens, top_k=1024, temperature=1.0,
                          batch=N, seed=777).long()
    dt = time.perf_counter() - t0
    h, v, rnd, run = stats(toks, side)
    rows.append((label, side, h, v))
    print('  %-24s %8d %9.2f%% %9.2f%% %7.2f%% %9.2f %9.2fx'
          % (label, step, h * 100, v * 100, rnd * 100, run, h / REAL.get(side, h)))

print()
print('  【真实图基准】')
print('    8x8 : 相邻同 2.79%%(H)  3.19%%(V)')
print('    16x16: 相邻同 4.47%%(H)  5.68%%(V)')
print()
print('  判读:')
print('    若所有 T=256 架构都坍塌  ⇒ 是序列长度/范式问题（与架构细节无关）')
print('    若只有部分坍塌          ⇒ 是特定架构的问题')
print('DONE')
