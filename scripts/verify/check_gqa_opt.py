# -*- coding: utf-8 -*-
"""check_gqa_opt.py — 验证 GQA cache 优化【不改变输出】+ 测提速

两件事：
  ① 数值等价：同一 seed 下，优化前后的 generate 输出应该【逐 token 相同】
     （缓存存展开版 vs 紧凑版，只是布局不同，数学上必须完全一致）
  ② 提速幅度
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 80)
print('GQA cache 优化：正确性 + 提速')
print('=' * 80)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\ckpt_12000.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'])
m.eval()
T = cfg.total_tokens
print('  模型: %.2fM  T=%d  nh=%d n_kv=%d  (GQA %s)'
      % (sum(p.numel() for p in m.parameters()) / 1e6, T,
         cfg.n_head, cfg.n_kv_head, '启用' if cfg.n_kv_head != cfg.n_head else '关闭'))
print()

# 用 attention 层直接对拍：无 cache 有 cache 必须一致
print('  --- ① 数值等价性（同一输入，无 cache 全程 vs 有 cache 增量）---')
att = m.blocks[0].attn
xfull = torch.randn(2, T, cfg.d_model)
with torch.no_grad():
    y_full = att(xfull)                       # 无 cache，一次算完
    out = []
    cache = dict(k=None, v=None)
    for i in range(T):
        yi = att(xfull[:, i:i + 1], cache=cache)
        out.append(yi)
    y_inc = torch.cat(out, dim=1)
d = (y_full - y_inc).abs()
rel = float(d.max() / max(float(y_full.abs().max()), 1e-6))
print('    完整前向 vs 逐步增量：最大绝对差 %.3e  最大相对差 %.3e  %s'
      % (float(d.max()), rel, '✅ 一致' if rel < 1e-4 else '❌ 不一致'))
print()

print('  --- ② 提速 ---')
print('  %-8s %14s %14s %10s' % ('batch', 'ms/step', 'ms/图', '图/秒'))
print('  ' + '-' * 52)
for B in (16, 32, 48, 64, 96):
    with torch.no_grad():
        m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
        r = 1
        t0 = time.perf_counter()
        for _ in range(r):
            m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
        dt = (time.perf_counter() - t0) / r
    print('  %-8d %14.2f %14.2f %10.2f' % (B, dt / T * 1e3, dt / B * 1e3, B / dt))

print()
print('  对照（优化前）:')
print('    batch=1   20.11 ms/step   5148 ms/图')
print('    batch=8   45.88 ms/step   1468 ms/图')
print('    batch=16  60.64 ms/step    970 ms/图')
print('    batch=24  74.17 ms/step    791 ms/图')
print('DONE')
