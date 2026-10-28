# -*- coding: utf-8 -*-
"""bench_gen_batch.py — AR 生成的批量 scaling（找最优批量）

背景修正：
  逐 token 解码是【访存受限】的 —— 每步都要读完整个模型权重
  模型 30.49M 参数 = 122 MB，CPU 带宽 29.8 GB/s ⇒ 单步地板 ~4.1 ms
  （★ 2026-09-25 更正：原写「18.8 GB/s ⇒ ~6.5 ms」。18.8 已被报告 §10.79 撤回，
    权威口径见 §10.80：CPU 普通 29.8 / NT 30.8 GB/s。122 MB ÷ 29.8 = 4.09 ms。）
  实测 batch=1 是 16.43 ms/token（2.5× 地板开销）

  之前测 batch=8 是 37.76 ms/step（看起来更慢），但【每图】只有 4.72 ms
  ⇒ 每图 3.48× 提速。这个要扫清楚。
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 84)
print('AR 生成：批量 scaling')
print('=' * 84)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\ckpt_12000.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'])
m.eval()
T = cfg.total_tokens
NP = sum(p.numel() for p in m.parameters())
print('  模型: %.2fM 参数 = %.0f MB   T=%d  loop=%d' % (NP / 1e6, NP * 4 / 1e6, T, cfg.loop_times))
print()

BW = 29.8e9
# ★★ 2026-09-25 修正：原写 BW = 18.8e9（已被 §10.79 撤回）。
#   注意这【不是显示字符串，是计算】—— 它决定"单步地板"这个参照值：
#   122 MB / 18.8 = 6.49 ms（偏高 59%）vs 122 MB / 29.8 = 4.09 ms。
#   地板偏高会让实测值看起来"更接近地板"，从而低估"开销"那一部分。
#   权威口径见报告 §10.80。
floor_ms = NP * 4 / BW * 1e3
print('  单步访存地板 = %.0f MB / %.1f GB/s = 【%.2f ms】' % (NP * 4 / 1e6, BW / 1e9, floor_ms))
print('  （每步都必须读完整个权重，与 batch 无关）')
print()
print('  %-8s %12s %12s %12s %10s' % ('batch', 'ms/step', 'ms/图', '图/秒', '相对 b=1'))
print('  ' + '-' * 60)

base = None
for B in (1, 2, 4, 8, 16, 24):
    try:
        with torch.no_grad():
            m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
            r = 2 if B <= 8 else 1
            t0 = time.perf_counter()
            for _ in range(r):
                m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
            dt = (time.perf_counter() - t0) / r
        per_step = dt / T * 1e3
        per_img = dt / B * 1e3
        if base is None:
            base = per_img
        print('  %-8d %12.2f %12.2f %12.1f %9.2fx'
              % (B, per_step, per_img, B / dt, base / per_img))
    except Exception as e:
        print('  %-8d 失败: %s' % (B, str(e)[:50]))

print()
print('  理论: ms/step = max(访存地板 %.2f, 计算量/算力)' % floor_ms)
print('        计算量(batch B) = 2 × %.1fM × B FLOP' % (NP / 1e6))
for B in (1, 8, 16, 32):
    c = 2 * NP * B / 124e9 * 1e3
    print('        batch=%-3d 计算 %.2f ms  访存 %.2f ms  ⇒ 理论 max %.2f ms/step'
          % (B, c, floor_ms, max(c, floor_ms)))
print('DONE')
