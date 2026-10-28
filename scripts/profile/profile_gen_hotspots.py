# -*- coding: utf-8 -*-
"""profile_gen_hotspots.py — 拆 AR 生成每步的 60~74 ms 花在哪

已知:
  batch=16  实测 60.64 ms/step，理论地板 7.87 ms  ⇒ 7.7× 开销
  batch=24  实测 74.17 ms/step，理论地板 11.8 ms  ⇒ 6.3× 开销

嫌疑（按怀疑度）:
  ① GQA 的 repeat_interleave —— 每步每层都调用，是慢算子
  ② KV cache 的 torch.cat —— 每步重新分配 + 拷贝
  ③ 每步重建 cond / pos / mask
  ④ Python 循环 + 算子 dispatch
  ⑤ top-k 采样
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
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 84)
print('AR 生成热点拆解')
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
print('  模型: %.2fM  T=%d  n_layer=%d  nh=%d  n_kv=%d' % (
    sum(p.numel() for p in m.parameters()) / 1e6, T, cfg.n_layer, cfg.n_head, cfg.n_kv_head))
print()

B = 16
print('  --- ① GQA repeat_interleave 的开销 ---')
nh, nkv, hd = cfg.n_head, cfg.n_kv_head, cfg.d_model // cfg.n_head
rep = nh // nkv
for Tk in (64, 128, 256):
    k = torch.randn(B, nkv, Tk, hd)
    v = torch.randn(B, nkv, Tk, hd)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = k.repeat_interleave(rep, dim=1)
        _ = v.repeat_interleave(rep, dim=1)
    dt = (time.perf_counter() - t0) / 200
    print('    T=%-4d  repeat_interleave 一对: %7.1f us  × %d 层 = %6.2f ms/step'
          % (Tk, dt * 1e6, cfg.n_layer, dt * 1e3 * cfg.n_layer))
# 对照：expand 不拷贝
k = torch.randn(B, nkv, 256, hd)
t0 = time.perf_counter()
for _ in range(200):
    _ = k.repeat_interleave(rep, dim=1)
dt_r = (time.perf_counter() - t0) / 200
t0 = time.perf_counter()
for _ in range(200):
    _ = k[:, :, None].expand(B, nkv, rep, 256, hd).reshape(B, nh, 256, hd)
dt_e = (time.perf_counter() - t0) / 200
print('    对照: repeat_interleave %.1f us  vs  expand+reshape %.1f us  ⇒ %.1fx'
      % (dt_r * 1e6, dt_e * 1e6, dt_r / dt_e if dt_e else 0))

print()
print('  --- ② KV cache torch.cat 的开销 ---')
for Tk in (64, 128, 256):
    kc = torch.randn(B, nkv, Tk, hd)
    kn = torch.randn(B, nkv, 1, hd)
    t0 = time.perf_counter()
    for _ in range(300):
        _ = torch.cat([kc, kn], dim=2)
    dt = (time.perf_counter() - t0) / 300
    print('    cat %d+1 -> %d: %7.1f us  × %d 层 × 2 = %6.2f ms/step'
          % (Tk, Tk + 1, dt * 1e6, cfg.n_layer, dt * 1e3 * cfg.n_layer * 2))

print()
print('  --- ③ scaled_dot_product_attention 本身 ---')
q = torch.randn(B, nh, 1, hd)
for Tk in (64, 128, 256):
    kk = torch.randn(B, nh, Tk, hd)
    vv = torch.randn(B, nh, Tk, hd)
    t0 = time.perf_counter()
    for _ in range(200):
        _ = F.scaled_dot_product_attention(q, kk, vv, is_causal=False)
    dt = (time.perf_counter() - t0) / 200
    print('    SDPA (1 x %4d): %7.1f us  × %d 层 = %6.2f ms/step'
          % (Tk, dt * 1e6, cfg.n_layer, dt * 1e3 * cfg.n_layer))

print()
print('  --- ④ 整个 generate 的每步实测 ---')
with torch.no_grad():
    m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
    t0 = time.perf_counter()
    m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
    dt = time.perf_counter() - t0
print('    batch=%d  generate: %.2f ms/step  (%.2f ms/图)'
      % (B, dt / T * 1e3, dt / B * 1e3))
print()
print('DONE')
