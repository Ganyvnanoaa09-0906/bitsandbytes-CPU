# -*- coding: utf-8 -*-
"""profile_ar_gen.py — 拆解 AR 采样的 16.43 ms/token 到底花在哪

已知:
  完整前向（一次算 256 位置）:  125.9 ms  = 0.492 ms/token
  generate（KV cache 逐 token）: 4207 ms  = 16.43 ms/token
  ⇒ 97% 是开销

要拆的嫌疑:
  ① Python 循环本身（256 次迭代）
  ② 每步重建 cond / pos / mask
  ③ KV cache 用 torch.cat 每步重新分配（O(T²) 次分配）
  ④ 每步的算子数量（dispatch 开销）
  ⑤ logits 的 top-k 采样
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

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 80)
print('AR 采样开销拆解')
print('=' * 80)

CKPT = sys.argv[1] if len(sys.argv) > 1 else r'D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\ckpt_12000.pt'
print('  checkpoint:', CKPT)
ck = torch.load(CKPT,
                map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'])
m.eval()
T = cfg.total_tokens
print('  模型: %.2fM  T=%d  loop=%d  use_bos=%s' % (
    sum(p.numel() for p in m.parameters()) / 1e6, T, cfg.loop_times, cfg.use_bos))
print()

x = torch.randint(0, 1024, (1, T))

with torch.no_grad():
    # --- 基准 ---
    for _ in range(3):
        m.forward_logits(x)
    t0 = time.perf_counter()
    for _ in range(10):
        m.forward_logits(x)
    t_fwd = (time.perf_counter() - t0) / 10
    print('  ① 完整前向（无 cache，256 位置一次算完）: %8.1f ms' % (t_fwd * 1e3))

    # --- 逐 token 前向，但不带 cache/采样，纯看循环开销 ---
    t0 = time.perf_counter()
    for i in range(T):
        m.forward_logits(x[:, :i + 1])
    t_loop = (time.perf_counter() - t0) / T
    print('  ② 纯 Python 循环 + 每次重算全部前缀:      %8.2f ms/token' % (t_loop * 1e3))

    # --- 单步 embed + blocks + head（最小单元）---
    emb = m.embed(x[:, :1])
    for _ in range(5):
        m.run_blocks(emb)
    t0 = time.perf_counter()
    for _ in range(50):
        m.run_blocks(emb)
    t_blocks = (time.perf_counter() - t0) / 50
    print('  ③ 单个 token 过全部 block（T=1）:        %8.2f ms' % (t_blocks * 1e3))

    # --- 单步完整 forward_logits（T=1）---
    x1 = x[:, :1]
    for _ in range(5):
        m.forward_logits(x1)
    t0 = time.perf_counter()
    for _ in range(50):
        m.forward_logits(x1)
    t_one = (time.perf_counter() - t0) / 50
    print('  ④ 单个 token 完整 forward_logits（T=1）: %8.2f ms' % (t_one * 1e3))

    # --- generate 全流程 ---
    for _ in range(1):
        m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=1, seed=1)
    t0 = time.perf_counter()
    for _ in range(2):
        m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=1, seed=1)
    t_gen = (time.perf_counter() - t0) / 2
    print('  ⑤ generate batch=1 全流程:               %8.2f ms/token' % (t_gen / T * 1e3))
    print()

print('  分析:')
print('    ⑤ generate        = %.2f ms/token' % (t_gen / T * 1e3))
print('    ④ T=1 forward      = %.2f ms/token   ← 理论下界（无 cache）' % t_one)
print('    ③ 仅 blocks        = %.2f ms' % t_blocks)
print('    ⇒ generate 比 T=1 forward 慢 %.1f 倍' % ((t_gen / T) / (t_one / 1e3)))
print()
print('    如果是 cache 反而更慢，说明 cat 重分配是负收益')

# --- 单独测 KV cache 的 cat 开销 ---
print()
print('  --- KV cache 的 torch.cat 开销 ---')
for Tcur in (64, 128, 256):
    k = torch.randn(1, 8, Tcur, 64)
    v = torch.randn(1, Tcur, 64)
    t0 = time.perf_counter()
    for _ in range(500):
        _ = torch.cat([k, torch.randn(1, 8, 1, 64)], dim=2)
    dt = (time.perf_counter() - t0) / 500
    print('    cat 到长度 %3d: %7.1f us  ⇒ %d 个 block × %d 层 = %.1f ms/step'
          % (Tcur, dt * 1e6, cfg.n_layer, 2, dt * 1e3 * cfg.n_layer * 2))
print('DONE')
