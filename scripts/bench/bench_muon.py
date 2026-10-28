# -*- coding: utf-8 -*-
"""bench_muon.py — Muon 的 Newton-Schulz 在我们模型上的真实成本

已知（用户提供的超参）:
  lr=0.02（比 AdamW 大 ~20×）, momentum=0.95 Nesterov
  NS 迭代 5 次, 仅用于 2D 权重矩阵
  Embedding / Router / 低秩参数用 AdamW

关心的问题:
  NS 的成本【与 batch 无关】（作用在权重上）
  而前反向【与 batch 成正比】
  ⇒ 我们的模型小（30M）+ batch 小（8）时，NS 可能是灾难

参考（我们实测过）:
  训练步（30.49M, batch=8, T=256, 6线程）: 2706 ms
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
import os, time
import torch

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
print('=' * 82)
print('Muon / Newton-Schulz 成本实测   线程=%d' % TH)
print('=' * 82)
print()


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Muon 用的 Newton-Schulz 正交化（Quintic 迭代）

    G: 2D tensor。Muon 官方实现（Keller Jordan）
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# 我们模型的 2D 权重矩阵形状（从 small_image_model_v2 的实际结构推算）
# per layer: Q(512,512), K(512,256), V(512,256), O(512,512),
#            FFN gate(512,2048), up(512,2048), down(2048,512)
SHAPES = [
    ('attn.Q',   512,  512),
    ('attn.K',   512,  256),
    ('attn.V',   512,  256),
    ('attn.O',   512,  512),
    ('ffn.gate', 512, 2048),
    ('ffn.up',   512, 2048),
    ('ffn.down', 2048, 512),
]
N_LAYER = 10
BATCH = 8
SEQ = 256
MODEL_PARAMS = 30.49e6

print('  %-12s %-14s %12s %12s' % ('矩阵', '形状', 'us/次', 'GFLOPS当量'))
print('  ' + '-' * 56)

per_mat_us = []
for name, m, n in SHAPES:
    G = torch.randn(m, n)
    # 预热
    for _ in range(2):
        zeropower_via_newtonschulz5(G, 5)
    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        zeropower_via_newtonschulz5(G, 5)
    t = (time.perf_counter() - t0) / reps
    per_mat_us.append(t)
    # FLOP 估算: 约 30·min(m,n)²·max(m,n)
    mm, nn = min(m, n), max(m, n)
    fl = 30 * mm * mm * nn
    print('  %-12s %-14s %12.1f %12.1f' % (name, '%dx%d' % (m, n), t * 1e6, fl / t / 1e9))

print()
per_layer_us = sum(per_mat_us)
total_us = per_layer_us * N_LAYER
print('  单层 NS 合计:  %.2f ms' % (per_layer_us * 1e3))
print('  %d 层合计:      %.1f ms' % (N_LAYER, total_us * 1e3))
print()

# 前反向成本（实测）
print('  对照：实测训练步时')
for B in (8, 16, 32, 64):
    # 前反向 ∝ batch；batch=8 实测 2706 ms
    fwd_bwd_ms = 2706.5 * B / 8
    ns_ms = total_us * 1e3
    print('    batch=%-3d  前反向 %7.1f ms   NS %6.1f ms   ⇒ NS 占比 %5.1f%%'
          % (B, fwd_bwd_ms, ns_ms, 100 * ns_ms / (fwd_bwd_ms + ns_ms)))

print()
print('  前反向的 FLOP 当量（用于对照）:')
for B in (8, 32):
    fl = 2 * MODEL_PARAMS * B * SEQ  # forward 近似
    print('    batch=%-3d  前向 %.1f GFLOP（反向再 ~2×）' % (B, fl / 1e9))
ns_fl = sum(30 * min(m, n) ** 2 * max(m, n) for _, m, n in SHAPES) * N_LAYER
print('    NS 合计 %.1f GFLOP（与 batch 无关）' % (ns_fl / 1e9))

print()
print('=' * 82)
print('判读')
print('=' * 82)
ns_ms = total_us * 1e3
for B in (8, 16, 32, 64):
    fwd_bwd_ms = 2706.5 * B / 8
    pct = 100 * ns_ms / (fwd_bwd_ms + ns_ms)
    verdict = '✅ 可接受' if pct < 15 else ('⚠️ 偏贵' if pct < 40 else '❌ 不可接受')
    print('  batch=%-3d  NS 占 %.1f%%  %s' % (B, pct, verdict))
print()
print('  如果大批量下可接受 ⇒ 【把 batch 开大再上 Muon】')
print('  如果全都不可接受 ⇒ 【只对部分矩阵用 Muon】或【跳过 Muon】')
print('DONE')
