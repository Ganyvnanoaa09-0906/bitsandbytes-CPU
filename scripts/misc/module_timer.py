# -*- coding: utf-8 -*-
"""module_timer.py — 用 forward hook 给每个子模块计时（profiler 在本机会段错误）。

输出每个模块类型的总耗时占比，找出真正的热点。
注意：hook 测的是「包含子模块」的墙钟时间，所以父模块会包含子模块，
      看的时候要看「叶子模块」（Hierarchy 里最底层的）。
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
from collections import defaultdict
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

T, B, THREADS = 256, 8, 6
torch.set_num_threads(THREADS)

cfg = SmallImageConfigV2(
    vocab_size=1024, total_tokens=T, grid=16,
    d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, head_mode='flat', cond_mode='none',
    pos_mode='2d', token_order='raster', loop_times=1,
)
m = SmallARImageModelV2(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
x = torch.randint(0, 1024, (B, T))
y = torch.randint(0, 1024, (B, T))

# 给每个叶子模块挂 hook
acc = defaultdict(float)
cnt = defaultdict(int)
leaf = set()


def all_leaf(mod):
    for name, child in mod.named_children():
        kids = list(child.children())
        if kids:
            all_leaf(child)
        else:
            leaf.add(child)


all_leaf(m)


def pre_hook(mod, inp):
    mod._t0 = time.perf_counter()


def post_hook(mod, inp, out):
    dt = time.perf_counter() - getattr(mod, '_t0', time.perf_counter())
    key = type(mod).__name__
    acc[key] += dt
    cnt[key] += 1


handles = []
for mod in leaf:
    handles.append(mod.register_forward_pre_hook(pre_hook))
    handles.append(mod.register_forward_hook(post_hook))


def step():
    out = m.forward_logits(x)
    gen = out[1]
    logits = gen[0] if isinstance(gen, tuple) else gen
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


step()
acc.clear(); cnt.clear()

N = 3
t0 = time.perf_counter()
for _ in range(N):
    step()
total = (time.perf_counter() - t0) / N

for h in handles:
    h.remove()

print('=' * 62, flush=True)
print('AR 模型叶子模块耗时分解  (b=%d T=%d threads=%d)' % (B, T, THREADS), flush=True)
print('单步墙钟 %.3f s   叶子模块合计 %.3f s (%.1f%%)'
      % (total, sum(acc.values()) / N, 100 * sum(acc.values()) / N / total), flush=True)
print('=' * 62, flush=True)
print('  %-18s %9s %8s %10s' % ('模块类型', '总 ms/步', '调用次数', '单次 µs'), flush=True)
rows = sorted(acc.items(), key=lambda kv: -kv[1])
for k, v in rows[:16]:
    ms = v / N * 1e3
    c = cnt[k] // 1
    print('  %-18s %9.2f %8d %10.1f' % (k, ms, c / 1, ms / max(1, c) * 1e3), flush=True)

print()
print('  其余时间（%.1f%%）= 非叶子算子 + Python 调度 + autograd'
      % (100 * (total - sum(acc.values()) / N) / total), flush=True)
print('DONE', flush=True)
