# -*- coding: utf-8 -*-
"""test_loop2.py — 关键对比：同样的算力预算，循环好还是加层好？

  A) n_layer=10, loop_times=2   10 层参数跑 14 次   30.49M
  B) n_layer=10, loop_times=1   10 层参数跑 10 次   30.49M
  C) n_layer=14, loop_times=1   14 层参数跑 14 次   约 42.7M   ← 与 A 等算力
  D) n_layer=14, loop_times=2   14 层参数跑 20 次   约 42.7M

如果 C ≈ A 的耗时，就说明「循环」这个机制在纯算力上没有优势，
而 C 比 A 多了 40% 参数 —— 明显更划算（除非循环的权重共享起了正则作用）。
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
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

T, B = 256, 8


def build(n_layer, loop_times):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=16,
        d_model=512, n_layer=n_layer, n_head=8, n_kv_head=4,
        ffn_dim=2048, head_mode='flat', cond_mode='none',
        pos_mode='2d', token_order='raster',
        loop_start=4, loop_end=7, loop_times=loop_times,
    )
    return SmallARImageModelV2(cfg)


def measure(n_layer, loop_times, threads=6, iters=4):
    torch.set_num_threads(threads)
    m = build(n_layer, loop_times)
    nparam = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x = torch.randint(0, 1024, (B, T))
    y = torch.randint(0, 1024, (B, T))

    def step():
        out = m.forward_logits(x)
        gen = out[1]
        logits = gen[0] if isinstance(gen, tuple) else gen
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(2):
        step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = (time.perf_counter() - t0) / iters
    del m, opt, x, y
    return dt, nparam


print('=' * 74, flush=True)
print('等算力对比（b=%d T=%d threads=6）' % (B, T), flush=True)
print('=' * 74, flush=True)
print('  %-30s %8s %9s %10s' % ('配置', 's/步', 'tok/s', '参数'), flush=True)

cases = [
    ('A) n_layer=10 loop_times=2', 10, 2),
    ('B) n_layer=10 loop_times=1', 10, 1),
    ('C) n_layer=14 loop_times=1', 14, 1),
    ('D) n_layer=14 loop_times=2', 14, 2),
    ('E) n_layer=18 loop_times=1', 18, 1),
]
res = {}
for tag, nl, lt in cases:
    dt, np_ = measure(nl, lt)
    res[tag] = (dt, np_)
    print('  %-30s %8.3f %9.1f %9.2fM' % (tag, dt, B * T / dt, np_ / 1e6), flush=True)

print()
a = res['A) n_layer=10 loop_times=2']
c = res['C) n_layer=14 loop_times=1']
print('  A vs C（本应等算力）: A %.3fs  C %.3fs  →  C 比 A %s %.1f%%'
      % (a[0], c[0], '快' if c[0] < a[0] else '慢', 100 * abs(c[0] - a[0]) / a[0]), flush=True)
print('  参数量: A %.2fM  →  C %.2fM  (+%.0f%%)'
      % (a[1] / 1e6, c[1] / 1e6, 100 * (c[1] - a[1]) / a[1]), flush=True)
print()
b = res['B) n_layer=10 loop_times=1']
print('  单纯关循环（B）比现在（A）快 %.1f%%' % (100 * (a[0] - b[0]) / a[0]), flush=True)
print('DONE', flush=True)
