# -*- coding: utf-8 -*-
"""bench_bnb_opt.py — bnb-cpu 的 AdamW8bit 在本机到底比 torch 原生慢还是快。

我们的训练只用 bnb 的 AdamW8bit。之前测过优化器占单步 3.4%，
但如果 bnb 版明显慢于 torch 原生，那就是白送的收益。

同时测不同参数量下的表现（我们 AR 模型 30.49M）。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, sys, time
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")
torch.set_num_threads(6)


def bench(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


print('=' * 66, flush=True)
print('bnb-cpu AdamW8bit  vs  torch.optim.AdamW', flush=True)
print('=' * 66, flush=True)

try:
    import bitsandbytes as bnb
    print('  bitsandbytes 版本:', getattr(bnb, '__version__', '?'))
    print('  可用优化器:', [x for x in dir(bnb.optim) if not x.startswith('_')][:12])
    print('  AdamW8bit 存在:', hasattr(bnb.optim, 'AdamW8bit'))
except Exception as e:
    print('  import bitsandbytes 失败:', e)
    sys.exit(1)

print()
print('  %-14s %14s %14s %10s' % ('参数量', 'torch AdamW', 'bnb AdamW8bit', '差距'), flush=True)

for dim, nlayer, name in ((128, 4, '0.1M'), (512, 4, '1.3M'), (1024, 8, '8.4M'),
                          (2048, 10, '42M'), (2816, 10, '79M')):
    m = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(nlayer)])
    nparam = sum(p.numel() for p in m.parameters())
    grads = [torch.randn_like(p) for p in m.parameters()]

    results = {}
    for tag, cls in (('torch', torch.optim.AdamW), ('bnb', bnb.optim.AdamW8bit)):
        try:
            opt = cls(m.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
            # 预热：先走几步让 state 建起来
            for _ in range(3):
                for p, g in zip(m.parameters(), grads):
                    p.grad = g.clone()
                opt.step()
                opt.zero_grad(set_to_none=True)

            def step():
                for p, g in zip(m.parameters(), grads):
                    p.grad = g
                opt.step()
                opt.zero_grad(set_to_none=True)

            results[tag] = bench(step, iters=40, warm=8)
        except Exception as e:
            print('  %-14s %-14s 失败: %s' % (name, tag, str(e)[:50]), flush=True)
            results[tag] = float('nan')

    a, b = results.get('torch'), results.get('bnb')
    if a and b and a == a and b == b:
        print('  %-14s %11.3f ms %11.3f ms %9.1f%%'
              % ('%s (%.1fM)' % (name, nparam / 1e6), a * 1e3, b * 1e3,
                 100 * (a - b) / a), flush=True)
    del m, grads

print()
print('  正数 = bnb 更快；负数 = bnb 更慢', flush=True)
print('DONE', flush=True)
