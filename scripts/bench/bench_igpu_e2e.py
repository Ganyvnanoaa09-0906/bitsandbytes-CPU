# -*- coding: utf-8 -*-
"""bench_igpu_e2e.py — 决定性实验：真实 AR 模型在核显上的端到端收益。

背景：注意力在 DML 上慢 5~6×，但我们的注意力只占算力 3.3%。
      决定性的是「注意力额外的绝对开销」能否被「GEMM 的 1.30×」盖过。

按 seq 扫描：
  T=256（现在）  →  预期净负
  T= 64（8x8 方案）→  预期转正

必须在 .venv_dml 里跑。
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
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch_directml as dml
    HAS_DML = True
except Exception:
    HAS_DML = False

torch.set_num_threads(6)
CPU = torch.device('cpu')
DEV = dml.device() if HAS_DML else CPU

from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2


def build(T, grid, d_model=512, n_layer=10):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=grid,
        d_model=d_model, n_layer=n_layer, n_head=8, n_kv_head=4,
        ffn_dim=2048, head_mode='flat', cond_mode='none',
        pos_mode='2d', token_order='raster', loop_times=1,
    )
    return SmallARImageModelV2(cfg)


def bench(fn, iters=6, warm=2):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def run(dev, tag, T, grid, B):
    import torch.nn.functional as F
    torch.manual_seed(0)
    try:
        m = build(T, grid).to(dev)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)   # 只为 .zero_grad()，不测优化器
        x = torch.randint(0, 1024, (B, T), device=dev)
        y = torch.randint(0, 1024, (B, T), device=dev)

        def step():
            out = m.forward_logits(x)
            gen = out[1]
            logits = gen[0] if isinstance(gen, tuple) else gen
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()

        dt = bench(step, iters=6, warm=2)
        return dt
    except Exception as e:
        print('    %-4s 失败: %s' % (tag, type(e).__name__ + ': ' + str(e)[:80]), flush=True)
        return None


print('=' * 72, flush=True)
print('AR 模型端到端：CPU vs 核显   (forward+backward, 不含优化器)', flush=True)
print('DirectML:', HAS_DML, '|', (dml.device_name(0) if HAS_DML else '-'), flush=True)
print('=' * 72, flush=True)
print('  %-22s %10s %10s %10s' % ('配置', 'CPU ms', 'DML ms', 'DML/CPU'), flush=True)

for (T, grid) in ((256, 16), (64, 8)):
    for B in (4, 8):
        tc = run(CPU, 'CPU', T, grid, B)
        td = run(DEV, 'DML', T, grid, B) if HAS_DML else None
        if tc and td:
            print('  T=%-4d grid=%-3d b=%-3d %10.1f %10.1f %9.2fx  %s'
                  % (T, grid, B, tc * 1e3, td * 1e3, td / tc,
                     '核显赢' if td < tc else 'CPU 赢'), flush=True)
        elif tc:
            print('  T=%-4d grid=%-3d b=%-3d %10.1f %10s' % (T, grid, B, tc * 1e3, '-'), flush=True)

print()
print('DONE', flush=True)
