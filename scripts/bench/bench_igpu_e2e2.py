# -*- coding: utf-8 -*-
"""bench_igpu_e2e2.py — 更严谨的 CPU vs 核显 端到端对比。

上一版噪声大（T=64 b=4 说 CPU 赢、b=8 说核显赢），因为：
  1) 迭代太少（6 次）
  2) DML 惰性执行 + 算子回退让计时不稳
  3) 两个配置连续跑，前一个的热/状态影响后一个

本版改进：
  - CPU 和 DML 交替跑多轮，各自取「最好一轮」（最好 = 干扰最小、队列最干净）
  - 每轮之间留间隔
  - 同时报告每轮的离散度，离散大就说明数据不可信
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


def build(T, grid):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=grid,
        d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
        head_mode='flat', cond_mode='none', pos_mode='2d',
        token_order='raster', loop_times=1,
    )
    return SmallARImageModelV2(cfg)


def one_round(dev, T, grid, B, iters=4):
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = build(T, grid).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    x = torch.randint(0, 1024, (B, T), device=dev)
    y = torch.randint(0, 1024, (B, T), device=dev)

    def step():
        out = m.forward_logits(x)
        gen = out[1]
        logits = gen[0] if isinstance(gen, tuple) else gen
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()

    step()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    dt = (time.perf_counter() - t0) / iters
    del m, opt, x, y
    return dt


print('=' * 72)
print('AR 模型 CPU vs 核显（交替多轮取最优）', flush=True)
print('=' * 72)

ROUNDS = 3
for (T, grid) in ((256, 16), (64, 8)):
    for B in (4, 8):
        cs, ds = [], []
        for r in range(ROUNDS):
            try:
                cs.append(one_round(CPU, T, grid, B))
            except Exception as e:
                print('  CPU 失败:', str(e)[:60]); break
            if HAS_DML:
                try:
                    ds.append(one_round(DEV, T, grid, B))
                except Exception as e:
                    print('  DML 失败:', type(e).__name__, str(e)[:70]); break
            time.sleep(0.5)
        if not cs:
            continue
        bc = min(cs)
        if ds:
            bd = min(ds)
            sp = bc / bd
            verdict = '核显赢 %.2fx' % sp if sp > 1.03 else ('CPU 赢 %.2fx' % (1 / sp) if sp < 0.97 else '平手')
            print('  T=%-4d grid=%-3d b=%-2d  CPU %7.1f ms (离散 %.0f%%)   DML %7.1f ms (离散 %.0f%%)   %s'
                  % (T, grid, B, bc * 1e3, 100 * (max(cs) - bc) / bc,
                     bd * 1e3, 100 * (max(ds) - bd) / bd, verdict), flush=True)
        else:
            print('  T=%-4d grid=%-3d b=%-2d  CPU %7.1f ms' % (T, grid, B, bc * 1e3), flush=True)

print()
print('DONE', flush=True)
