# -*- coding: utf-8 -*-
"""test_l3_full.py — 用【正确配置】重测全模型：L³ 到底能快多少

上一版全模型测试用了 4gram V=4096 k=8（单层 0.78x），所以结论是"L³ 没用"。
k/V/键模式扫描显示：1gram V=1024 系列快得多。

候选（单层实测相对速度）:
  1gram V=1024 k=2   3.43x   1.66M 参数
  1gram V=1024 k=4   2.19x   2.97M 参数   （与 SwiGLU 的 3.15M 接近，公平对比）
  1gram V=1024 k=8   1.73x   5.59M 参数   （容量 1.77x）
  SwiGLU             1.00x   3.15M

测全模型：前向+反向+优化器，B=8 T=256。
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
T, D, NL, G = 256, 512, 10, 16
B = 8


def build(**kw):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=G,
        d_model=D, n_layer=NL, n_head=8, n_kv_head=4, ffn_dim=2048,
        head_mode='flat', cond_mode='none', pos_mode='2d',
        token_order='raster', loop_times=1, use_bos=True, **kw)
    return SmallARImageModelV2(cfg)


x = torch.randint(0, 1024, (B, T))

print('=' * 76)
print('全模型 L³ 对比  (B=%d, T=%d, n_layer=%d, d=%d)' % (B, T, NL, D))
print('=' * 76)

cases = [
    ('SwiGLU 基线',           dict(use_l3=False)),
    ('L³ 1gram V=1024 k=2',   dict(use_l3=True, l3_v=1024, l3_k=2, l3_key_mode='1gram')),
    ('L³ 1gram V=1024 k=4',   dict(use_l3=True, l3_v=1024, l3_k=4, l3_key_mode='1gram')),
    ('L³ 1gram V=1024 k=8',   dict(use_l3=True, l3_v=1024, l3_k=8, l3_key_mode='1gram')),
    ('L³ 4gram V=4096 k=2',   dict(use_l3=True, l3_v=4096, l3_k=2, l3_key_mode='4gram')),
]

results = []
base_dt = None
for tag, kw in cases:
    try:
        torch.manual_seed(0)
        m = build(**kw)
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        npar = sum(p.numel() for p in m.parameters())

        def step():
            o = m.forward_logits(x)
            gg = o[1]
            lg = gg[0] if isinstance(gg, tuple) else gg
            l = F.cross_entropy(lg.reshape(-1, 1024), x.reshape(-1))
            opt.zero_grad(set_to_none=True)
            l.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            return float(l)

        step(); step()
        t0 = time.perf_counter()
        for _ in range(3):
            lv = step()
        dt = (time.perf_counter() - t0) / 3
        if base_dt is None:
            base_dt = dt
        results.append((tag, dt, npar, lv))
        print('  %-24s %7.2f s/步  %6.2fx  %8.2fM 参数  loss=%.3f'
              % (tag, dt, base_dt / dt, npar / 1e6, lv), flush=True)
        del m, opt
    except Exception as e:
        import traceback
        print('  %-24s 失败: %s' % (tag, type(e).__name__ + ': ' + str(e)[:60]), flush=True)

print()
print('  判据: > 1.00x 说明比 SwiGLU 基线快', flush=True)
print('  注: L³ 表是【每层】21M/5.6M, n_layer=10 所以总量是 10 倍', flush=True)
print('DONE', flush=True)
