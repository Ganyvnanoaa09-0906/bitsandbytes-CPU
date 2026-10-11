# -*- coding: utf-8 -*-
"""test_nbr_fuse.py — 邻居融合自检：形状 / 因果安全 / 训练前进 / 生成前进

覆盖：window=0（上+左两个 tap）与 window=1（因果半平面 4 tap）、raster 与 zorder。
"""
import sys, torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import (SmallImageConfigV2, SmallARImageModelV2,
                                 make_token_order)

G, T, V = 16, 256, 1024
print('=' * 88)
print('邻居融合自检   grid=%d T=%d' % (G, T))
print('=' * 88)

for win in (0, 1):
    for order in ('raster', 'zorder'):
        print()
        print('[window=%d  order=%s]' % (win, order))
        cfg = SmallImageConfigV2(vocab_size=V, d_model=512, n_layer=2, n_head=8,
                                 n_kv_head=4, ffn_dim=2048, total_tokens=T, grid=G,
                                 pos_mode='2d', token_order=order, mask_mode='causal',
                                 head_mode='flat', cond_mode='none', use_bos=True,
                                 nbr_fuse=True, nbr_window=win)
        m = SmallARImageModelV2(cfg)
        nb = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print('  params %.2fM   offsets=%s' % (nb / 1e6, m.nbr_offsets))

        tabs = m.nbr_idx_tables                     # (n_tap, T)
        bad = 0
        for ti in range(tabs.shape[0]):
            for p in range(T):
                q = int(tabs[ti, p])
                if q >= 0 and q >= p:
                    bad += 1
        missing = int((tabs < 0).sum())
        print('  邻居位置越界: %d  %s' % (bad, 'OK' if bad == 0 else '❌ 会泄漏'))
        print('  越界边界(缺邻居)的 tap 次数: %d' % missing)

        x = torch.randint(0, V, (2, T))
        out = m.forward_logits(x)
        lg = out[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        print('  训练前向 -> %s 有限值 %s' % (tuple(lg.shape), bool(torch.isfinite(lg).all())))

        m.eval()
        with torch.no_grad():
            s = m.generate(steps=T, top_k=50, temperature=1.0, batch=2, seed=0)
        print('  生成 -> %s  取值[%d,%d]' % (tuple(s.shape), int(s.min()), int(s.max())))

        # 独立验证因果性：改后半段，前半段 logits 必须完全不变
        m.train()
        x2 = x.clone()
        x2[:, T // 2:] = torch.randint(0, V, (2, T - T // 2))
        o1 = m.forward_logits(x)[1]; o1 = o1[0] if isinstance(o1, tuple) else o1
        o2 = m.forward_logits(x2)[1]; o2 = o2[0] if isinstance(o2, tuple) else o2
        d = (o1[:, :T // 2 - 1] - o2[:, :T // 2 - 1]).abs().max().item()
        print('  因果性: 改后半段后前半段 logits 最大差 %.3e  %s'
              % (d, 'OK（无泄漏）' if d < 1e-5 else '❌ 泄漏！'))

print()
print('注：因果性由【构造】保证（邻居的序列位置必然 < 自身，建表时 assert），')
print('    扰动测试是对这个构造的独立验证。')
print('DONE')
