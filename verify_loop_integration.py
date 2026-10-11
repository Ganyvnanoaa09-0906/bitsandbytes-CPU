# -*- coding: utf-8 -*-
"""verify_loop_integration.py — 验证循环接入后不破坏原有行为

五件事：
  1 默认（loop_cfg=None）时前向正常
  2 挂载循环后前向正常，L=1 时与未挂载接近
  3 反向，循环模块的梯度可达
  4 采样（generate）路径正常 —— 最容易被漏掉
  5 L=3 时也要能跑
"""
import sys
import torch
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, run_blocks
from loop_transformer import LoopConfig

torch.set_num_threads(6)
torch.manual_seed(1234)
print('=' * 78)
print('循环接入验证')
print('=' * 78)

cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=True, coarse_grid=4, loop_start=4, loop_end=7, loop_times=1,
    fuse_qkv=False, use_bos=True)
m = SmallARImageModelV2(cfg)
m.eval()
tok = torch.randint(0, 1024, (2, 256))


def unwrap(t):
    return t[0] if isinstance(t, (tuple, list)) else t


print()
print('[1] 默认（未挂载循环）')
with torch.no_grad():
    t1, g1 = m.forward_logits(tok)
t1, g1 = unwrap(t1), unwrap(g1)
print('  think %s   gen %s' % (tuple(t1.shape), tuple(g1.shape)))
print('  loop_cfg=%s  loop_mods=%s' % (m.loop_cfg, m.loop_mods))
with torch.no_grad():
    x = m.embed_bos(tok)
    x2 = run_blocks(m.blocks, x, m.loop_idx, cfg.loop_times, None, None, None)
print('  手工 run_blocks 输出范数 %.6f' % float(x2.norm()))

print()
print('[2] 挂载循环')
lc = LoopConfig(loop_start=4, loop_end=7, loop_L=1, loop_L_max=3, loop_random=False)
m.set_loop(lc)
print('  loop_mods keys: %s' % (list(m.loop_mods.keys()) if m.loop_mods else None))
with torch.no_grad():
    t2, g2 = m.forward_logits(tok)
t2, g2 = unwrap(t2), unwrap(g2)
print('  think %s   gen %s' % (tuple(t2.shape), tuple(g2.shape)))
d = float((g1 - g2).abs().max())
print('  L=1 与未挂载的最大绝对差 %.4e  %s' % (d, 'OK' if d < 0.1 else 'WARN'))
print('  循环模块参数 %d' % sum(p.numel() for p in m.loop_mods.parameters()))

print()
print('[3] 反向')
m.train()
t3, g3 = m.forward_logits(tok)
g3 = unwrap(g3)
loss = torch.nn.functional.cross_entropy(g3.reshape(-1, 1024), tok.reshape(-1))
loss.backward()
n_grad = sum(1 for p in m.parameters() if p.grad is not None)
n_all = sum(1 for _ in m.parameters())
lm_grad = sum(1 for p in m.loop_mods.parameters() if p.grad is not None)
lm_all = sum(1 for _ in m.loop_mods.parameters())
print('  loss=%.4f  总梯度 %d/%d' % (float(loss), n_grad, n_all))
print('  循环模块梯度 %d/%d  %s' % (lm_grad, lm_all, 'OK' if lm_grad == lm_all else 'FAIL'))

print()
print('[4] 采样路径')
m.eval()
with torch.no_grad():
    out = m.generate(None, steps=16, top_k=64, temperature=1.0, batch=2, seed=1)
print('  generate 输出 %s  (应为 (2,16))' % (tuple(out.shape),))
ok = 0 <= int(out.min()) and int(out.max()) < 1024
print('  token 范围 [%d, %d]  %s' % (int(out.min()), int(out.max()), 'OK' if ok else 'FAIL'))

print()
print('[5] L=3')
lc3 = LoopConfig(loop_start=4, loop_end=7, loop_L=3, loop_L_max=3, loop_random=False)
m.set_loop(lc3)
with torch.no_grad():
    t4, g4 = m.forward_logits(tok)
g4 = unwrap(g4)
print('  L=3 前向 OK, gen %s' % (tuple(g4.shape),))
with torch.no_grad():
    out3 = m.generate(None, steps=8, top_k=64, temperature=1.0, batch=2, seed=1)
print('  L=3 generate OK, %s' % (tuple(out3.shape),))

print()
print('=' * 78)
print('DONE')
