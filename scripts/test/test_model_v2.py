# -*- coding: utf-8 -*-
"""test_model_v2.py - 验证 v2：形状 / 参数 / KV Cache 正确性 / 推理加速。

注意：本环境 stdout 是 GBK，脚本里不要用 emoji。
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, run_blocks, count_params

torch.manual_seed(0)
torch.set_num_threads(6)

cfg = SmallImageConfigV2(d_model=512, n_layer=10, n_head=8, n_kv_head=8, ffn_dim=2048)
m = SmallARImageModelV2(cfg)
print('=== v2 参数 ===')
print('  params: %.2fM' % (count_params(m) / 1e6))
print('  cond_mode=%s pos=%s order=%s head=%s n_kv_head=%d' % (
    cfg.cond_mode, cfg.pos_mode, cfg.token_order, cfg.head_mode, cfg.n_kv_head))

B, T = 2, 256
tokens = torch.randint(0, 1024, (B, T))
cond = torch.randn(B, 77, 512)

th, ge = m.forward_logits(tokens, cond)
full = m.merge(ge[0], ge[1], None)
print()
print('=== 形状 ===')
print('  think cluster %s / code %s' % (tuple(th[0].shape), tuple(th[1].shape)))
print('  gen   cluster %s / code %s' % (tuple(ge[0].shape), tuple(ge[1].shape)))
print('  merged gen logits %s  (vocab=%d)' % (tuple(full.shape), cfg.vocab_size))

# ---------- KV Cache 正确性 ----------
print()
print('=== KV Cache 正确性 ===')
m.eval()
P = 40
with torch.no_grad():
    pooled, ctx = m.prep_cond(cond, B, tokens.device)
    # A) 全量前向
    xa = m.embed(tokens[:, :P], pos_offset=0)
    xa = run_blocks(m.blocks, xa, m.loop_idx, cfg.loop_times, pooled, ctx, None)
    la = m.merge(*m.gen_head.logits(xa[:, -1:]), None)[:, -1]
    # B) 增量前向（带 cache）
    caches = [dict(k=None, v=None) for _ in range(cfg.n_layer)]
    for t in range(P):
        xt = m.embed(tokens[:, t:t + 1], pos_offset=t)
        xt = run_blocks(m.blocks, xt, m.loop_idx, cfg.loop_times, pooled, ctx, caches)
    lb = m.merge(*m.gen_head.logits(xt), None)[:, -1]

d = (la - lb).abs().max().item()
print('  全量 vs 增量 最大差异: %.3e' % d)
print('  结论: %s' % ('一致 PASS' if d < 1e-4 else '不一致 FAIL'))

# ---------- 推理速度 ----------
print()
print('=== 推理速度（生成 %d token）===' % 48)
N = 48


def gen_full(model, c, n):
    B = c.shape[0]
    x = torch.zeros(B, 1, dtype=torch.long)
    for _ in range(n):
        cl, cd = model.forward_logits(x, c)[1]
        lg = model.merge(cl, cd, None)[:, -1]
        x = torch.cat([x, torch.multinomial(F.softmax(lg, dim=-1), 1)], dim=1)
    return x


with torch.no_grad():
    t0 = time.perf_counter(); gen_full(m, cond, N); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); m.generate(cond, steps=N, top_k=100); t2 = time.perf_counter() - t0
print('  全量重算（v1 行为）: %.1fs  (%.3f s/token)' % (t1, t1 / N))
print('  KV Cache（v2）     : %.1fs  (%.3f s/token)' % (t2, t2 / N))
print('  加速: %.1fx' % (t1 / max(t2, 1e-9)))

# ---------- GQA 规模对比 ----------
print()
print('=== GQA 对 KV Cache 体积的影响（T=256, d=512, n_head=8, 10 层, fp32）===')
hd = cfg.d_model // cfg.n_head
for nkv in (8, 4, 2, 1):
    per = 2 * cfg.n_layer * 256 * nkv * hd * 4
    print('  n_kv_head=%d -> %.2f MB   (KV 维度 %d)' % (nkv, per / 1e6, nkv * hd))
