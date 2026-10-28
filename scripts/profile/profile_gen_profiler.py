# -*- coding: utf-8 -*-
"""profile_gen_profiler.py — 用 torch.profiler 拿【真实】算子级分解

为什么不用微基准外推：
  GQA repeat_interleave 的微基准说占 44%，实际修完只省 12%
  ⇒ 孤立算子的测量不能外推到真实循环（缓存复用/预热状态都不同）

这个脚本直接 profile 真实 generate 循环，按算子聚合。
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
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 88)
print('AR generate 的真实算子分解（torch.profiler）')
print('=' * 88)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\ckpt_12000.pt'
B = int(sys.argv[2]) if len(sys.argv) > 2 else 48
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'])
m.eval()
T = cfg.total_tokens
NP = sum(p.numel() for p in m.parameters())
print('  模型: %.2fM  T=%d  batch=%d' % (NP / 1e6, T, B))
print()

# 先测一次总时长
with torch.no_grad():
    m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
    t0 = time.perf_counter()
    m.generate(None, steps=T, top_k=1024, temperature=1.0, batch=B, seed=1)
    total = time.perf_counter() - t0
print('  每步 %.2f ms  每图 %.2f ms' % (total / T * 1e3, total / B * 1e3))
print()

# profile
from torch.profiler import profile, ProfilerActivity
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        m.generate(None, steps=64, top_k=1024, temperature=1.0, batch=B, seed=1)

print('  --- 按算子类型聚合（前 22）---')
print('  %-44s %10s %10s' % ('算子', '总ms', '占比'))
print('  ' + '-' * 68)
evs = prof.key_averages()
tot = sum(e.self_cpu_time_total for e in evs)
rows = []
for e in evs:
    rows.append((e.self_cpu_time_total, e.key, e.count))
rows.sort(reverse=True)
shown = 0.0
for us, key, cnt in rows[:22]:
    ms = us / 1e3
    shown += us
    k = key if len(key) < 44 else key[:41] + '...'
    print('  %-44s %10.2f %9.1f%%' % (k, ms, 100 * us / tot))
print('  ' + '-' * 68)
print('  %-44s %10.2f %9.1f%%' % ('前 22 项合计', shown / 1e3, 100 * shown / tot))

print()
print('  --- 按类别归并 ---')
cat = {}
for us, key, cnt in rows:
    k = key.lower()
    if 'gemm' in k or 'mm' in k or 'linear' in k or 'addmm' in k or 'matmul' in k:
        c = 'GEMM/matmul'
    elif 'attention' in k or 'sdpa' in k or 'flash' in k or 'bmm' in k:
        c = 'attention'
    elif 'cat' in k or 'copy' in k or 'concat' in k:
        c = 'cat/copy'
    elif 'repeat' in k or 'expand' in k:
        c = 'repeat/expand'
    elif 'norm' in k or 'layer' in k:
        c = 'norm'
    elif 'softmax' in k or 'topk' in k or 'multinomial' in k or 'sort' in k:
        c = 'sampling'
    elif 'silu' in k or 'sigmoid' in k or 'mul' in k or 'add' in k:
        c = 'elementwise'
    elif 'embedding' in k:
        c = 'embedding'
    else:
        c = 'other'
    cat[c] = cat.get(c, 0) + us
print('  %-24s %10s %10s' % ('类别', '总ms', '占比'))
print('  ' + '-' * 46)
for c, us in sorted(cat.items(), key=lambda x: -x[1]):
    print('  %-24s %10.2f %9.1f%%' % (c, us / 1e3, 100 * us / tot))
print('DONE')
