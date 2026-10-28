"""profile_step_batch8.py -- find the non-matmul half of a batch-8 training step.

Established by direct measurement:

    the model's own matmul shapes run at 207-256 GFLOPS standalone
    the whole step runs at 114 GFLOPS (1.5 TFLOP in 13.1 s at batch 8)
    batch 2 sustains 198.5, and its matmul shapes are only 16% faster than batch 8's

So the matmuls are not the shortfall. The step's non-matmul share must be far larger at
batch 8 than at batch 2 -- attention, normalisation, activations, the loss, and the
elementwise work around residual adds all scale with batch*sequence while the matmul
efficiency barely moves.

An earlier profile of this model put aten::mm at 55.78%, flash attention at 18.1% total,
and aten::mul at 10.88% across 2258 calls, but that was taken under a different
configuration and cannot be assumed to hold here. Re-measure rather than reuse.

Reported: the top operators by self time for one forward+backward, and the split by
category so the size of the non-matmul share is explicit.
"""
from __future__ import annotations
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

import dataclasses
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from scripts.lib.small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2
from scripts.lib.loop_transformer import LoopConfig

CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
torch.set_num_threads(6)

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
m = SmallARImageModelV2(cfg)
m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                      loop_L=1, loop_L_max=3, loop_random=False, loop_cache='shared'))
m.load_state_dict(ck['model'], strict=False)
m.train()

tok = torch.load(TOKENS, map_location='cpu')
x = tok[:8].long()

# warm up so the first-call costs do not dominate
for _ in range(2):
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()
    m.zero_grad(set_to_none=True)

from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()

evts = prof.key_averages()
total = sum(e.self_device_time_total + e.self_cpu_time_total for e in evts)
rows = sorted(evts, key=lambda e: -(e.self_device_time_total + e.self_cpu_time_total))
print('batch 8, T=1024，一次前向+反向')
print('%-42s %12s %8s %10s' % ('算子', '自身时间 ms', '占比', '调用次数'))
print('-' * 78)
shown = 0
cats = {}
for e in rows:
    t = e.self_device_time_total + e.self_cpu_time_total
    if t <= 0:
        continue
    if shown < 24:
        print('%-42s %12.1f %7.2f%% %10d'
              % (e.key[:42], t / 1000.0, 100.0 * t / max(total, 1), e.count))
        shown += 1
    k = e.key
    if 'mm' in k or 'addmm' in k or 'bmm' in k or 'linear' in k.lower():
        c = 'matmul'
    elif 'flash' in k or 'attention' in k or 'sdpa' in k:
        c = 'attention'
    elif 'norm' in k.lower():
        c = 'norm'
    elif any(s in k for s in ('mul', 'div', 'add', 'sub', 'exp', 'silu', 'gelu',
                              'sigmoid', 'tanh', 'clamp', 'copy', 'cat', 'reshape',
                              'view', 'transpose', 'permute', 'contiguous')):
        c = 'elementwise/reshape'
    elif 'cross_entropy' in k or 'log_softmax' in k or 'nll' in k:
        c = 'loss'
    else:
        c = 'other'
    cats[c] = cats.get(c, 0) + t

print()
print('=== 按类别 ===')
for c, t in sorted(cats.items(), key=lambda kv: -kv[1]):
    print('  %-22s %10.1f ms  %6.2f%%' % (c, t / 1000.0, 100.0 * t / max(total, 1)))
print('  %-22s %10.1f ms' % ('合计', total / 1000.0))
print()
print('注: 这些是 CPU 自身时间（self time），已按算子去重；')
print('    mm 若占多数而整体只有 114 GFLOPS，则差距在其余算子上。')
