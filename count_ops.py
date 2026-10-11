# -*- coding: utf-8 -*-
"""count_ops.py —— 只数【算子调用次数】，不看耗时。

为什么只数次数：抢核会让耗时失真，但"这一步派发了多少个算子"是确定的。
判据很简单 —— CPU 上每个算子都要 fork-join 线程池，按 6 线程约 0.3ms/次算，
2400 次 ≈ 720ms，跟实测的非 GEMM 1122ms 同量级。所以要提速就得减少算子数。

★ 融合 RMSNorm 之后再数一次，看剩下的簇在哪。
"""
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

import small_image_model_v2 as M
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(int(os.environ.get('TH', '6')))
CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck.get('model') or ck.get('state_dict') or ck)
m.train()
T = cfg.total_tokens
print('融合 RMSNorm = %s' % M._RMS_FUSED)
print('模型 %.2fM  d=%d L=%d T=%d B=%d' % (sum(p.numel() for p in m.parameters()) / 1e6,
                                          cfg.d_model, cfg.n_layer, T, B))

tok = torch.randint(0, cfg.vocab_size, (B, T))


def flat(o):
    r = []
    if o is None:
        return r
    if isinstance(o, (tuple, list)):
        for i in o:
            r += flat(i)
        return r
    if torch.is_tensor(o):
        r.append(o)
    return r


def step():
    out = m.forward_logits(tok)
    sum(f.float().sum() for f in flat(out)).backward()


step()

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()

cnt = Counter()
for e in prof.events():
    nm = getattr(e, 'name', None)
    if nm:
        cnt[nm] += 1

total = sum(cnt.values())
print()
print('  一步派发的算子总数: %d' % total)
print()
print('  %-58s %7s' % ('算子', '次数'))
print('  ' + '-' * 68)
for nm, c in cnt.most_common(35):
    print('  %-58s %7d' % (nm[:58], c))

# 按大类归并，看融合还有多大空间
FUSE_HINT = {
    'norm': ('rms_norm', 'layer_norm', 'native_norm'),
    'elementwise': ('mul', 'add', 'div', 'sub', 'pow', 'sqrt', 'rsqrt', 'neg', 'abs', 'clamp'),
    'reduction': ('sum', 'mean', 'max', 'min', 'norm.'),
    'activation': ('silu', 'gelu', 'relu', 'sigmoid', 'tanh'),
    'matmul': ('mm', 'addmm', 'bmm', 'matmul', 'linear'),
    'copy/alloc': ('copy_', 'fill_', 'zero_', 'clone', 'empty', 'resize', 'cat', 'stack'),
    'view/meta': ('view', 'reshape', 'permute', 'transpose', 'expand', 'contiguous', 'detach',
                  'slice', 'select', 'narrow', 'as_strided', 'split', 'chunk', 'unbind'),
}
print()
print('  ---- 按大类归并 ----')
print('  %-14s %8s' % ('类别', '次数'))
print('  ' + '-' * 26)
buckets = Counter()
for nm, c in cnt.items():
    short = nm.split('::')[-1].split('.')[-1]
    hit = None
    for cat, keys in FUSE_HINT.items():
        if any(k in nm for k in keys):
            hit = cat
            break
    buckets[hit or ('其它: ' + nm.split('::')[-1])] += c
for cat, c in buckets.most_common(16):
    print('  %-14s %8d' % (str(cat)[:14], c))

# 只统计"会产生实际计算/访存"的算子（排除纯元数据操作）
META = ('view', 'reshape', 'permute', 'transpose', 'expand', 'detach', 'slice', 'select',
        'narrow', 'as_strided', 'split', 'chunk', 'unbind', 'size', 'dim', 'numel',
        'is_contiguous', 'stride', 'get_device', 'item', 'set_')
real = sum(c for nm, c in cnt.items() if not any(k in nm for k in META))
print()
print('  会真正访存的算子: %d 次' % real)
print('  纯元数据算子    : %d 次' % (total - real))
print()
print('DONE')
