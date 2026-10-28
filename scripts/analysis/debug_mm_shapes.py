# -*- coding: utf-8 -*-
"""debug_mm_shapes.py —— 把一步里【所有】mm/addmm/bmm 的形状原样倒出来。

为什么单独写：profile_ar_ops.py 报出「186 次 aten::mm，但只有 40 次能配出形状」，
剩下 146 次被我的提取逻辑漏掉了。而「GEMM 总共多少 FLOPs」正是判断
「值不值得把 AVX2 内核接进预训练」的核心数字，不能含糊。

这里不做任何猜测性的解析，直接把 name + input_shapes 原样打印并计数。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
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
print('模型 %.2fM  T=%d  B=%d  d=%d  L=%d' % (sum(p.numel() for p in m.parameters()) / 1e6, T, B,
                                              cfg.d_model, cfg.n_layer))

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
    loss = sum(flat(out)[i].sum() for i in range(len(flat(out))))
    loss.backward()


step()   # 预热

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    step()

cnt = Counter()
flops = {}
for e in prof.events():
    nm = getattr(e, 'name', None)
    if not nm:
        continue
    if not any(k in nm for k in ('mm', 'addmm', 'bmm', 'linear', 'matmul', 'einsum')):
        continue
    sh = getattr(e, 'input_shapes', None)
    key = (nm, tuple(tuple(s) if s else () for s in (sh or [])))
    cnt[key] += 1

print()
print('=' * 104)
print('  所有矩阵乘事件（原样）')
print('=' * 104)
tot = 0.0
rows = []
for (nm, shp), c in cnt.items():
    dims = [list(s) for s in shp if s]
    fl = 0.0
    desc = ''
    if len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
        M, K = dims[0]
        N2, K2 = dims[1]
        desc = '%d x %d x %d' % (M, N2, K)
        if K == K2:
            fl = 2.0 * M * N2 * K
    elif len(dims) >= 2 and len(dims[0]) == 3 and len(dims[1]) == 3:
        b, M, K = dims[0]
        _, N2, K2 = dims[1]
        desc = 'b%d %d x %d x %d' % (b, M, N2, K)
        if K == K2:
            fl = 2.0 * b * M * N2 * K
    else:
        desc = ' / '.join(str(s) for s in dims)
    tot += fl * c
    rows.append((fl * c, nm, c, desc, shp))

rows.sort(key=lambda r: -r[0])
print('  %-26s %6s %-34s %12s' % ('算子', '次数', '形状 (M x N x K)', 'GFLOP'))
print('  ' + '-' * 88)
for g, nm, c, desc, shp in rows:
    print('  %-26s %6d %-34s %12.3f' % (nm[:26], c, desc[:34], g / 1e9))

print('  ' + '-' * 88)
print('  %-26s %6d %-34s %12.3f' % ('合计', sum(r[2] for r in rows), '', tot / 1e9))
print()
print('  ---- 无法解析出 FLOPs 的事件（形状原样） ----')
for g, nm, c, desc, shp in rows:
    if g == 0:
        print('    %-24s x%-4d %s' % (nm[:24], c, shp))
print('DONE')
