# -*- coding: utf-8 -*-
"""probe_step_breakdown.py —— 一步的时间到底花在哪（墙上时间，不是 profiler 的跨线程求和）。

起因：把模型里全部 62 个 nn.Linear 换成 AVX2 内核，整步【一点没快】（0.978x）。
数值是对的（loss 1e-7 / 梯度 2e-6），所以结论只能是：GEMM 不是瓶颈。
但 torch profiler 说 aten::mm 占 46.9%~54.6% —— 那个数是【跨线程求和】的，
跟墙上占比不是一回事（ab_rmsnorm.py 已经把这条路证伪过一次）。

这里直接量：让 forge 内核统计自己在 mm 里花的墙上时间，跟整步时间比。
  forge 时间占比 ≈ Linear 的 GEMM 占比（因为所有 Linear 都走了 forge）
  1 - 那个占比 = 其他一切（attention / norm / 激活 / 优化器 / 数据搬运）

顺带查出：
  · 每个唯一形状的实际调用次数和总 FLOPs
  · 单形状 torch.mm 的实测 GFLOPS（同样的抢核条件下）
  · 内存/分页文件压力（15.4GB 内存，编码任务同时也在吃）
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
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from scripts.lib import forge_gemm
from scripts.lib import forge_patch
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
FG.set_threads(TH)

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))

print('=' * 94)
print('一步时间拆解（墙上时间）   线程=%d  B=%d' % (TH, B))
print('=' * 94)

# ---- 给 FG.mm 挂计时与形状统计 ----
_raw_mm = FG.mm
STAT = {'ns': 0, 'calls': 0}
SHAPES = Counter()
FLOP = Counter()


def timed_mm(A, Bm, layout='NT'):
    t0 = time.perf_counter()
    C = _raw_mm(A, Bm, layout)
    STAT['ns'] += time.perf_counter() - t0
    STAT['calls'] += 1
    I, J = C.shape
    R = A.shape[0] if layout in ('TN', 'TT') else A.shape[1]
    SHAPES[(layout, I, J, R)] += 1
    FLOP[(layout, I, J, R)] += 2.0 * I * J * R
    return C


FG.mm = timed_mm

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
torch.manual_seed(7)
tok = torch.randint(0, cfg.vocab_size, (B, T))

m = SmallARImageModelV2(cfg)
m.load_state_dict(sd)
m.train()
names = forge_patch.enable(m)
print('  模型 %.2fM  替换 %d 个 Linear' % (sum(p.numel() for p in m.parameters()) / 1e6, len(names)))


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
    sum(f.sum() for f in flat(out)).backward()
    m.zero_grad(set_to_none=True)


for _ in range(2):
    step()

STAT['ns'] = 0; STAT['calls'] = 0; SHAPES.clear(); FLOP.clear()
best = 1e30
for _ in range(3):
    t0 = time.perf_counter()
    step()
    best = min(best, time.perf_counter() - t0)

tot = best * 1e3
gms = STAT['ns'] / 3 * 1e3 / 1  # 统计的是 3 步累计，除 3
# 注意 best 只取了最快的一步，但 STAT 累计了 3 步 —— 按均值比
avg = (STAT['ns'] / 3) * 1e3
print()
print('  整步(min-of-3)      : %9.1f ms' % tot)
print('  forge 里花的墙上时间 : %9.1f ms  = %.1f%% （3 步均值）' % (avg, avg / tot * 100))
print('  forge 调用次数       : %d 次/步' % (STAT['calls'] // 3))
tot_flop = sum(FLOP.values()) / 3
print('  forge 总 FLOPs       : %.2f GFLOP/步' % (tot_flop / 1e9))
if avg > 0:
    print('  forge 实测吞吐       : %.1f GFLOPS' % (tot_flop / (STAT['ns'] / 3) / 1e9))
print()
print('  ⇒ 非 GEMM 部分       : %9.1f ms  = %.1f%%' % (tot - avg, (tot - avg) / tot * 100))
print()

print('  ---- 每个唯一形状（每步） ----')
print('  %-5s %-24s %8s %12s' % ('布局', 'I x J x R', '次数', 'GFLOP'))
rows = sorted(FLOP.items(), key=lambda kv: -kv[1])
for k, fl in rows[:16]:
    lay, I, J, R = k
    print('  %-5s %-24s %8d %12.3f' % (lay, '%d x %d x %d' % (I, J, R), SHAPES[k] // 3, fl / 3 / 1e9))

# ---- 同条件下 torch.mm 的实测吞吐 ----
print()
print('  ---- 同一条件下 torch.mm vs forge（把编码抢核的因素一起带上） ----')
print('  %-22s %10s %10s %10s' % ('I x J x R', 'torch', 'forge', 'forge/torch'))
for k, fl in rows[:8]:
    lay, I, J, R = k
    if lay != 'NT':
        continue
    A = torch.randn(I, R)
    Bm = torch.randn(J, R)
    for _ in range(2):
        torch.mm(A, Bm.t())
    bt = 1e30
    for _ in range(3):
        t0 = time.perf_counter(); torch.mm(A, Bm.t()); bt = min(bt, time.perf_counter() - t0)
    bf = 1e30
    for _ in range(3):
        t0 = time.perf_counter(); _raw_mm(A, Bm, 'NT'); bf = min(bf, time.perf_counter() - t0)
    f = 2.0 * I * J * R / 1e9
    print('  %-22s %10.1f %10.1f %10.2fx'
          % ('%d x %d x %d' % (I, J, R), f / bt, f / bf, bt / bf))

print()
try:
    import psutil
    vm = psutil.virtual_memory()
    print('  内存: 总 %.1f GB  可用 %.1f GB  占用 %.0f%%'
          % (vm.total / 1e9, vm.available / 1e9, vm.percent))
except Exception:
    pass
print('DONE')
