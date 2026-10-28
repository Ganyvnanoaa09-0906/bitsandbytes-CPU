# -*- coding: utf-8 -*-
"""probe_train_knobs.py —— 机器空着时，把预训练能拧的旋钮一次量完。

背景（干净测量已经确立的事实）：
  · 整步 2711 ms；加权 GEMM 约 1473 ms(54%)，非 GEMM 约 1240 ms(46%)
  · torch/oneDNN 在这批形状上已经是 185~293 GFLOPS，接近机器实测上限
    （持续 233 / 突发 351 GFLOPS fp32），【GEMM 几乎没有余量】
  · 自己写的 AVX2 内核加权只快 5.8%，端到端反而慢 14%（每步 186 次调用的
    ctypes+OpenMP 建队开销约 2ms/次）
  ⇒ 所以加速预训练不能再盯着 GEMM，要看别的。

这个脚本量四件事（都只看墙钟）：
  ① 前向 / 反向 / 优化器 三段占比
  ② batch 扫描：每图成本随 batch 怎么变（GEMM 的 M 越大效率越高）
  ③ 优化器对照：bnb AdamW8bit vs torch AdamW（fused/非 fused）
  ④ 线程数扫描：复核「6 线程比 5 慢 5.9%（15W 功耗墙）」这个旧结论
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
import os
import sys
import time

import torch
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')

print('=' * 96)
print('预训练旋钮测量')
print('=' * 96)
try:
    import psutil
    print('  CPU 占用 %.0f%%   内存可用 %.1f GB' % (psutil.cpu_percent(interval=1.0),
                                                psutil.virtual_memory().available / 1e9))
except Exception:
    pass

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
print('  模型配置: d=%d L=%d vocab=%d T=%d' % (cfg.d_model, cfg.n_layer, cfg.vocab_size, T))
print()


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


def build(seed=0):
    torch.manual_seed(seed)
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(sd)
    m.train()
    return m


# ================= ① 三段占比 + ② batch 扫描 =================
print('---- ① 前向/反向/优化器 三段，以及 ② batch 扫描 ----')
print('  %-6s %9s %9s %9s %11s %11s %13s'
      % ('batch', '前向ms', '反向ms', '优化ms', '合计ms', 'ms/图', '每图相对B8'))
print('  ' + '-' * 84)

torch.set_num_threads(6)
try:
    import bitsandbytes
    OptCls, oname = bnb.optim.AdamW8bit, 'bnb AdamW8bit'
except Exception:
    OptCls, oname = torch.optim.AdamW, 'torch AdamW'
print('  (优化器用 %s)' % oname)
print()

res = {}
base_per_img = None
for B in (4, 8, 16, 32):
    m = build()
    opt = OptCls(m.parameters(), lr=1e-4)
    tok = torch.randint(0, cfg.vocab_size, (B, T))

    def one():
        t0 = time.perf_counter()
        out = m.forward_logits(tok)
        loss = sum(f.float().sum() for f in flat(out))
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none=True)
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2

    for _ in range(2):
        one()
    n = 2
    fw = bw = op = 0.0
    best = 1e30
    t0 = time.perf_counter()
    for _ in range(n):
        a, b, c = one()
        fw += a; bw += b; op += c
        best = min(best, a + b + c)
    per_img = best / B
    if base_per_img is None:
        base_per_img = per_img
    res[B] = per_img
    print('  %-6d %9.1f %9.1f %9.1f %11.1f %11.2f %12.3fx'
          % (B, fw / n * 1e3, bw / n * 1e3, op / n * 1e3, best * 1e3, per_img * 1e3,
             per_img / base_per_img))

bestB = min(res, key=lambda b: res[b])
print()
print('  每图最省时的 batch = %d  (%.2f ms/图)' % (bestB, res[bestB] * 1e3))
print('  batch=8 是 %.2f ms/图 => batch%d 每图 %s %.2fx'
      % (res[8] * 1e3, bestB, '省' if res[bestB] < res[8] else '费',
         max(res[bestB], res[8]) / min(res[bestB], res[8])))

# ================= ③ 优化器对照 =================
print()
print('---- ③ 优化器对照（B=8，只改优化器） ----')
B = 8
tok = torch.randint(0, cfg.vocab_size, (B, T))
print('  %-26s %11s %10s' % ('优化器', 'ms/步(仅step)', '相对'))
print('  ' + '-' * 52)
opt_ms = {}
cands = []
try:
    import bitsandbytes
    cands.append(('bnb AdamW8bit', lambda ps: bnb.optim.AdamW8bit(ps, lr=1e-4)))
except Exception:
    pass
cands.append(('torch AdamW', lambda ps: torch.optim.AdamW(ps, lr=1e-4)))
try:
    cands.append(('torch AdamW fused', lambda ps: torch.optim.AdamW(ps, lr=1e-4, fused=True)))
except Exception:
    pass

for nm, mk in cands:
    try:
        m = build()
        opt = mk(m.parameters())
        for _ in range(2):
            sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        best = 1e30
        for _ in range(3):
            sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
            t0 = time.perf_counter()
            opt.step()
            best = min(best, time.perf_counter() - t0)
            opt.zero_grad(set_to_none=True)
        opt_ms[nm] = best
    except Exception as ex:
        print('  %-26s 失败: %s' % (nm, str(ex)[:40]))
        continue

bref = opt_ms.get('bnb AdamW8bit') or min(opt_ms.values())
for nm in opt_ms:
    print('  %-26s %11.1f %9.2fx' % (nm, opt_ms[nm] * 1e3, bref / opt_ms[nm]))

# ================= ④ 线程扫描 =================
print()
print('---- ④ 线程数扫描（整步，B=8） ----')
m = build()
opt = OptCls(m.parameters(), lr=1e-4)
tok = torch.randint(0, cfg.vocab_size, (B, T))
print('  %-8s %12s' % ('线程', 'ms/步'))
print('  ' + '-' * 24)
tres = {}
for nt in (4, 5, 6):
    torch.set_num_threads(nt)
    for _ in range(2):
        sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    best = 1e30
    for _ in range(3):
        t0 = time.perf_counter()
        sum(f.float().sum() for f in flat(m.forward_logits(tok))).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        best = min(best, time.perf_counter() - t0)
    tres[nt] = best
    print('  %-8d %12.1f' % (nt, best * 1e3))
bn = min(tres, key=lambda k: tres[k])
print()
print('  最快的线程数 = %d (%.1f ms)   6 线程是 %.1f ms => %s %.2fx'
      % (bn, tres[bn] * 1e3, tres[6] * 1e3, '省' if tres[bn] < tres[6] else '费',
         max(tres[bn], tres[6]) / min(tres[bn], tres[6])))

print()
print('DONE')
