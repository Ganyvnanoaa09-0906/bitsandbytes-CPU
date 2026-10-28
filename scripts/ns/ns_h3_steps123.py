"""H3 round 2, steps 1-3: audit the NS implementation, benchmark it at real shapes, and set
up a numerical acceptance test that actually applies to non-square matrices.

Step 1 -- audit. The implementation is `zeropower_via_newtonschulz5`: quintic iteration with
coefficients (3.4445, -4.7750, 2.0315), five iterations, input normalised by its Frobenius
norm, and any matrix with rows > cols transposed before iterating. Three things to check and
report: the normalisation, the transpose, and the actual shapes it sees in training.

The shapes matter and I had never enumerated them from the live model -- earlier work used a
hand-written list. This builds the model and reads them off.

Step 2 -- benchmark at those shapes, NS = 3, 4, 5, reporting per shape rather than an average
across shapes, plus the full optimiser step.

Step 3 -- numerical acceptance, and here the earlier work was flawed. The check I used was
||X^T X - I|| / sqrt(m), which is meaningless for a tall matrix: X^T X = I is impossible when
m > n, so the residual is large no matter how many iterations are run. That is why 512x256
and 2048x512 showed ~0.74 and ~0.88 at every iteration count and looked unresponsive.

What Muon actually needs is not orthogonality but a well-conditioned update: all singular
values near 1. The naturals measure is therefore the spread of the singular values, which is
well defined for any shape. Both are reported, with the inapplicable one labelled.

No algorithm change is made. The audit is read-only apart from the corrected metric, which is
a measurement, and the old metric is kept in the output for comparison.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import dataclasses
import io
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

torch.set_num_threads(6)

LOG = []
def w(s=''):
    LOG.append(s)
    print(s, flush=True)

# ================================================================ step 1: audit
w('=' * 78)
w('H3 第二轮 · 步骤 1：审计现有 NS 实现')
w('=' * 78)
w()

SRC = os.path.join(r'D:\work\bitsandbytes-CPU', 'ab_muon_v2.py')
txt = io.open(SRC, encoding='utf-8').read()
i = txt.find('def zeropower_via_newtonschulz5')
j = txt.find('\nclass ', i)
w('实现（逐字，来自 %s）:' % os.path.basename(SRC))
w('```python')
w(txt[i:j if j > 0 else i + 900].rstrip())
w('```')
w()
w('审计要点:')
w()
w('| 项 | 现状 | 判断 |')
w('|---|---|---|')
w('| 归一化 | `X / (X.norm() + 1e-7)`，Frobenius 范数 | ✓ 与原文一致：保证初始谱范数 ≤ 1 |')
w('| 矩阵方向 | `X.size(0) > X.size(1)` 时转置 | ✓ 使迭代作用在「高 ≤ 宽」的版本上，'
  '输出再转回 |')
w('| 系数 | (3.4445, −4.7750, 2.0315) | ✓ quintic，原文系数 |')
w('| 迭代次数 | 5，硬编码默认参数 | ← **本实验要测的变量** |')
w('| 后缩放 | `d * max(1, rows/cols)**0.5` | ✓ Muon 原文的更新尺度修正 |')
w('| dtype | `.float()` 强制 fp32 | ✓ 数值稳定 |')
w()

# the real shapes, read from the live model
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
import torch.nn

ck = torch.load(r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt',
                map_location='cpu', weights_only=False)
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
model = SmallARImageModelV2(cfg)

muon_p, adam_p = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (muon_p if (p.ndim == 2 and 'emb' not in n and 'head' not in n) else adam_p).append((n, p))

from collections import Counter  # noqa: E402
shapes = Counter(tuple(p.shape) for _, p in muon_p)
w('**Muon 实际看到的矩阵形状**（从活模型读出，不是手写清单）:')
w()
w('| 形状 | 个数 | 每层 | 说明 |')
w('|---|---|---|---|')
kind = {(512, 512): 'attn Q/O', (512, 256): 'attn K/V（GQA，n_kv_head=4）',
        (512, 2048): 'FFN up/gate', (2048, 512): 'FFN down'}
for (m, n), c in sorted(shapes.items(), key=lambda kv: -kv[1]):
    w('| %d×%d | **%d** | %.1f | %s |'
      % (m, n, c, c / 10.0, kind.get((m, n), '—')))
w('| **合计** | **%d** | — | AdamW 另外管 %d 个（embedding / head / 1-D） |'
  % (len(muon_p), len(adam_p)))
w()
w('**方向统计**：方阵 %d 个 · 高矩阵（m>n）%d 个 · 宽矩阵（m<n）%d 个。'
  % (sum(c for (m, n), c in shapes.items() if m == n),
     sum(c for (m, n), c in shapes.items() if m > n),
     sum(c for (m, n), c in shapes.items() if m < n)))
w('⇒ **转置分支确实会被走到**（%d 个高矩阵），所以判据必须对非方阵成立。'
  % sum(c for (m, n), c in shapes.items() if m > n))
w()

# ================================================================ step 3 metrics
w('=' * 78)
w('H3 第二轮 · 步骤 3：数值判据（先定义，再使用）')
w('=' * 78)
w()
w('**旧判据的问题**：`‖XᵀX − I‖/√m` 在 m > n 时**不可能接近 0**（XᵀX 是 n×n 而秩 ≤ n < m），')
w('因此它对 512×256 与 2048×512 永远给出大值。此前据此判断「迭代数不能减」是**无效推断**。')
w()
w('| 指标 | 定义 | 适用条件 |')
w('|---|---|---|')
w('| **奇异值谱** | `svdvals(X)` 的最大/最小值与相对标准差 | **任意形状** ← 主判据 |')
w('| 谱偏离 | `max|σ − 1|` | 任意形状 ← 主判据 |')
w('| 条件数 | `σ_max / σ_min` | 任意形状 |')
w('| 旧残差 | `‖XᵀX − I‖/√m` | **仅方阵**（保留以便对照） |')
w('| 非有限值 | NaN / Inf 计数 | 任意形状 |')
w()
w('**为什么用奇异值**：Muon 需要的不是严格正交，而是一个**谱平坦的更新方向**'
  '（所有 σ ≈ 1）。这对任意形状都有意义，也直接对应它最终要做的事。')
w()

def ns(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X

def metrics(X):
    m, n = X.shape
    sv = torch.linalg.svdvals(X)
    finite = bool(torch.isfinite(X).all())
    old = None
    if m == n:
        old = float((X @ X.T - torch.eye(m)).norm()) / (m ** 0.5)
    return dict(sv_max=float(sv.max()), sv_min=float(sv.min()),
                dev=float((sv - 1).abs().max()),
                cond=float(sv.max() / max(sv.min(), 1e-12)),
                rel_std=float(sv.std() / max(sv.mean(), 1e-12)),
                old=old, finite=finite)

# ================================================================ step 2 benchmark
w('=' * 78)
w('H3 第二轮 · 步骤 2：真实形状微基准（逐形状，不给跨形状平均值）')
w('=' * 78)
w()

SHAPES = sorted(shapes.items(), key=lambda kv: -(kv[0][0] * kv[0][1]))
w('| 形状 | 个数 | 迭代 | 中位耗时 ms | 波动 % | σ 偏离 | 条件数 | 旧残差 | 有限 |')
w('|---|---|---|---|---|---|---|---|---|')
raw = {}
for (m, n), cnt in SHAPES:
    # a matrix of the real shape, plus its real multiplicity so the totals are meaningful
    G = torch.randn(m, n, generator=torch.Generator().manual_seed(7))
    for steps in (3, 4, 5):
        ns(G, steps)                       # warm
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            X = ns(G, steps)
            ts.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(ts)
        spread = (max(ts) - min(ts)) / med * 100
        mt = metrics(X)
        raw[(m, n, steps)] = (med, cnt, mt)
        w('| %d×%d | %d | **%d** | %.2f | %.1f | %.4f | %.2f | %s | %s |'
          % (m, n, cnt, steps, med, spread, mt['dev'], mt['cond'],
             '%.4f' % mt['old'] if mt['old'] is not None else '不适用',
             '✓' if mt['finite'] else '✗ NaN/Inf'))
w()
w('**按迭代数汇总整个 Muon 参数集（65 个矩阵的真实总耗时）**:')
w()
w('| 迭代 | 全部矩阵总耗时 ms | 相对 5 次 | 最大 σ 偏离 | 最差条件数 |')
w('|---|---|---|---|---|')
tot = {}
for steps in (3, 4, 5):
    tot[steps] = sum(raw[(m, n, steps)][0] * raw[(m, n, steps)][1] for (m, n), _ in SHAPES)
for steps in (3, 4, 5):
    s = tot[steps]
    devs = [raw[(m, n, steps)][2]['dev'] for (m, n), _ in SHAPES]
    conds = [raw[(m, n, steps)][2]['cond'] for (m, n), _ in SHAPES]
    w('| **%d** | **%.1f** | %.3f× | %.4f | %.2f |'
      % (steps, s, s / tot[5], max(devs), max(conds)))
w()
w('⇒ NS 从 5 次减到 3 次，**纯 NS 部分的耗时比为 %.3f×**（%.1f ms → %.1f ms）。'
  % (tot[3] / tot[5], tot[5], tot[3]))
w()

# ================================================================ full optimiser step
w('**完整优化器更新耗时**（含 NS 之外的部分：动量、AdamW、参数写回）:')
w()
try:
    import bitsandbytes
    m2 = SmallARImageModelV2(cfg)
    mp, ap = [], []
    for n_, p in m2.named_parameters():
        if p.requires_grad:
            (mp if (p.ndim == 2 and 'emb' not in n_ and 'head' not in n_) else ap).append(p)
    for p in m2.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 1e-3
    aopt = bnb.optim.AdamW8bit(ap, lr=1e-4)
    buf = [torch.zeros_like(p) for p in mp]
    w('| 阶段 | 中位耗时 ms |')
    w('|---|---|')
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for k, p in enumerate(mp):
            buf[k].mul_(0.95).add_(p.grad)
            d = p.grad.add(buf[k], alpha=0.95)
            if d.ndim == 2:
                d = ns(d, 5) * max(1.0, d.size(0) / d.size(1)) ** 0.5
            p.add_(d, alpha=-0.02)
        ts.append((time.perf_counter() - t0) * 1000)
    t_ns5 = statistics.median(ts)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for k, p in enumerate(mp):
            buf[k].mul_(0.95).add_(p.grad)
            d = p.grad.add(buf[k], alpha=0.95)
            if d.ndim == 2:
                d = ns(d, 3) * max(1.0, d.size(0) / d.size(1)) ** 0.5
            p.add_(d, alpha=-0.02)
        ts.append((time.perf_counter() - t0) * 1000)
    t_ns3 = statistics.median(ts)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        aopt.step()
        ts.append((time.perf_counter() - t0) * 1000)
    t_adam = statistics.median(ts)
    w('| Muon 全部参数（NS=5） | **%.1f** |' % t_ns5)
    w('| Muon 全部参数（NS=3） | **%.1f** |' % t_ns3)
    w('| AdamW8bit（其余参数） | %.1f |' % t_adam)
    w()
    w('⇒ **优化器总耗时**：NS=5 为 %.1f ms，NS=3 为 %.1f ms，**下降 %.1f ms（%.1f%%）**。'
      % (t_ns5 + t_adam, t_ns3 + t_adam, t_ns5 - t_ns3,
         (t_ns5 - t_ns3) / (t_ns5 + t_adam) * 100))
    w()
    step_s = 3.952          # baseline step, baseline.md
    w('**代入完整训练步（基线 %.3f s）**:' % step_s)
    w()
    w('| 口径 | NS=5 | NS=3 | 加速比 |')
    w('|---|---|---|---|')
    w('| 优化器阶段 | %.3f s | %.3f s | %.2f× |'
      % ((t_ns5 + t_adam) / 1000, (t_ns3 + t_adam) / 1000,
         (t_ns5 + t_adam) / (t_ns3 + t_adam)))
    w('| **完整一步** | %.3f s | %.3f s | **%.4f×** |'
      % (step_s, step_s - (t_ns5 - t_ns3) / 1000,
         step_s / (step_s - (t_ns5 - t_ns3) / 1000)))
    w()
    w('⇒ **NS 局部省 %.1f%% 只换来整步的 %.2f%%**。这是理论上限，'
      % ((t_ns5 - t_ns3) / (t_ns5 + t_adam) * 100,
         (1 - (step_s - (t_ns5 - t_ns3) / 1000) / step_s) * 100))
    w('  而且**尚未验证质量是否保持**（步骤 4）。')
except Exception as e:
    w('完整优化器计时失败: %s' % e)
w()

out = r'D:\work\research\artifacts\benchmark_logs\ns_audit_and_microbench.txt'
io.open(out, 'w', encoding='utf-8', newline='').write('\n'.join(LOG))
print()
print('  原始日志: %s' % out)
