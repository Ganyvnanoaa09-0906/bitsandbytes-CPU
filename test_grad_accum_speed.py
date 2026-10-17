"""test_grad_accum_speed.py -- same effective batch, more, smaller steps.

The whole point is speed with quality unchanged, and this is the one lever found so far
that is exactly quality-neutral: splitting a batch into micro-batches and accumulating
gradients computes the same gradient as one large batch, up to floating-point association.
No approximation, no quantisation, no format change. If it is also faster, it is free.

The evidence that it might be comes from measurements already taken:

    batch 2 training          198.5 GFLOPS   (7.22)
    batch 8 training          114 GFLOPS     (measured today, 13.1 s per step)
    the model's matmul shapes standalone at batch 8 only 16% slower than at batch 2
                              235 vs 272 GFLOPS

So the 1.74x gap is not explained by the matmul shapes. The step's non-matmul share --
attention at 18.81%, elementwise at 18.82% including the RMSNorm internals -- scales with
tokens, and per-token overheads grow with micro-batch size as caches are exceeded.

Compares directly: one step at batch 8, against four steps at batch 2 with gradients
accumulated. Same tokens, same gradient. Also reports batch 4 twice, in case the optimum
is in between.

Loss normalisation is made identical: each micro-batch's loss is scaled by its share, so
the accumulated gradient matches the single large-batch gradient. A quick numerical check
confirms that rather than assuming it.
"""
from __future__ import annotations

import dataclasses
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


from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
torch.set_num_threads(6)

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
tok = torch.load(TOKENS, map_location='cpu')
FULL = tok[:8].long()


def build():
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(ck['model'], strict=False)
    m.train()
    return m


def one_step(m, x, scale=1.0, backward=True):
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    loss = torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    if backward:
        (loss * scale).backward()
    return float(loss)


def accum_step(m, micro, n_micro):
    m.zero_grad(set_to_none=True)
    for i in range(n_micro):
        x = FULL[i * micro:(i + 1) * micro]
        # scale so the total gradient matches one batch-8 step
        one_step(m, x, scale=1.0 / n_micro)
    return m


print('=== 数值等价性验证（梯度是否与整批一致 ✓）===')
m1 = build()
m1.zero_grad(set_to_none=True)
one_step(m1, FULL)
g1 = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}
del m1

m2 = build()
accum_step(m2, 2, 4)
g2 = {n: p.grad.clone() for n, p in m2.named_parameters() if p.grad is not None}
del m2

worst = 0.0
worst_name = ''
for n in g1:
    if n not in g2:
        continue
    a, b = g1[n], g2[n]
    d = float((a - b).norm()) / (float(a.norm()) or 1.0)
    if d > worst:
        worst, worst_name = d, n
print('  batch 8 一步 vs batch 2 累积四步：最大梯度相对差 %.3e（%s）%s'
      % (worst, worst_name, '✓ 等价' if worst < 1e-4 else '✗ 不等价'))
print()

print('=== 速度 ===')
print('  %-26s %10s %10s %10s' % ('配置', '毫秒/步', '等效毫秒', '相对'))
print('  ' + '-' * 62)
base = None
for micro, n_micro, label in ((8, 1, 'batch 8 × 1（现状）'),
                              (4, 2, 'batch 4 × 2 累积'),
                              (2, 4, 'batch 2 × 4 累积'),
                              (1, 8, 'batch 1 × 8 累积')):
    m = build()
    # warm up
    accum_step(m, micro, n_micro)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        accum_step(m, micro, n_micro)
        ts.append(time.perf_counter() - t0)
    dt = statistics.median(ts)
    if base is None:
        base = dt
    print('  %-26s %10.1f %10.1f %9.2fx'
          % (label, dt * 1e3, dt * 1e3, base / dt))
    del m
print()
print('判据: 同样 8 个样本的有效 batch，耗时更低者更快，且梯度已验证等价。')
