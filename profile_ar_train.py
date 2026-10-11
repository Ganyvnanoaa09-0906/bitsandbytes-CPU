# -*- coding: utf-8 -*-
"""profile_ar_train.py — 拆 AR 训练步，找可优化点

当前: 30.49M, T=256, batch 8, 1.31 s/step (7线程 ARM)
关注:
  ① 前向/反向/优化器 各占多少
  ② 有没有可以做的：更大 batch（每图成本几乎不变）、梯度累积、融合算子
  ③ 数据加载有没有成为瓶颈
"""
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
print('=' * 84)
print('AR 训练步拆解  线程=%d' % TH)
print('=' * 84)

# 用本地 checkpoint 的配置
ck = torch.load(r'D:\work\bitsandbytes-CPU\.autopilot\v3_noloop\ckpt_12000.pt',
                map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
m.load_state_dict(ck['model'])
m.train()
NP = sum(p.numel() for p in m.parameters())
T = cfg.total_tokens
print('  模型 %.2fM  T=%d  d=%d L=%d' % (NP / 1e6, T, cfg.d_model, cfg.n_layer))
print()

try:
    import bitsandbytes as bnb
    Opt = bnb.optim.AdamW8bit
    print('  优化器: bnb AdamW8bit')
except Exception:
    Opt = torch.optim.AdamW
    print('  优化器: torch AdamW')
opt = Opt(m.parameters(), lr=1e-4)


def step(B):
    tok = torch.randint(0, cfg.vocab_size, (B, T))
    t0 = time.perf_counter()
    tl, gl = m.forward_logits(tok)
    gl = gl[0] if isinstance(gl, (tuple, list)) else gl
    if m.use_bos:
        loss = F.cross_entropy(gl.reshape(-1, gl.shape[-1]), tok.reshape(-1))
    else:
        loss = F.cross_entropy(gl[:, :-1].reshape(-1, gl.shape[-1]), tok[:, 1:].reshape(-1))
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    opt.step(); opt.zero_grad(set_to_none=True)
    t3 = time.perf_counter()
    return (t1 - t0), (t2 - t1), (t3 - t2), float(loss)


print('  %-8s %10s %10s %10s %10s %12s %12s' % (
    'batch', '前向ms', '反向ms', '优化ms', '合计ms', 'ms/图', 's/1000步'))
print('  ' + '-' * 84)
res = {}
for B in (4, 8, 16, 24, 32):
    for _ in range(2):
        step(B)
    n = 4
    fw = bw = op = 0.0
    t0 = time.perf_counter()
    for _ in range(n):
        a, b, c, L = step(B)
        fw += a; bw += b; op += c
    dt = (time.perf_counter() - t0) / n
    ms = 1e3
    res[B] = dt
    print('  %-8d %10.1f %10.1f %10.1f %10.1f %12.2f %12.1f' % (
        B, fw / n * ms, bw / n * ms, op / n * ms, dt * ms, dt / B * ms, dt * 1000))

print()
print('  判读:')
b1 = res[8]
best = min(res, key=lambda B: res[B] / B)
print('    每图最省时的 batch = %d  (%.2f ms/图)' % (best, res[best] / best * 1e3))
print('    batch=8  是 %.2f ms/图' % (res[8] / 8 * 1e3))
print('    ⇒ 相对 batch8，batch%d 每图 %s%.2fx' % (
    best, '省' if res[best] / best < res[8] / 8 else '费',
    max(res[best] / best, res[8] / 8) / min(res[best] / best, res[8] / 8)))
print()
print('    大 batch 的另一个好处: 梯度噪声按 sqrt(B) 下降')
for B in (8, 16, 32):
    print('      batch=%-3d 噪声相对 batch8: %.2fx' % (B, (8.0 / B) ** 0.5))
print('DONE')
