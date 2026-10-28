"""LoRA training on Tiny-SD's UNet, and a probe that separates forward from backward.

The inference probe gave one number per op class. A training step has a different composition, and the
difference is not uniform: the backward pass recomputes the same convolutions and matmuls for their
inputs and weights, so those classes roughly double; the elementwise classes grow differently because
gradient accumulation touches every intermediate. Measuring only the whole step would hide which part
grew, so forward and backward are timed and profiled separately.

Only the UNet is in the loop. Latents and text embeddings are cached, so a step is exactly one forward,
one backward and one optimizer update on the UNet and nothing else -- which is what makes the profile
readable.

LoRA rather than a full fine-tune, for two reasons that both matter on this machine. The optimiser
state is proportional to trainable parameters, and 323M of them in fp32 AdamW would be 2.6 GB of state
against 7 GB of free RAM. And the point of the run is the shape of the training step, which does not
depend on how many parameters are being updated -- only on how many are being traversed, and LoRA still
traverses all of them in the backward pass.

Targets are the attention projections, which is the standard choice for SD LoRA. Timesteps are sampled
uniformly, noise is added by the standard closed form, and the loss is the usual epsilon prediction.
"""
from __future__ import annotations
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
import statistics
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

MODEL = r'D:\work\models\models\tiny-sd'
CACHE = r'D:\work\bitsandbytes-CPU\tinysd_cache'
STEPS = 6

torch.set_num_threads(6)


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float('nan')


print()
print('  === 加载 UNet ===')
from diffusers import UNet2DConditionModel  # noqa: E402

t0 = time.time()
unet = UNet2DConditionModel.from_pretrained(os.path.join(MODEL, 'unet'))
print('    %.1f s · %.1fM 参数 · RSS %.2f GB'
      % (time.time() - t0, sum(p.numel() for p in unet.parameters()) / 1e6, rss()))

print()
print('  === 加 LoRA ===')
from peft import LoraConfig, get_peft_model  # noqa: E402

cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias='none',
                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'])
unet = get_peft_model(unet, cfg)
tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
tot = sum(p.numel() for p in unet.parameters())
print('    可训练 %.3fM / 总 %.1fM ⇒ %.2f%%' % (tr / 1e6, tot / 1e6, tr / tot * 100))
unet.train()

lat_all = torch.load(os.path.join(CACHE, 'latents.pt')).float()
ctx = torch.load(os.path.join(CACHE, 'text_emb.pt')).float()
print('    latents %s · ctx %s' % (tuple(lat_all.shape), tuple(ctx.shape)))
opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)
gen = torch.Generator().manual_seed(0)

# ★ 用 DDPMScheduler.add_noise,而不是自己推 closed form。
#   我第一版手写了 alpha_bar 的公式,而它在 torch.cos 那里就错了(传了 float)——
#   错的原因不重要,重要的是【那是一个已经被实现、被测试过的公式,不该在这里重写一遍】。
#   而 checkpoint 里那个 scheduler 是 DPMSolverMultistepScheduler(推理用),
#   训练用的噪声日程是 DDPMScheduler,两者的 alpha 表不同。
from diffusers import DDPMScheduler  # noqa: E402

noise_sched = DDPMScheduler(num_train_timesteps=1000)


def one_batch(i):
    lat = lat_all[i % lat_all.shape[0]].unsqueeze(0)
    noise = torch.randn(lat.shape, generator=gen)
    ts = torch.randint(0, 1000, (1,), generator=gen)
    noisy = noise_sched.add_noise(lat, noise, ts)
    return noisy, ts, noise


print()
print('  === 训练 ===')
losses = []
times = []
for i in range(STEPS):
    noisy, ts, noise = one_batch(i)
    t0 = time.perf_counter()
    pred = unet(noisy, ts, ctx).sample
    loss = torch.nn.functional.mse_loss(pred, noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    dt = time.perf_counter() - t0
    times.append(dt)
    losses.append(float(loss.detach()))
    print('    步 %d: loss %.4f · %.2f s' % (i + 1, losses[-1], dt))
print('    中位步时 %.2f s · loss %.4f → %.4f' % (statistics.median(times), losses[0], losses[-1]))

print()
print('  === 探针：前向 vs 反向 ===')
from torch.profiler import profile, ProfilerActivity  # noqa: E402

noisy, ts, noise = one_batch(0)

with torch.no_grad():
    unet(noisy, ts, ctx)
    with profile(activities=[ProfilerActivity.CPU]) as pf:
        unet(noisy, ts, ctx)
ev_f = [e for e in pf.key_averages() if e.self_cpu_time_total > 0]
tf = sum(e.self_cpu_time_total for e in ev_f)

opt.zero_grad(set_to_none=True)
# ★ unet(...) 返回 UNet2DConditionOutput,不是 tensor。训练循环里我用了 .sample,而这里漏了。
#   报错信息是 "'UNet2DConditionOutput' object has no attribute 'size'",指向 mse_loss 内部 ——
#   而真正的问题是这个返回值多包了一层。
unet(noisy, ts, ctx)
with profile(activities=[ProfilerActivity.CPU]) as pb:
    pred = unet(noisy, ts, ctx).sample
    torch.nn.functional.mse_loss(pred, noise).backward()
ev_b = [e for e in pb.key_averages() if e.self_cpu_time_total > 0]
tb = sum(e.self_cpu_time_total for e in ev_b)
opt.zero_grad(set_to_none=True)


def bucket(ev):
    pats = {'卷积': ('conv',), '矩阵乘': ('addmm', 'aten::mm', 'matmul', 'linear'),
            '注意力': ('attention', 'sdpa', 'bmm'), '归一化/激活': ('norm', 'silu', 'gelu', 'softmax'),
            'elementwise/搬运': ('mul', 'add', 'cat', 'upsample', 'interpolate', 'copy', 'to',
                                 'contiguous', 'reshape', 'view', 'permute', 'transpose', 'slice',
                                 'select', 'expand', 'narrow', 'fill', 'zero')}
    acc = {k: 0.0 for k in pats}
    other = 0.0
    for e in ev:
        k = e.key.lower()
        for b, ps in pats.items():
            if any(p in k for p in ps):
                acc[b] += e.self_cpu_time_total
                break
        else:
            other += e.self_cpu_time_total
    return acc, other


bf, of = bucket(ev_f)
bb, ob = bucket(ev_b)
print()
print('  %-18s %14s %8s | %14s %8s' % ('类别', '前向 ms', '占比', '前+反向 ms', '占比'))
print('  ' + '-' * 74)
for k in ('卷积', '矩阵乘', '注意力', '归一化/激活', 'elementwise/搬运'):
    print('  %-18s %14.1f %7.1f%% | %14.1f %7.1f%%'
          % (k, bf[k] / 1000, bf[k] / tf * 100, bb[k] / 1000, bb[k] / tb * 100))
print('  %-18s %14.1f %7.1f%% | %14.1f %7.1f%%'
      % ('其余', of / 1000, of / tf * 100, ob / 1000, ob / tb * 100))
print('  %-18s %14.1f %7s | %14.1f %7s' % ('合计', tf / 1000, '100%', tb / 1000, '100%'))
print()
print('  反向/前向 比值: %.2f×' % (tb / tf))
print('  ⇒ ★ 训练步 ≈ 前向 %.2f s + 反向后向 %.2f s + 优化器'
      % (tf / 1000 / STEPS if False else statistics.median(times) / (1 + tb / tf),
         statistics.median(times) * (tb / tf) / (1 + tb / tf)))
print()
print('  === 与推理探针对照 ===')
print('    推理单步（§7.91）: 卷积 41.6% · 矩阵乘 27.7% · 注意力 15.2% · elementwise 11.8%')
for k in ('卷积', '矩阵乘', '注意力', 'elementwise/搬运'):
    print('    训练 前+反向      : %-8s %5.1f%%  ⇒ %s'
          % (k, bb[k] / tb * 100,
             '份额上升' if bb[k] / tb > bf[k] / tf else '份额下降'))
print()
print('  ⇒ ★【训练步的构成与推理步【不同】,而差异集中在哪一类是本次要回答的】')
