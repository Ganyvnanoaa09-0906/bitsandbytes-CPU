"""Train a real LoRA on Tiny-SD and show whether it changed anything.

The 6-step run proved the loop works and gave the training step's composition. This one trains long
enough to matter and, more importantly, produces the artefact that decides whether it worked: the same
prompts and the same seeds generated before and after, side by side.

A short diffusion LoRA on 48 images will not produce a good model, and claiming otherwise would be the
kind of overstatement this project keeps finding in other people's documentation. What it can show
honestly is whether the weights moved and whether the outputs moved with them. A constant caption means
the adapter is learning style rather than a text-conditioned concept, so the expectation is a shift in
palette and rendering rather than in content.

Runs standalone so it can be backgrounded while other work proceeds. Saves the LoRA weights, the before
and after images, and a loss log with the timestep recorded alongside each loss -- because the timestep
is sampled uniformly, a single step's loss is not comparable to the next one's, and a log without t
invites exactly the wrong reading.
"""
from __future__ import annotations
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

import json
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


MODEL = r'D:\work\models\models\tiny-sd'
CACHE = r'D:\work\bitsandbytes-CPU\tinysd_cache'
OUTDIR = r'D:\work\bitsandbytes-CPU\tinysd_lora'
STEPS = 150
LR = 1e-4
GEN_STEPS = 4

os.makedirs(OUTDIR, exist_ok=True)
torch.set_num_threads(6)

PROMPTS = [
    ('astronaut', 'a photo of an astronaut riding a horse on the moon'),
    ('city', 'a flat vector illustration of a city at night'),
    ('portrait', 'a portrait of a woman, flat vector style'),
    ('landscape', 'a flat vector illustration of mountains and a lake'),
]
SEED = 4242


def log(msg):
    print(msg, flush=True)


log('')
log('  === 加载 ===')
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32,
                                               safety_checker=None, requires_safety_checker=False)
pipe.set_progress_bar_config(disable=True)
lat_all = torch.load(os.path.join(CACHE, 'latents.pt')).float()
ctx = torch.load(os.path.join(CACHE, 'text_emb.pt')).float()
log('    latents %s · ctx %s' % (tuple(lat_all.shape), tuple(ctx.shape)))


def generate(tag):
    """Same prompts, same seed, so any difference is the weights and not the sampling."""
    imgs = {}
    for name, prompt in PROMPTS:
        g = torch.Generator().manual_seed(SEED)
        pipe.scheduler.set_timesteps(GEN_STEPS)
        im = pipe(prompt, num_inference_steps=GEN_STEPS, guidance_scale=7.5,
                  height=512, width=512, generator=g).images[0]
        p = os.path.join(OUTDIR, '%s_%s.png' % (tag, name))
        im.save(p)
        imgs[name] = p
    return imgs


log('')
log('  === 训练前出图 ===')
t0 = time.time()
before = generate('before')
log('    4 张 · %.1f s' % (time.time() - t0))

log('')
log('  === 加 LoRA 并训练 ===')
unet = pipe.unet
for p in unet.parameters():
    p.requires_grad = False
cfg = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias='none',
                 target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'])
unet = get_peft_model(unet, cfg)
pipe.unet = unet
tr = sum(p.numel() for p in unet.parameters() if p.requires_grad)
log('    可训练 %.3fM / 总 %.1fM' % (tr / 1e6, sum(p.numel() for p in unet.parameters()) / 1e6))
unet.train()

opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=LR)
sched = DDPMScheduler(num_train_timesteps=1000)
gen = torch.Generator().manual_seed(0)

w0 = None
for n, p in unet.named_parameters():
    if p.requires_grad and w0 is None:
        w0 = p.detach().clone()
        log('    记录初始权重: %s' % n)
        break

log('')
log('  === 训练 %d 步 ===' % STEPS)
t0 = time.time()
log_losses = []
times = []
for i in range(STEPS):
    lat = lat_all[i % lat_all.shape[0]].unsqueeze(0)
    noise = torch.randn(lat.shape, generator=gen)
    ts = torch.randint(0, 1000, (1,), generator=gen)
    noisy = sched.add_noise(lat, noise, ts)
    t1 = time.perf_counter()
    pred = unet(noisy, ts, ctx).sample
    loss = torch.nn.functional.mse_loss(pred, noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    dt = time.perf_counter() - t1
    times.append(dt)
    log_losses.append((i, float(loss.detach()), int(ts[0])))
    if (i + 1) % 25 == 0:
        recent = [l for _, l, _ in log_losses[-25:]]
        el = time.time() - t0
        log('    步 %3d/%d · loss(近25步均值) %.4f · 已 %.0f s · 预计总 %.0f s'
            % (i + 1, STEPS, statistics.mean(recent), el, el / (i + 1) * STEPS))
el = time.time() - t0
med = statistics.median(times)
log('    完成: %.1f s · 中位步时 %.2f s · 每步 %.1f MB/s' % (el, med, 0))

w1 = None
for n, p in unet.named_parameters():
    if p.requires_grad:
        w1 = p.detach()
        break
delta = (w1 - w0).abs().mean().item() if w0 is not None else float('nan')
log('    ★ 权重实际变化量 (mean|Δ|): %.3e' % delta)

log('')
log('  === 保存 ===')
lora_dir = os.path.join(OUTDIR, 'lora')
unet.save_pretrained(lora_dir)
log('    LoRA → %s' % lora_dir)
with open(os.path.join(OUTDIR, 'loss_log.json'), 'w', encoding='utf-8') as fh:
    json.dump({'steps': STEPS, 'lr': LR, 'median_step_s': med, 'total_s': el,
               'weight_delta': delta, 'log': log_losses}, fh, indent=1)
log('    日志 → loss_log.json（含每步的 timestep,t 是均匀采样的 ⇒ 单步 loss 不可比）')

log('')
log('  === 训练后出图 ===')
unet.eval()
t0 = time.time()
after = generate('after')
log('    4 张 · %.1f s' % (time.time() - t0))

log('')
log('  === 差异（像素级）===')
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

for name, _ in PROMPTS:
    a = np.asarray(Image.open(before[name]).convert('RGB'), dtype=np.float32)
    b = np.asarray(Image.open(after[name]).convert('RGB'), dtype=np.float32)
    d = np.abs(a - b)
    log('    %-10s 平均差 %6.2f · 最大差 %6.1f · 变化像素占比 %5.1f%%'
        % (name, d.mean(), d.max(), (d.max(axis=2) > 8).mean() * 100))

log('')
log('  产物:')
for name, _ in PROMPTS:
    log('    %s / %s' % (os.path.basename(before[name]), os.path.basename(after[name])))
log('')
log('  ★ 如实预期:48 张图 · 恒定 caption · %d 步 ⇒ 这学的是【风格倾向】,不是概念。' % STEPS)
log('    能问的不是"好不好",而是【权重动了没有、输出跟着动了没有】。')
log('')
