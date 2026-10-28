"""Tiny-SD on CPU: does it load, does it generate, and where does the time go.

The second probe target, chosen because it is a different workload class from the NF4 LLM. That one was
matmul-dominated -- 71% dense GEMM and 22% weight dequantisation in prefill, 74% fused 4-bit kernel in
decode. A diffusion model is convolution-dominated, and bitsandbytes quantises no convolutions, so the
op mix should look nothing like it. If the CPU/iGPU split is going to be a general statement rather
than an LLM statement, it needs a second, unlike workload.

Tiny-SD is the distilled SD1.5: three UNet blocks instead of four and one layer per block instead of
two, so a quarter of the parameters. The checkpoint is SG161222/Realistic_Vision_V4.0's pipeline written
by diffusers 0.19.0.dev0 while 0.40.0 is installed, so an API break is a real possibility and the first
job is to find out.

Also carried over from the previous probe: the reference images ship with the checkpoint
(grid_tiny.png, val_imgs_grid.png), so the output can be compared against the model's own samples
rather than merely eyeballed.

Measured: load time and RSS, wall time per step, and a profiler breakdown of one denoising step. The
step breakdown is the part that matters -- the sampling loop is the whole cost of inference, and its
composition decides whether the cache-blocking result from 7.89 has anything to bite on.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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
OUT = r'D:\work\bitsandbytes-CPU\tinysd_out.png'


def rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float('nan')


torch.set_num_threads(6)
print()
print('  === 加载管线 ===')
from diffusers import StableDiffusionPipeline  # noqa: E402

t0 = time.time()
try:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                   safety_checker=None, requires_safety_checker=False)
except Exception as e:
    print('  ✗ from_pretrained 失败: %s: %s' % (type(e).__name__, str(e)[:200]))
    raise SystemExit(1)
print('    %.1f s · RSS %.2f GB' % (time.time() - t0, rss()))
pipe = pipe.to('cpu')
pipe.set_progress_bar_config(disable=True)

n_par = sum(p.numel() for p in pipe.unet.parameters())
print('    UNet %.1fM 参数 · VAE %.1fM · text_encoder %.1fM'
      % (n_par / 1e6,
         sum(p.numel() for p in pipe.vae.parameters()) / 1e6,
         sum(p.numel() for p in pipe.text_encoder.parameters()) / 1e6))

print()
print('  === 出图 ===')
PROMPT = 'a photo of an astronaut riding a horse on the moon'
gen = torch.Generator().manual_seed(1234)
for steps in (4, 8):
    pipe.scheduler.set_timesteps(steps)
    t0 = time.time()
    img = pipe(PROMPT, num_inference_steps=steps, guidance_scale=7.5,
               height=512, width=512, generator=gen).images[0]
    dt = time.time() - t0
    print('    %d 步: %.1f s ⇒ %.1f s/步' % (steps, dt, dt / steps))
img.save(OUT)
print('    已存 %s（%s）' % (OUT, img.size))

print()
print('  === profile：单步去噪 ===')
from torch.profiler import profile, ProfilerActivity  # noqa: E402

lat = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(0))
# ★ UNet2DConditionModel 的第二个参数是【标量 timestep】,不是预计算的时间嵌入。
#   传一个 [1, 320] 张量会得到 "expand(torch.FloatTensor{[1, 320]}, size=[1])" ——
#   错误信息说的是 expand,而真正的问题是这个参数的身份。
ts = torch.tensor(500.0)
ctx = torch.randn(1, 77, 768)
with torch.no_grad():
    for _ in range(2):
        pipe.unet(lat, ts, ctx)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            pipe.unet(lat, ts, ctx)

ev = [e for e in prof.key_averages() if e.self_cpu_time_total > 0]
ev.sort(key=lambda e: -e.self_cpu_time_total)
tot = sum(e.self_cpu_time_total for e in ev)
print()
print('  %-48s %12s %8s' % ('算子', '自耗时 ms', '占比'))
print('  ' + '-' * 72)
for e in ev[:18]:
    print('  %-48s %12.1f %7.1f%%' % (e.key[:48], e.self_cpu_time_total / 1000,
                                      e.self_cpu_time_total / tot * 100))
print('  %-48s %12.1f' % ('合计（3 次）', tot / 1000))

print()
print('  === 按类别归并 ===')
buckets = {'卷积': ('conv', 'convolution', 'cudnn', 'mkldnn'), '注意力': ('attention', 'sdpa', 'bmm'),
           '矩阵乘': ('addmm', 'aten::mm', 'matmul', 'linear'), '归一化/激活': ('norm', 'group_norm',
           'silu', 'gelu', 'softmax'), 'elementwise/搬运': ('mul', 'add', 'cat', 'upsample',
           'interpolate', 'copy', 'to', 'contiguous', 'reshape', 'view', 'permute', 'transpose')}
acc = {k: 0.0 for k in buckets}
other = 0.0
for e in ev:
    k = e.key.lower()
    hit = None
    for b, pats in buckets.items():
        if any(p in k for p in pats):
            hit = b
            break
    if hit:
        acc[hit] += e.self_cpu_time_total
    else:
        other += e.self_cpu_time_total
for b, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print('  %-16s %10.1f ms %7.1f%%' % (b, v / 1000, v / tot * 100))
print('  %-16s %10.1f ms %7.1f%%' % ('其余', other / 1000, other / tot * 100))
print()
print('  ⇒ 若【卷积】占大头 ⇒ 与 LLM 那个探针(matmul 主导)是【不同的负载】')
print('  ⇒ 而 bnb 不量化卷积 ⇒ 这类负载上我们的库侧工作【用不上】,杠杆在别处')
