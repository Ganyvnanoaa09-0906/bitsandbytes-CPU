# -*- coding: utf-8 -*-
"""profile_unet.py — 把 SDXL 的 UNet 前向按模块切开计时，找出 650 秒花在哪。

背景：之前的"4bit 慢 1.64×"结论是错形状造成的误导。
      用扩散的真实形状重测后，4bit/fp32/8bit 基本同速。
      ⇒ 提速必须靠【找出真正的热点】，不能靠猜。

要区分的三类嫌疑：
  a) Linear（attention 的 qkv/o、ff 的两层）—— 已知 ~216 GFLOPS，接近本机混合峰值
  b) **注意力本身**（O(M²)，且 torch 的 CPU sdpa 可能不走优化路径）
  c) conv / GroupNorm（SDXL 的 ResNet 块）
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import gc, os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
torch.set_num_threads(int(os.environ.get('THREADS', '6')))
B = r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9'

from diffusers import UNet2DConditionModel
from diffusers.utils import logging as dlog
dlog.set_verbosity_error()


def replace_linear(module):
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            q = Linear4bit(child.in_features, child.out_features,
                           bias=child.bias is not None, compute_dtype=torch.float32,
                           quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.float(), requires_grad=False,
                                             quant_type='nf4', compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data.float())
            setattr(module, name, q)
            n += 1
        else:
            n += replace_linear(child)
    return n


def upcast(module):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    for m in module.modules():
        if isinstance(m, (Linear4bit, Linear8bitLt)):
            continue
        for p in list(m.parameters(recurse=False)):
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
        for b in list(m.buffers(recurse=False)):
            if b.dtype in (torch.float16, torch.bfloat16):
                b.data = b.data.float()


MODE = os.environ.get('MODE', '4bit')
print('=== 加载 UNet (%s) ===' % MODE, flush=True)
unet = UNet2DConditionModel.from_pretrained(os.path.join(B, 'unet'), torch_dtype=torch.float16,
                                            variant='fp16', use_safetensors=True)
if MODE == '4bit':
    n = replace_linear(unet)
    print('  量化 %d 层' % n, flush=True)
upcast(unet)
gc.collect()
unet.eval()

lat = int(os.environ.get('LAT', '96'))     # 768/8 = 96
x = torch.randn(1, 4, lat, lat)
t = torch.tensor([1.0])
ctx = torch.randn(1, 77, 2048)
add = {'text_embeds': torch.randn(1, 1280), 'time_ids': torch.randn(1, 6)}

print('\n=== 整次前向 ===', flush=True)
with torch.no_grad():
    unet(x, t, ctx, added_cond_kwargs=add)
    t0 = time.perf_counter()
    unet(x, t, ctx, added_cond_kwargs=add)
    total = time.perf_counter() - t0
print('  整次前向 %.2f s  (latent %d×%d)' % (total, lat, lat), flush=True)

# --- 按【模块类】聚合计时 ---
print('\n=== 按模块类聚合（用 forward hook）===', flush=True)
acc = {}
hooks = []


def mk(name):
    def h(mod, inp, out):
        acc.setdefault(name, [0.0, 0])
    return h


class Timed(torch.nn.Module):
    pass


def hook_factory(key):
    def hook(module, inp, out):
        t0 = time.perf_counter()
        # 无法在这里计时（hook 是事后调用）⇒ 改用 forward_pre + forward
    return hook


# 更可靠：forward_pre_hook 记开始时间，forward_hook 记结束
starts = {}


def pre_factory(key):
    def pre(module, inp):
        starts[id(module)] = time.perf_counter()
    return pre


def post_factory(key):
    def post(module, inp, out):
        dt = time.perf_counter() - starts.get(id(module), time.perf_counter())
        a = acc.setdefault(key, [0.0, 0])
        a[0] += dt
        a[1] += 1
    return post


def keyof(m):
    n = type(m).__name__
    if 'Attention' in n:
        return 'Attention(整块)'
    if 'Transformer2D' in n:
        return 'Transformer2D(整块)'
    if 'ResnetBlock' in n or 'ResNet' in n:
        return 'ResnetBlock(整块)'
    if 'GEGLU' in n or 'FeedForward' in n:
        return 'FeedForward(整块)'
    if isinstance(m, torch.nn.Conv2d):
        return 'Conv2d'
    if isinstance(m, torch.nn.GroupNorm):
        return 'GroupNorm'
    return None


for m in unet.modules():
    k = keyof(m)
    if k:
        hooks.append(m.register_forward_pre_hook(pre_factory(k)))
        hooks.append(m.register_forward_hook(post_factory(k)))

with torch.no_grad():
    unet(x, t, ctx, added_cond_kwargs=add)

for h in hooks:
    h.remove()

print('  %-24s %10s %8s %8s' % ('模块类', '累计(s)', '次数', '占比'))
items = sorted(acc.items(), key=lambda kv: -kv[1][0])
for k, (s, c) in items[:12]:
    print('  %-24s %10.2f %8d %7.1f%%' % (k, s, c, 100.0 * s / max(1e-9, total)))
print('  （整块会包含子模块的时间，所以有重叠：看 Attention / Conv2d / GroupNorm 这类叶子更有意义）')
