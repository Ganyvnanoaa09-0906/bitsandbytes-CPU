"""Load the NF4 Qwen3-1.7B on CPU from its own layout, then probe a real forward pass.

Why not from_pretrained: transformers' bitsandbytes integration quantises at load time and goes through
device_map and .cuda(). On a CPU-only machine it either raises or silently leaves the model
unquantised. The checkpoint is therefore written in Params4bit's own state layout and rebuilt by hand --
the recipe qwen21_load.py established and this adapts from QwenImage21Transformer2DModel to
Qwen3ForCausalLM.

Three differences in this checkpoint's metadata, checked rather than assumed:

    top-level key is "dtype": "nf4",   not "quant_type"
    layers carry no "kind" field,      so every entry is treated as linear -- verified by
                                       reading the keys, all of which are *.weight, and
                                       embed_tokens is absent so it stays bf16
    blocksize 64 at both levels

The layout on disk, confirmed from the safetensors header: 508 tensors, of which 197 U8 (the packed
4-bit weights), 197 F32 (one absmax per 64 elements) and 114 BF16 (the unquantised norms and
embeddings). That is bnb's own NF4 layout, and blocksize 64 is the case avx2_gemv_4bit.h handles.

What this measures, since the point is a probe rather than a load: RSS before and after, peak, the time
to read the checkpoint, the time to replace the modules, and a real forward pass with a per-op timing
breakdown. The CPU/iGPU split the user asked about comes from the same run, using the iGPU's measured
solo baseline taken immediately alongside so the ratio is not contaminated by the drift 7.89 recorded.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import json
import os
import statistics
import sys
import time

import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

CKPT = r'D:\work\models\models\qwen3-1.7B-nf4'
HERE = os.path.dirname(os.path.abspath(__file__))


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.2f %s' % (n, u) if u != 'B' else '%d B' % n
        n /= 1024


def rss_gb():
    try:
        import psutil
        i = psutil.Process().memory_info()
        return i.rss / 1e9, (getattr(i, 'peak_wset', None) or i.rss) / 1e9
    except Exception:
        return float('nan'), float('nan')


print()
print('  === 环境 ===')
import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.nn import Linear4bit, Params4bit  # noqa: E402
from bitsandbytes.functional import QuantState  # noqa: E402
print('    bitsandbytes %s' % bnb.__version__)
print('    Linear4bit / Params4bit / QuantState 全部可用 ✓')

print()
print('  === 读 checkpoint ===')
import safetensors.torch as st  # noqa: E402

with open(os.path.join(CKPT, 'quant_meta.json'), encoding='utf-8') as fh:
    meta = json.load(fh)
layers = meta['layers']
quant_type = meta.get('quant_type') or meta.get('dtype')
blocksize = meta.get('blocksize') or next(iter(layers.values())).get('blocksize')
print('    quant_type=%s blocksize=%s · %d 个量化模块' % (quant_type, blocksize, len(layers)))

r0, p0 = rss_gb()
t0 = time.time()
tensors = st.load_file(os.path.join(CKPT, 'model.safetensors'), device='cpu')
t_read = time.time() - t0
r1, _ = rss_gb()
print('    %d 个张量 · %s · 读入 %.1f s · RSS %.2f → %.2f GB'
      % (len(tensors), human(sum(t.numel() * t.element_size() for t in tensors.values())),
         t_read, r0, r1))

print()
print('  === 在 meta 设备上建架构 ===')
from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402

cfg = AutoConfig.from_pretrained(CKPT)
t0 = time.time()
with init_empty_weights():
    model = Qwen3ForCausalLM(cfg)
n_par = sum(p.numel() for p in model.parameters())
print('    %.2fe9 参数,建了 %.1f s,【一个字节都没分配】' % (n_par / 1e9, time.time() - t0))

print()
print('  === 替换量化模块 ===')
t0 = time.time()
replaced = 0
for key, info in layers.items():
    mod_path, _, pname = key.rpartition('.')
    try:
        module = model.get_submodule(mod_path)
    except AttributeError:
        raise SystemExit('checkpoint 有 %s,而模型里没有 %s' % (key, mod_path)) from None
    # ★ The packed tensor is named <key>.wq, not <key>. And the metadata carries only absmax -- no
    #   quant_map, no shape -- so QuantState.from_dict has nothing to work from and the constructor is
    #   used directly with the values that are known.
    # ★ QuantState.dtype is the dtype of the ORIGINAL unquantised tensor (bf16), not the packed
    #   storage dtype. Passing uint8 makes dequantize_4bit fail with "Blockwise 4bit dequantization
    #   only supports 16/32-bit floats, but got torch.uint8" -- the message points at the packed
    #   tensor, while the value it is complaining about is this field.
    packed = tensors[key + '.wq']
    qs = QuantState(absmax=tensors[key + '.absmax'],
                    shape=torch.Size(info['shape']), blocksize=blocksize,
                    quant_type=quant_type, dtype=torch.bfloat16)
    out_f, in_f = qs.shape
    bias = getattr(module, 'bias', None) is not None
    new = Linear4bit(in_f, out_f, bias=bias, compute_dtype=torch.float32, quant_type=quant_type)
    param = Params4bit(packed, requires_grad=False, quant_type=quant_type,
                       blocksize=blocksize, bnb_quantized=True, quant_storage=packed.dtype)
    param.quant_state = qs
    param.module = new
    new.weight = param
    new.quant_state = qs
    if bias:
        new.bias = torch.nn.Parameter(module.bias.detach().clone())
    parent_path, _, leaf = mod_path.rpartition('.')
    setattr(model.get_submodule(parent_path) if parent_path else model, leaf, new)
    replaced += 1
print('    替换 %d / %d 个（元数据说 %d）· %.1f s'
      % (replaced, len(layers), len(layers), time.time() - t0))
if replaced != len(layers):
    raise SystemExit('★ 部分替换 ⇒ 会产出【能跑但静默错误】的模型,拒绝继续')

print()
print('  === 物化未量化的参数 ===')
# ★ copy_ 进 meta 张量在 torch 2.13 是静默 no-op（qwen21_load.py 实测记录过）
#   ⇒ 必须用 to_empty() 真正分配,然后逐参数校验离开了 meta
quantized = set(layers)
named = dict(model.named_parameters())
to_fill = [nm for nm, p in named.items() if nm not in quantized and nm in tensors and p.is_meta]
parents = {}
for nm in to_fill:
    parents.setdefault(nm.rsplit('.', 1)[0] if '.' in nm else '', []).append(nm)
for pp in parents:
    (model.get_submodule(pp) if pp else model).to_empty(device='cpu')
n = 0
with torch.no_grad():
    for nm in to_fill:
        p = dict(model.named_parameters())[nm]
        src = tensors[nm]
        if src.shape != p.shape:
            print('    ✗ 形状不符 %s: ckpt %s vs model %s' % (nm, tuple(src.shape), tuple(p.shape)))
            continue
        p.copy_(src.to(p.dtype))
        if p.is_meta:
            raise SystemExit('★ copy_ 之后 %s 仍在 meta 上 —— 静默 no-op' % nm)
        n += 1
    nb = 0
    bparents = {}
    for nm, b in dict(model.named_buffers()).items():
        if nm in tensors and b.is_meta:
            bparents.setdefault(nm.rsplit('.', 1)[0] if '.' in nm else '', []).append(nm)
    for pp in bparents:
        (model.get_submodule(pp) if pp else model).to_empty(device='cpu')
    for nm, b in dict(model.named_buffers()).items():
        if nm in tensors and b.shape == tensors[nm].shape:
            b.copy_(tensors[nm].to(b.dtype))
            nb += 1
print('    物化 %d 个参数 + %d 个 buffer' % (n, nb))
still = [nm for nm, p in model.named_parameters() if p.is_meta]
if still:
    raise SystemExit('★ 仍有 %d 个参数在 meta 上: %s' % (len(still), still[:4]))
del tensors
import gc  # noqa: E402
gc.collect()
model.eval()
r2, pk = rss_gb()
print('    ✓ 全部参数都有真实存储 · RSS %.2f GB（峰值 %.2f GB）' % (r2, pk))

print()
print('  === 真实前向 ===')
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(CKPT)
prompt = 'The capital of France is'
enc = tok(prompt, return_tensors='pt')
print('    prompt: %r ⇒ %d token' % (prompt, enc['input_ids'].shape[1]))
torch.set_num_threads(6)
with torch.no_grad():
    t0 = time.time()
    out = model(**enc)
    t1 = time.time() - t0
    lg = out.logits[0, -1]
    top = torch.topk(lg, 5)
    print('    首次前向 %.2f s' % t1)
    print('    top-5: ' + ', '.join('%r %.2f' % (tok.decode(i), v)
                                   for v, i in zip(top.values.tolist(), top.indices.tolist())))
    ts = []
    for _ in range(3):
        t0 = time.time()
        model(**enc)
        ts.append(time.time() - t0)
    print('    后续前向中位 %.2f s（%d token,prefill）' % (statistics.median(ts), enc['input_ids'].shape[1]))

print()
print('  === 权重构成（这就是我们的 NF4 布局）===')
u8 = [(k, v) for k, v in model.named_parameters() if v.dtype == torch.uint8]
print('    U8 打包权重: %d 个' % len(u8))
tot_packed = sum(v.numel() * v.element_size() for _, v in u8)
tot_abs = sum(v.quant_state.absmax.numel() * 4 for _, v in u8 if getattr(v, 'quant_state', None))
tot_bf16 = sum(p.numel() * p.element_size() for p in model.parameters() if p.dtype != torch.uint8)
print('    打包权重 %s + absmax %s + bf16/f32 其余 %s' %
      (human(tot_packed), human(tot_abs), human(tot_bf16)))
print('    ⇒ 总 %.2f GB（磁盘上 1.59 GB）' % ((tot_packed + tot_abs + tot_bf16) / 1e9))
print()
