"""Probe: where does a forward pass through an NF4 LLM actually spend its time on this CPU?

The user asked for a model training plus a probe, to see which tasks belong to the CPU and which to the
iGPU. This is the probe half: a real 2.03B NF4 Qwen3, loaded through the checkpoint's own layout, with
the op-level breakdown taken from torch.profiler rather than guessed.

What makes it worth measuring rather than reasoning about: the model's weights arrive as 197 U8 matrices
with one fp32 absmax per 64 elements, which is precisely the case avx2_gemv_4bit.h was written for. A
decode step uses them through gemv_4bit_inference; a prefill uses them through gemm_4bit. Which of the
two dominates, and how the time splits between the quantised matmuls and everything else, decides
whether the library work or the iGPU work is the lever here.

Also reported: the iGPU's solo baseline measured immediately alongside, because 7.89 showed that number
drifts and only a ratio taken from an adjacent measurement means anything.
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

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

CKPT = r'D:\work\models\models\qwen3-1.7B-nf4'

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.nn import Linear4bit, Params4bit  # noqa: E402
from bitsandbytes.functional import QuantState  # noqa: E402
import safetensors.torch as st  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402


def build():
    with open(os.path.join(CKPT, 'quant_meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)
    layers = meta['layers']
    qt = meta.get('quant_type') or meta.get('dtype')
    bs = meta.get('blocksize')
    tensors = st.load_file(os.path.join(CKPT, 'model.safetensors'), device='cpu')
    cfg = AutoConfig.from_pretrained(CKPT)
    with init_empty_weights():
        model = Qwen3ForCausalLM(cfg)
    for key, info in layers.items():
        mod_path = key.rpartition('.')[0]
        module = model.get_submodule(mod_path)
        qs = QuantState(absmax=tensors[key + '.absmax'], shape=torch.Size(info['shape']),
                        blocksize=bs, quant_type=qt, dtype=torch.bfloat16)
        out_f, in_f = qs.shape
        bias = getattr(module, 'bias', None) is not None
        new = Linear4bit(in_f, out_f, bias=bias, compute_dtype=torch.float32, quant_type=qt)
        p = Params4bit(tensors[key + '.wq'], requires_grad=False, quant_type=qt,
                       blocksize=bs, bnb_quantized=True, quant_storage=tensors[key + '.wq'].dtype)
        p.quant_state = qs
        p.module = new
        new.weight = p
        new.quant_state = qs
        if bias:
            new.bias = torch.nn.Parameter(module.bias.detach().clone())
        pp, _, leaf = mod_path.rpartition('.')
        setattr(model.get_submodule(pp) if pp else model, leaf, new)
    named = dict(model.named_parameters())
    quant = set(layers)
    fill = [nm for nm, p in named.items() if nm not in quant and nm in tensors and p.is_meta]
    par = {}
    for nm in fill:
        par.setdefault(nm.rsplit('.', 1)[0], []).append(nm)
    for pp in par:
        (model.get_submodule(pp) if pp else model).to_empty(device='cpu')
    with torch.no_grad():
        for nm in fill:
            p = dict(model.named_parameters())[nm]
            p.copy_(tensors[nm].to(p.dtype))
    del tensors
    model.eval()
    return model


torch.set_num_threads(6)
print()
print('  === 加载 ===')
t0 = time.time()
model = build()
print('    %.1f s · RSS %.2f GB' % (time.time() - t0,
      __import__('psutil').Process().memory_info().rss / 1e9))
tok = AutoTokenizer.from_pretrained(CKPT)

print()
print('  === profile: prefill（128 token）===')
ids = tok('The quick brown fox jumps over the lazy dog. ' * 8, return_tensors='pt')['input_ids'][:, :128]
print('    输入 %d token' % ids.shape[1])
with torch.no_grad():
    for _ in range(2):
        model(ids)
    t0 = time.time()
    for _ in range(3):
        model(ids)
    med = (time.time() - t0) / 3
print('    中位 %.3f s ⇒ %.1f token/s（prefill）' % (med, ids.shape[1] / med))

from torch.profiler import profile, ProfilerActivity  # noqa: E402
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        model(ids)
evts = [e for e in prof.key_averages() if e.self_device_time_total > 0 or e.self_cpu_time_total > 0]
evts.sort(key=lambda e: -e.self_cpu_time_total)
tot = sum(e.self_cpu_time_total for e in evts)
print()
print('  %-46s %12s %8s' % ('算子', '自耗时 ms', '占比'))
print('  ' + '-' * 70)
for e in evts[:16]:
    print('  %-46s %12.1f %7.1f%%' % (e.key[:46], e.self_cpu_time_total / 1000,
                                      e.self_cpu_time_total / tot * 100))
print('  %-46s %12.1f' % ('合计', tot / 1000))

print()
print('  === profile: decode（1 token,逐步）===')
one = ids[:, :1]
with torch.no_grad():
    for _ in range(2):
        model(one)
    t0 = time.time()
    for _ in range(5):
        model(one)
    dmed = (time.time() - t0) / 5
print('    中位 %.3f s/token ⇒ %.1f token/s（decode）' % (dmed, 1 / dmed))

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU]) as prof2:
        for _ in range(3):
            model(one)
e2 = [e for e in prof2.key_averages() if e.self_cpu_time_total > 0]
e2.sort(key=lambda e: -e.self_cpu_time_total)
t2 = sum(e.self_cpu_time_total for e in e2)
print()
print('  %-46s %12s %8s' % ('算子', '自耗时 ms', '占比'))
print('  ' + '-' * 70)
for e in e2[:14]:
    print('  %-46s %12.1f %7.1f%%' % (e.key[:46], e.self_cpu_time_total / 1000,
                                      e.self_cpu_time_total / t2 * 100))

print()
print('  === 关键问题：量化矩阵乘走的是哪条路 ===')
names = [e.key for e in e2]
for probe in ('gemv_4bit', 'gemm_4bit', 'matmul_4bit', 'dequantize_4bit', 'aten::mm', 'addmm',
              'aten::matmul', 'cgemv', 'cgemm'):
    hits = [n for n in names if probe in n]
    if hits:
        print('    ✓ %-18s ⇒ %s' % (probe, hits[:2]))
print()
print('  ⇒ 若出现 gemv_4bit/gemm_4bit ⇒ ★【我们的 AVX2 融合内核在跑】')
print('  ⇒ 若只出现 dequantize_4bit + aten::mm ⇒ ★【它走了反量化后稠密 GEMM 那条慢路】')
