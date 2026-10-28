"""Thunder-2bit: Qwen3-1.7B inference formats fp16 / NF4 / q3bit / q2bit.

One precision per process. Measures prefill M=64/128, decode M=1, model RSS,
weight bytes, effective weight bandwidth, PPL and saves logits for top-1 comparison.

Usage: python thunder2bit_bench.py --precision fp16|nf4|q3bit|q2bit
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

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
sys.path.insert(0, r'D:\work\bnb-quant')

MODEL = r'D:\work\llm_env\Qwen3-1.7B'
TEXT = ('The quick brown fox jumps over the lazy dog. '
        'A CPU is a general purpose processor. ' * 16)


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def replace_4bit(module, prefix=''):
    import bitsandbytes as bnb
    n = 0
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, torch.nn.Linear):
            q = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  compute_dtype=torch.float32,
                                  quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.clone(),
                                             requires_grad=False,
                                             quant_type='nf4', compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
            setattr(module, name, q)
            n += 1
        else:
            n += replace_4bit(child, full)
    return n


def weight_bytes(model, precision):
    if precision == 'fp16':
        return sum(p.numel() * p.element_size() for p in model.parameters())
    if precision == 'nf4':
        # packed 4-bit + absmax + other params
        total = 0
        for m in model.modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'quant_state'):
                total += m.weight.numel() * 0.5
                qs = m.weight.quant_state
                for b in getattr(qs, '__dict__', {}).values():
                    if torch.is_tensor(b):
                        total += b.numel() * b.element_size()
            elif hasattr(m, 'quant_state'):
                pass
        others = sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                     if 'weight' not in n and not n.endswith('bias'))
        return total + others + sum(p.numel() * 4 for n, p in model.named_parameters()
                                    if n.endswith('bias'))
    if precision == 'q3bit':
        from bitsandbytes import q3bit
        total = 0
        for m in model.modules():
            if isinstance(m, q3bit.QuantLinear3bit):
                total += m.packed.numel() + m.absmax.numel() * 4
                total += (m.bias.numel() * 4 if m.bias is not None else 0)
                total += m.lora_A.numel() * 4 + m.lora_B.numel() * 4
        total += sum(p.numel() * p.element_size() for n, p in model.named_parameters()
                     if not any(k in n for k in ('.lora_',)) and 'q3' not in n)
        return total
    if precision == 'q2bit':
        from scripts.lib import linear2bit
        fp = linear2bit.footprint(model)
        return fp['total']
    return 0


def load_model(precision):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16
    kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    except TypeError:
        kw.pop('torch_dtype'); kw['dtype'] = dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    model.eval()
    n = 0
    if precision == 'nf4':
        n = replace_4bit(model)
    elif precision == 'q3bit':
        from bitsandbytes import q3bit
        n = q3bit.convert_linears_3bit(model, block=64, lora_r=1,
                                       skip_names=('lm_head',))
    elif precision == 'q2bit':
        from scripts.lib import linear2bit
        n, _ = linear2bit.convert(model)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return model, tok, n


def median_ms(fn, iters=3):
    fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', required=True, choices=['fp16', 'nf4', 'q3bit', 'q2bit'])
    ap.add_argument('--outdir', default=r'D:\work\research\bench_results\thunder2bit')
    args = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.perf_counter(); model, tok, nconv = load_model(args.precision)
    load_s = time.perf_counter() - t0
    wb = weight_bytes(model, args.precision)
    ids64 = torch.randint(1, 1000, (1, 64))
    ids128 = torch.randint(1, 1000, (1, 128))
    id1 = torch.randint(1, 1000, (1, 1))
    with torch.no_grad():
        p64 = median_ms(lambda: model(input_ids=ids64, use_cache=False).logits)
        p128 = median_ms(lambda: model(input_ids=ids128, use_cache=False).logits)
        d1 = median_ms(lambda: model(input_ids=id1, use_cache=False).logits)
        # PPL on fixed text
        enc = tok(TEXT, return_tensors='pt', truncation=True, max_length=128)
        out = model(input_ids=enc['input_ids'], labels=enc['input_ids'])
        loss = float(out.loss)
        # save logits for top1 comparison
        logits = model(input_ids=ids128, use_cache=False).logits.detach().float().numpy()
    np.save(os.path.join(args.outdir, 'logits_%s.npy' % args.precision), logits)
    res = dict(precision=args.precision, converted=nconv, load_s=load_s,
               weight_bytes=wb, weight_gb=wb / 1e9, rss_gb=rss_gb(),
               prefill64_ms=p64, prefill128_ms=p128, decode1_ms=d1,
               decode_tokens_per_s=1000.0 / d1,
               effective_weight_gbps=(wb / 1e9) / (d1 / 1000.0),
               ppl=float(np.exp(loss)), loss=loss)
    print('RESULT ' + json.dumps(res), flush=True)
    with open(os.path.join(args.outdir, '%s.json' % args.precision), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
