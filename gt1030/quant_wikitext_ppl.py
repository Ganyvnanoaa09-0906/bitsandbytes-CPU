"""wikitext-2 test PPL for base/GPTQ/AWQ Qwen2.5-0.5B (40 chunks x 2048)."""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from gptq_awq_cpu import dequant_awq, dequant_gptq

BASE = r'D:\work\models\qwen05_base.safetensors'
CFG = r'D:\work\models\qwen05_base_config.json'
TOK = r'D:\work\models\Qwen2.5-0.5B-Instruct-AWQ'
CKPT = {'gptq': r'D:\work\models\qwen05_gptq.safetensors',
        'awq': r'D:\work\models\qwen05_awq.safetensors'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['base', 'gptq', 'awq'])
    ap.add_argument('--chunks', type=int, default=40)
    args = ap.parse_args()
    torch.set_num_threads(6)
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(x['text'] for x in ds)
    tok = AutoTokenizer.from_pretrained(TOK)
    cfg = AutoConfig.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.load_state_dict(load_file(BASE), strict=False)
    n = 0
    if args.format != 'base':
        ck = safe_open(CKPT[args.format], framework='pt', device='cpu')
        keys = set(ck.keys())
        nq = {k: ck.get_tensor(k) for k in keys
              if not any(x in k for x in ('.qweight', '.qzeros', '.scales', '.g_idx'))}
        model.load_state_dict(nq, strict=False)
        for name, module in model.named_modules():
            if not hasattr(module, 'weight') or module.weight.ndim != 2:
                continue
            pref = name + '.'
            if pref + 'qweight' not in keys or pref + 'qzeros' not in keys:
                continue
            qw = ck.get_tensor(pref + 'qweight'); qz = ck.get_tensor(pref + 'qzeros'); sc = ck.get_tensor(pref + 'scales')
            try:
                w = dequant_gptq(qw, qz, sc) if args.format == 'gptq' else dequant_awq(qw, qz, sc)
            except Exception:
                continue
            if tuple(w.shape) == tuple(module.weight.shape):
                module.weight.data.copy_(w.to(module.weight.dtype)); n += 1
            elif tuple(w.t().shape) == tuple(module.weight.shape):
                module.weight.data.copy_(w.t().to(module.weight.dtype)); n += 1
    ids = tok(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = []
    for i in range(args.chunks):
        s = i * 2048
        if s + 2048 > ids.numel():
            break
        chunks.append(ids[s:s + 2048])
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for c in chunks:
            inp = c.unsqueeze(0)
            out = model(input_ids=inp, labels=inp)
            total_nll += float(out.loss) * (c.numel() - 1)
            total_tok += c.numel() - 1
    ppl = math.exp(total_nll / total_tok)
    print('RESULT ' + json.dumps(dict(fmt=args.format, replaced=n, chunks=len(chunks),
                                      tokens=total_tok, ppl=ppl)), flush=True)


if __name__ == '__main__':
    main()
