"""PPL of base vs dequant GPTQ/AWQ on one fixed text (CPU)."""
from __future__ import annotations

import argparse
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
TEXT = ('The quick brown fox jumps over the lazy dog. A CPU is a general purpose processor. '
        'Quantization reduces memory but can change model outputs. ' * 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['gptq', 'awq'])
    args = ap.parse_args()
    torch.set_num_threads(6)
    cfg = AutoConfig.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    model.load_state_dict(load_file(BASE), strict=False)
    tok = AutoTokenizer.from_pretrained(TOK)
    enc = tok(TEXT, return_tensors='pt', truncation=True, max_length=128)
    ids = enc['input_ids']

    def ppl():
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        return float(torch.exp(out.loss)), float(out.loss)

    base_ppl, base_loss = ppl()
    ck = safe_open(CKPT[args.format], framework='pt', device='cpu')
    keys = set(ck.keys())
    # overlay nonquant weights from the quantized checkpoint (lm_head/embed/norms)
    nq = {k: ck.get_tensor(k) for k in keys
          if not any(x in k for x in ('.qweight', '.qzeros', '.scales', '.g_idx'))}
    if nq:
        model.load_state_dict(nq, strict=False)
        print('[overlay] nonquant keys', len(nq), flush=True)
    n = 0
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
    q_ppl, q_loss = ppl()
    print('RESULT %s replaced=%d base_ppl=%.4f quant_ppl=%.4f delta_pct=%.2f'
          % (args.format, n, base_ppl, q_ppl, 100.0 * (q_ppl / base_ppl - 1.0)), flush=True)


if __name__ == '__main__':
    main()
