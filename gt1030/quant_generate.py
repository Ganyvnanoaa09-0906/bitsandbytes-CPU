"""Generate 20 greedy tokens from base vs dequantized GPTQ/AWQ model."""
from __future__ import annotations

import argparse
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from gptq_awq_cpu import dequant_gptq, dequant_awq

BASE_STATE = r'D:\work\models\qwen05_base.safetensors'
BASE_CFG = r'D:\work\models\qwen05_base_config.json'
TOK = r'D:\work\models\Qwen2.5-0.5B-Instruct-AWQ'
CKPT = {'gptq': r'D:\work\models\qwen05_gptq.safetensors',
        'awq': r'D:\work\models\qwen05_awq.safetensors'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['gptq', 'awq'])
    args = ap.parse_args()
    torch.set_num_threads(6)
    cfg = AutoConfig.from_pretrained(BASE_CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    sd = load_file(BASE_STATE)
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained(TOK)
    prompt = 'Once upon a time'
    ids = tok(prompt, return_tensors='pt', return_token_type_ids=False)['input_ids']
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    print('BASE    :', tok.decode(out[0], skip_special_tokens=True), flush=True)
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
        except Exception as e:
            print('ERR', name, e); continue
        if tuple(w.shape) == tuple(module.weight.shape):
            module.weight.data.copy_(w.to(module.weight.dtype)); n += 1
        elif tuple(w.t().shape) == tuple(module.weight.shape):
            module.weight.data.copy_(w.t().to(module.weight.dtype)); n += 1
    print('replaced', n, flush=True)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
    print(args.format.upper() + ': ', tok.decode(out[0], skip_special_tokens=True), flush=True)


if __name__ == '__main__':
    main()
