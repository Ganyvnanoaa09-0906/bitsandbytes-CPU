"""CPU dequant of one GPTQ/AWQ Qwen2.5-0.5B checkpoint into a base HF model, then logits pass.

Usage: python gptq_awq_cpu.py --format gptq|awq
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE_STATE = r'D:\work\models\qwen05_base.safetensors'
BASE_CFG = r'D:\work\models\qwen05_base_config.json'
AWQ_DIR = r'D:\work\models\Qwen2.5-0.5B-Instruct-AWQ'
CKPT = {r'gptq': r'D:\work\models\qwen05_gptq.safetensors',
        'awq': r'D:\work\models\qwen05_awq.safetensors'}


def unpack_int32(packed: torch.Tensor, k: int) -> torch.Tensor:
    return (packed >> (4 * k)) & 0xF


def dequant_gptq(qweight, qzeros, scales, group_size=128):
    # GPTQ: qweight [in/8, out] packs IN rows; qzeros [in/group, out/8]; scales [in/group, out]
    in8, out = qweight.shape
    inn = in8 * 8
    q = torch.empty((inn, out), dtype=torch.int32)
    for k in range(8):
        q[k::8, :] = unpack_int32(qweight, k)
    ng, _ = qzeros.shape
    z = torch.empty((ng, out), dtype=torch.int32)
    for k in range(8):
        z[:, k::8] = unpack_int32(qzeros, k)
    g = torch.arange(inn) // group_size
    g = g.clamp(max=ng - 1)
    w = (q - (z[g] + 1)).float() * scales[g].float()
    return w.t()


AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def dequant_awq(qweight, qzeros, scales, group_size=128):
    # Exact autoawq dequantize_gemm for bits=4 on CPU.
    shifts = torch.arange(0, 32, 4)
    iw = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iw = iw.reshape(iw.shape[0], -1)
    iz = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    iz = iz.reshape(iz.shape[0], -1)
    rev = torch.arange(iw.shape[-1], dtype=torch.int32).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
    iw = iw[:, rev]
    iz = iz[:, rev]
    iw = torch.bitwise_and(iw, 0xF)
    iz = torch.bitwise_and(iz, 0xF)
    sc = scales.repeat_interleave(group_size, dim=0)
    iz = iz.repeat_interleave(group_size, dim=0)
    return ((iw - iz) * sc).t().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['gptq', 'awq'])
    args = ap.parse_args()
    torch.set_num_threads(6)
    cfg = AutoConfig.from_pretrained(BASE_CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    base = torch.load  # noqa
    from safetensors.torch import load_file
    t = time.time()
    sd = load_file(BASE_STATE)
    model.load_state_dict(sd, strict=False)
    print('[load] base miss/unexpected', len(model.load_state_dict(sd, strict=False).missing_keys),
          len(model.load_state_dict(sd, strict=False).unexpected_keys), flush=True)
    tok = AutoTokenizer.from_pretrained(AWQ_DIR)
    ids = tok('The quick brown fox', return_tensors='pt', return_token_type_ids=False)['input_ids']
    with torch.no_grad():
        base_logits = model(input_ids=ids).logits.float()
    torch.save(base_logits, r'D:\work\research\bench_results\q_logits_base.pt')
    print('[base logits]', tuple(base_logits.shape), flush=True)

    ck = safe_open(CKPT[args.format], framework='pt', device='cpu')
    keys = set(ck.keys())
    replaced = 0
    errs = []
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight.ndim != 2:
            continue
        pref = name + '.'
        qk = None
        for cand in (pref + 'qweight', pref + 'weight'):
            if cand in keys and (pref + 'qzeros') in keys:
                qk = cand
                break
        if qk is None:
            continue
        qweight = ck.get_tensor(pref + 'qweight')
        qzeros = ck.get_tensor(pref + 'qzeros')
        scales = ck.get_tensor(pref + 'scales')
        if args.format == 'gptq':
            w = dequant_gptq(qweight, qzeros, scales)
        else:
            w = dequant_awq(qweight, qzeros, scales)
        if tuple(w.shape) != tuple(module.weight.shape):
            # try transpose
            if tuple(w.t().shape) == tuple(module.weight.shape):
                w = w.t()
            else:
                print('[skip]', name, tuple(w.shape), tuple(module.weight.shape), flush=True)
                continue
        module.weight.data.copy_(w.to(module.weight.dtype))
        replaced += 1
        # weight error vs base
        bw = sd[name + '.weight'].float()
        errs.append(float((w.float() - bw).norm() / bw.norm().clamp_min(1e-9)))
    print('[dequant] replaced', replaced, 'weight_rel_median', float(torch.tensor(errs).median()),
          'max', max(errs), flush=True)
    with torch.no_grad():
        q_logits = model(input_ids=ids).logits.float()
    rel = float((q_logits - base_logits).norm() / base_logits.norm())
    print('RESULT ' + json.dumps(dict(fmt=args.format, replaced=replaced,
                                       weight_rel_median=float(torch.tensor(errs).median()),
                                       weight_rel_max=max(errs), logits_rel=rel,
                                       elapsed_s=round(time.time() - t, 1))), flush=True)


if __name__ == '__main__':
    main()
