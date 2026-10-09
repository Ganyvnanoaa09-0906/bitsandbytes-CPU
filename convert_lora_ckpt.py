"""convert_lora_ckpt.py -- this fork's trainer checkpoint -> a diffusers LoRA file.

The trainer saves what model.state_dict() calls the adapters, which comes out of
PEFT with its wrapper prefixes:

    base_model.model.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight

pipe.load_lora_weights expects:

    unet.down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_k.lora_A.weight

so three substitutions: strip base_model.model., drop the .default adapter name,
and prefix unet. Getting this wrong does not raise -- load_lora_weights reports
how many keys it matched, and a silent zero match would just sample the base
model and look like "the LoRA did nothing".

Usage:
  python convert_lora_ckpt.py --ckpt i5build/adiff_run1/ckpt_500.pt \
      --out i5build/adiff_run1/diffusers_lora.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8')


def convert(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if 'lora' not in k.lower():
            continue
        k2 = k
        if k2.startswith('base_model.model.'):
            k2 = 'unet.' + k2[len('base_model.model.'):]
        k2 = k2.replace('.lora_A.default.', '.lora_A.').replace('.lora_B.default.', '.lora_B.')
        k2 = k2.replace('.lora_A.default', '.lora_A').replace('.lora_B.default', '.lora_B')
        out[k2] = v.detach().to(torch.float32).contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('lora') or ck.get('model') or ck
    n_all = len(sd)
    conv = convert(sd)
    if not conv:
        raise SystemExit('no LoRA tensors found in %s (%d tensors total)' % (a.ckpt, n_all))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    from safetensors.torch import save_file
    save_file(conv, a.out)
    print('  输入 %d 张量 -> LoRA %d 个' % (n_all, len(conv)))
    ks = sorted(conv)
    print('  示例键: %s' % ks[0])
    a_keys = [k for k in ks if k.endswith('lora_A.weight')]
    b_keys = [k for k in ks if k.endswith('lora_B.weight')]
    print('  lora_A=%d  lora_B=%d  %s' % (len(a_keys), len(b_keys),
                                          '成对 ✓' if len(a_keys) == len(b_keys) else '不成对 ✗'))
    print('  写出 %s  %.2f MB' % (a.out, os.path.getsize(a.out) / 2**20))
    print('  自检: 全部键以 unet. 开头 ⇒ %s'
          % ('是 ✓' if all(k.startswith('unet.') for k in ks) else '否 ✗'))
    print('       不含 .default ⇒ %s'
          % ('是 ✓' if not any('.default' in k for k in ks) else '否 ✗'))


if __name__ == '__main__':
    main()
