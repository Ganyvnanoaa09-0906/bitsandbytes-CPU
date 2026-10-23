"""Wan2.1-T2V-1.3B conversion: original WanModel checkpoint -> diffusers WanTransformer3DModel.

Mapping is strictly 1:1 by shape; use load_wan_transformer() then dummy forward.
"""
from __future__ import annotations

import os
import re
import sys

import torch

CKPT = r'D:\work\textmodel\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors'


def build_model():
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128,
        in_channels=16, out_channels=16, text_dim=4096, freq_dim=256,
        ffn_dim=8960, num_layers=30, cross_attn_norm=True,
        qk_norm='rms_norm', eps=1e-6, rope_max_seq_len=1024)


def map_key(k):
    if k == 'patch_embedding.weight' or k == 'patch_embedding.bias':
        return k
    if k == 'head.head.weight':
        return 'proj_out.weight'
    if k == 'head.head.bias':
        return 'proj_out.bias'
    if k == 'head.modulation':
        return 'scale_shift_table'
    if k.startswith('time_embedding.0.'):
        return k.replace('time_embedding.0.', 'condition_embedder.time_embedder.linear_1.')
    if k.startswith('time_embedding.2.'):
        return k.replace('time_embedding.2.', 'condition_embedder.time_embedder.linear_2.')
    if k.startswith('text_embedding.0.'):
        return k.replace('text_embedding.0.', 'condition_embedder.text_embedder.linear_1.')
    if k.startswith('text_embedding.2.'):
        return k.replace('text_embedding.2.', 'condition_embedder.text_embedder.linear_2.')
    if k.startswith('time_projection.1.'):
        return k.replace('time_projection.1.', 'condition_embedder.time_proj.')
    m = re.match(r'^blocks\.(\d+)\.(.*)$', k)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    if rest == 'modulation':
        return f'blocks.{i}.scale_shift_table'
    if rest.startswith('self_attn.'):
        rest = rest.replace('self_attn.', 'attn1.')
    elif rest.startswith('cross_attn.'):
        rest = rest.replace('cross_attn.', 'attn2.')
    elif rest.startswith('norm3.'):
        rest = rest.replace('norm3.', 'norm2.')
    elif rest.startswith('ffn.0.'):
        rest = rest.replace('ffn.0.', 'ffn.net.0.proj.')
    elif rest.startswith('ffn.2.'):
        rest = rest.replace('ffn.2.', 'ffn.net.2.')
    else:
        return None
    # diffusers attention output is to_out.0
    rest = rest.replace('.o.weight', '.to_out.0.weight').replace('.o.bias', '.to_out.0.bias')
    rest = rest.replace('.q.', '.to_q.').replace('.k.', '.to_k.').replace('.v.', '.to_v.')
    rest = rest.replace('.norm_q.', '.norm_q.').replace('.norm_k.', '.norm_k.')
    return f'blocks.{i}.{rest}'


def load_wan_transformer(ckpt=CKPT, verbose=True):
    from safetensors.torch import load_file
    model = build_model()
    sd = load_file(ckpt)
    new, bad = {}, []
    for k, v in sd.items():
        nk = map_key(k)
        if nk is None:
            bad.append(k)
            continue
        new[nk] = v
    ref = model.state_dict()
    missing = [k for k in ref if k not in new]
    unexpected = [k for k in new if k not in ref]
    shape_bad = [(k, tuple(new[k].shape), tuple(ref[k].shape)) for k in new if k in ref and tuple(new[k].shape) != tuple(ref[k].shape)]
    if verbose:
        print('mapped %d / %d | missing %d unexpected %d shape_bad %d unmapped_orig %d'
              % (len(new), len(sd), len(missing), len(unexpected), len(shape_bad), len(bad)), flush=True)
        for k in missing[:5]:
            print('  missing', k, tuple(ref[k].shape))
        for k in unexpected[:5]:
            print('  unexpected', k)
        for b in shape_bad[:5]:
            print('  shape', b)
        for k in bad[:5]:
            print('  unmapped', k)
    model.load_state_dict(new, strict=True)
    model.eval()
    return model


def main():
    model = load_wan_transformer()
    torch.set_num_threads(6)
    B, C, T, H, W = 1, 16, 1, 32, 32
    latent = torch.randn(B, C, T, H, W)
    text = torch.randn(B, 512, 4096)
    ts = torch.tensor([500.0])
    with torch.no_grad():
        out = model(latent, timestep=ts, encoder_hidden_states=text, return_dict=False)[0]
    print('dummy forward', tuple(out.shape), 'finite', bool(torch.isfinite(out).all()),
          'mean %.4f std %.4f' % (float(out.mean()), float(out.std())))


if __name__ == '__main__':
    main()
