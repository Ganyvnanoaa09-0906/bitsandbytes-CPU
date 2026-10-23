"""Inspect Wan2.1 original checkpoint keys vs diffusers WanTransformer3DModel keys."""
from __future__ import annotations

import torch
from diffusers import WanTransformer3DModel

CKPT = r'D:\work\textmodel\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors'


def main():
    from safetensors.torch import load_file
    sd = load_file(CKPT)
    print('original keys', len(sd))
    for k in list(sd)[:40]:
        print('  ', k, tuple(sd[k].shape))
    m = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=12,
                              attention_head_dim=128, in_channels=16, out_channels=16,
                              text_dim=4096, freq_dim=256, ffn_dim=8960,
                              num_layers=30, cross_attn_norm=True,
                              qk_norm='rms_norm', eps=1e-6,
                              image_dim=None, rope_max_seq_len=1024)
    d = m.state_dict()
    print('diffusers keys', len(d))
    for k in list(d)[:50]:
        print('  ', k, tuple(d[k].shape))


if __name__ == '__main__':
    main()
