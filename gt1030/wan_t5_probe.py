"""Inspect Wan UMT5 checkpoint keys vs transformers UMT5EncoderModel keys (meta load)."""
from __future__ import annotations

import torch
from transformers import UMT5Config, UMT5EncoderModel

P = r'D:\work\textmodel\Wan2.1-T2V-1.3B\models_t5_umt5-xxl-enc-bf16.pth'


def main():
    sd = torch.load(P, map_location='meta', weights_only=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    print('orig keys', len(sd))
    for k in list(sd)[:30]:
        print(' ', k, tuple(sd[k].shape))
    cfg = UMT5Config(vocab_size=256384, d_model=4096, d_kv=64, d_ff=10240,
                     num_layers=24, num_heads=64, relative_attention_num_buckets=32,
                     dropout_rate=0.0, layer_norm_epsilon=1e-6, feed_forward_proj='gated-gelu')
    m = UMT5EncoderModel(cfg)
    d = m.state_dict()
    print('diff keys', len(d))
    for k in list(d)[:30]:
        print(' ', k, tuple(d[k].shape))


if __name__ == '__main__':
    main()
