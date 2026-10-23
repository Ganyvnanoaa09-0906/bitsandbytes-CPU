"""Convert Wan UMT5 pth -> safetensors (single load/save) for streaming model load."""
from __future__ import annotations

import os
import time

import torch
from safetensors.torch import save_file

P = r'D:\work\textmodel\Wan2.1-T2V-1.3B\models_t5_umt5-xxl-enc-bf16.pth'
OUT = r'D:\work\textmodel\Wan2.1-T2V-1.3B\umt5_encoder_bf16.safetensors'


def main():
    t0 = time.perf_counter()
    sd = torch.load(P, map_location='cpu', weights_only=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    print('loaded %d tensors %.1fs' % (len(sd), time.perf_counter() - t0), flush=True)
    t0 = time.perf_counter()
    save_file(sd, OUT, metadata={'format': 'pt'})
    print('saved %s %.1fs' % (OUT, time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    main()
