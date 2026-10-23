"""Split UMT5 into 8 shards and stream-load into UMT5EncoderModel."""
from __future__ import annotations

import gc
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from wan21_t5 import build_model, map_key, TOK

P = r'D:\work\textmodel\Wan2.1-T2V-1.3B\models_t5_umt5-xxl-enc-bf16.pth'
DIR = r'D:\work\textmodel\Wan2.1-T2V-1.3B\umt5_shards'
N = 8


def split():
    os.makedirs(DIR, exist_ok=True)
    sd = torch.load(P, map_location='cpu', weights_only=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    keys = list(sd)
    total = sum(sd[k].numel() * sd[k].element_size() for k in keys)
    bins = [{} for _ in range(N)]
    used = [0] * N
    for k in keys:
        b = sd[k].numel() * sd[k].element_size()
        i = min(range(N), key=lambda j: used[j])
        bins[i][k] = sd[k]; used[i] += b
    for i, d in enumerate(bins):
        save_file(d, os.path.join(DIR, 'shard%02d.safetensors' % i))
        print('shard %d: %d tensors %.2fGB' % (i, len(d), used[i] / 1e9), flush=True)
    del sd, bins
    gc.collect()


def load_model():
    model = build_model()
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    n = bad = 0
    for fn in sorted(os.listdir(DIR)):
        with safe_open(os.path.join(DIR, fn), framework='pt', device='cpu') as f:
            for k in f.keys():
                nk = map_key(k)
                if nk is None or nk not in targets:
                    bad += 1; continue
                t = targets[nk]; v = f.get_tensor(k)
                if tuple(v.shape) != tuple(t.shape):
                    bad += 1; continue
                with torch.no_grad():
                    t.data.copy_(v.to(t.dtype))
                if nk == 'shared.weight' and 'encoder.embed_tokens.weight' in targets:
                    targets['encoder.embed_tokens.weight'].data.copy_(v.to(t.dtype))
                n += 1
    print('loaded %d bad %d rss %.2fGB' % (n, bad,
          __import__('psutil').Process().memory_info().rss / 1e9), flush=True)
    model.eval()
    return model


def main():
    torch.set_num_threads(6)
    if not os.path.isdir(DIR) or len(os.listdir(DIR)) < N:
        split()
    model = load_model()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    enc = tok(['a cat walking on grass'], padding='max_length', max_length=64,
              truncation=True, return_tensors='pt')
    with torch.no_grad():
        h = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
    print('prompt encoding', tuple(h.shape), 'finite', bool(torch.isfinite(h).all()),
          'len', int(enc['attention_mask'].sum()), 'mean %.4f std %.4f'
          % (float(h.float().mean()), float(h.float().std())))


if __name__ == '__main__':
    main()
