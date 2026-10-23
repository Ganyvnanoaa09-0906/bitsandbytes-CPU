"""Split Wan UMT5 safetensors into two shards, then stream-load into UMT5EncoderModel."""
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
SH1 = r'D:\work\textmodel\Wan2.1-T2V-1.3B\umt5_shard0.safetensors'
SH2 = r'D:\work\textmodel\Wan2.1-T2V-1.3B\umt5_shard1.safetensors'


def split():
    sd = torch.load(P, map_location='cpu', weights_only=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    keys = list(sd)
    # balance by cumulative bytes
    total = sum(sd[k].numel() * sd[k].element_size() for k in keys)
    acc = 0
    shard0 = {}
    for k in keys:
        b = sd[k].numel() * sd[k].element_size()
        if acc < total / 2:
            shard0[k] = sd[k]; acc += b
    shard1 = {k: v for k, v in sd.items() if k not in shard0}
    print('split %d + %d tensors, %.2f vs %.2f GB' %
          (len(shard0), len(shard1), acc / 1e9, (total - acc) / 1e9), flush=True)
    save_file(shard0, SH1); save_file(shard1, SH2)
    del sd, shard0, shard1
    gc.collect()


def load_model():
    model = build_model()
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    n = bad = 0
    for path in (SH1, SH2):
        with safe_open(path, framework='pt', device='cpu') as f:
            for k in f.keys():
                nk = map_key(k)
                if nk is None or nk not in targets:
                    bad += 1
                    continue
                t = targets[nk]
                v = f.get_tensor(k)
                if tuple(v.shape) != tuple(t.shape):
                    bad += 1
                    continue
                with torch.no_grad():
                    t.data.copy_(v.to(t.dtype))
                if nk == 'shared.weight':
                    targets['encoder.embed_tokens.weight'].data.copy_(v.to(t.dtype))
                n += 1
    print('loaded %d bad %d rss %.2fGB' % (n, bad,
          __import__('psutil').Process().memory_info().rss / 1e9), flush=True)
    model.eval()
    return model


def main():
    torch.set_num_threads(6)
    if not (os.path.exists(SH1) and os.path.exists(SH2)):
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
