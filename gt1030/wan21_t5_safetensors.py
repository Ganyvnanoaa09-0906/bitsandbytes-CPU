"""Load Wan UMT5 from safetensors with per-tensor streaming, then encode a real prompt."""
from __future__ import annotations

import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from wan21_t5 import build_model, map_key, TOK

SAFE = r'D:\work\textmodel\Wan2.1-T2V-1.3B\umt5_encoder_bf16.safetensors'


def main():
    torch.set_num_threads(6)
    model = build_model()
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))
    t0 = time.perf_counter()
    n = bad = 0
    with safe_open(SAFE, framework='pt', device='cpu') as f:
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
    print('loaded %d tensors bad %d %.1fs rss %.2fGB' % (n, bad, time.perf_counter() - t0,
          __import__('psutil').Process().memory_info().rss / 1e9), flush=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    enc = tok(['a cat walking on grass'], padding='max_length', max_length=64,
              truncation=True, return_tensors='pt')
    with torch.no_grad():
        h = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
    print('prompt encoding', tuple(h.shape), 'finite', bool(torch.isfinite(h).all()),
          'mean %.4f std %.4f' % (float(h.float().mean()), float(h.float().std())))


if __name__ == '__main__':
    main()
