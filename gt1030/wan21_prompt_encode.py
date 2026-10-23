"""Stage 1: encode real prompt with streamed UMT5, save (1,512,4096) hidden."""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from wan21_t5 import TOK
from wan21_t5_shards8 import load_model


def main():
    torch.set_num_threads(6)
    t0 = time.perf_counter()
    model = load_model()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    enc = tok(['a cat walking on grass'], padding='max_length', max_length=512,
              truncation=True, return_tensors='pt')
    with torch.no_grad():
        h = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
    print('encode %.1fs shape %s finite %s mean %.4f std %.4f'
          % (time.perf_counter() - t0, tuple(h.shape), bool(torch.isfinite(h).all()),
             float(h.float().mean()), float(h.float().std())), flush=True)
    torch.save(h.detach().to(torch.float32),
               r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_cat_hidden.pt')


if __name__ == '__main__':
    main()
