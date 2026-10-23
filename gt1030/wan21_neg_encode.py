"""Encode empty prompt with streamed UMT5 -> negative embeds (512,4096)."""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from wan21_t5 import TOK
from wan21_t5_shards8 import load_model


def main():
    torch.set_num_threads(6)
    model = load_model()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    enc = tok([''], padding=False, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        h = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
    seq = int(enc['attention_mask'].sum())
    h = h[0, :seq]
    h = torch.cat([h, h.new_zeros(512 - seq, h.shape[1])], dim=0).unsqueeze(0)
    print('neg seq %d shape %s finite %s mean %.6f std %.6f'
          % (seq, tuple(h.shape), bool(torch.isfinite(h).all()),
             float(h.float().mean()), float(h.float().std())), flush=True)
    torch.save(h.detach().float(),
               r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_empty_hidden.pt')


if __name__ == '__main__':
    main()
