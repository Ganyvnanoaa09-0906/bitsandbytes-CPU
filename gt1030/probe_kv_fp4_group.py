"""probe_kv_fp4_group.py — does finer MXFP4 scale grouping rescue real-cache tokens?"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from probe_kv_fp4_cache import QuantizedKVCache  # noqa: E402

MODEL = r'D:\work\llm_env\Qwen3-1.7B'


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    prompt = ('The key idea behind KV cache compression for long-context inference is '
              'to reduce the memory footprint while preserving attention quality. ' * 4)
    ids = tok(prompt, return_tensors='pt').input_ids
    with torch.no_grad():
        out = model(ids, use_cache=True)
    first = out.logits[:, -1:].argmax(-1)

    def run(cache, steps=8):
        with torch.no_grad():
            o = model(ids, past_key_values=cache, use_cache=True)
            cur = o.logits[:, -1:].argmax(-1)
            toks = []
            for _ in range(steps):
                o = model(cur, past_key_values=cache, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1)
                toks.append(int(cur))
        return toks

    base = run(DynamicCache())
    print('BF16', base, flush=True)
    for group in (16, 8, 4):
        c = QuantizedKVCache(group=group)
        t = run(c)
        match = sum(a == b for a, b in zip(base, t))
        packed = sum(l.kp.numel() + l.ks.numel() * l.ks.element_size() +
                     l.vp.numel() + l.vs.numel() * l.vs.element_size()
                     for l in c.layers)
        print('group=%2d | match %d/%d | packed %.2f MB | tokens %s'
              % (group, match, len(base), packed / 1e6, t), flush=True)


if __name__ == '__main__':
    main()
