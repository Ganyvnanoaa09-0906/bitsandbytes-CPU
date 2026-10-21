"""probe_kv_4k_real.py — real 4K prefill + short decode, BF16 vs FP8 KV cache.

Runs sequentially in one process: BF16 DynamicCache first, then packed FP8 cache.
Reports prefill time, RSS, first decode tokens and token agreement. CPU ~10 min/mode.
"""
from __future__ import annotations

import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from probe_kv_fp4_cache import QuantizedKVCache  # noqa: E402

MODEL = r'D:\work\llm_env\Qwen3-1.7B'


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    text = ('Long-context CPU inference is memory-bandwidth bound. KV cache compression '
            'trades capacity for bandwidth. This sentence is repeated to build a 4K prompt. ')
    ids = tok(text * 400, return_tensors='pt').input_ids[0, :4096].unsqueeze(0)
    print('prompt tokens', ids.shape[1], 'RSS %.2f GB' % rss(), flush=True)

    def run(cache, label):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(ids, past_key_values=cache, use_cache=True)
            pre = time.perf_counter() - t0
            r = rss()
            cur = out.logits[:, -1:].argmax(-1)
            toks = []
            for _ in range(2):
                o = model(cur, past_key_values=cache, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1)
                toks.append(int(cur))
        print('%s | prefill %.1f s | RSS %.2f GB | tokens %s'
              % (label, pre, r, toks), flush=True)
        return toks

    base = run(DynamicCache(), 'BF16 ')
    f8 = run(QuantizedKVCache(mode='fp8'), 'FP8  ')
    print('token match %d/%d' % (sum(a == b for a, b in zip(base, f8)), len(base)), flush=True)


if __name__ == '__main__':
    main()
