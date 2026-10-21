"""probe_kv_fp8_longctx.py — FP8 KV cache at longer contexts on Qwen3-1.7B.

For 256/512/1024-token prompts: BF16 DynamicCache vs packed FP8 cache. Measures prefill
time, decode s/token, RSS after prefill and after decode, and greedy-token agreement.
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


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    base_text = ('Long-context CPU inference must keep the KV cache in ordinary DRAM, '
                 'so cache compression directly trades memory bandwidth for capacity. ')
    all_ids = tok(base_text * 200, return_tensors='pt').input_ids[0]

    def run(cache, ids, steps=4):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(ids.unsqueeze(0), past_key_values=cache, use_cache=True)
            pre = time.perf_counter() - t0
            r_pre = rss_gb()
            cur = out.logits[:, -1:].argmax(-1)
            toks = []
            t0 = time.perf_counter()
            for _ in range(steps):
                o = model(cur, past_key_values=cache, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1)
                toks.append(int(cur))
            dec = (time.perf_counter() - t0) / steps
            r_dec = rss_gb()
        return toks, pre, dec, r_pre, r_dec

    for n in (256, 512, 1024):
        ids = all_ids[:n]
        base_t, bp, bd, brp, brd = run(DynamicCache(), ids)
        f8c = QuantizedKVCache(mode='fp8')
        f8_t, fp, fd, frp, frd = run(f8c, ids)
        match = sum(a == b for a, b in zip(base_t, f8_t))
        print('n=%4d | BF16 pre %.1fs dec %.3fs RSS %.2f/%.2f GB | '
              'FP8 pre %.1fs dec %.3fs RSS %.2f/%.2f GB | match %d/%d'
              % (n, bp, bd, brp, brd, fp, fd, frp, frd, match, len(base_t)),
              flush=True)


if __name__ == '__main__':
    main()
