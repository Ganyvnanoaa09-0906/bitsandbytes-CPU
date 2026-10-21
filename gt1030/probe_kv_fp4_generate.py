"""probe_kv_fp4_generate.py — greedy decode with FP8/MXFP4 KV cache on Qwen3-1.7B.

For each prompt: prefill once, then 16 greedy steps with BF16 cache vs a cache re-quantized
to FP8 E4M3 / MXFP4 E2M1 after every step. Compares top-1 tokens and per-step logit KL.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from probe_kv_fp4 import quant_fp8, quant_mxfp4  # noqa: E402

MODEL = r'D:\work\llm_env\Qwen3-1.7B'
PROMPTS = [
    ('The key idea behind KV cache compression for long-context inference is '
     'to reduce the memory footprint while preserving attention quality. ' * 4),
    ('Explain why CPU inference of large language models is often memory-bandwidth '
     'bound, and how KV cache quantization changes that trade-off. ' * 3),
]


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()

    for pi, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors='pt').input_ids
        with torch.no_grad():
            out = model(ids, use_cache=True)
        orig = [(layer.keys.clone(), layer.values.clone()) for layer in out.past_key_values.layers]
        first = out.logits[:, -1:].argmax(-1)

        def run(quant_fn=None, steps=16):
            cache = DynamicCache()
            for k, v in orig:
                cache.update(k.clone(), v.clone(), len(cache.layers))
            cur = first.clone()
            toks, lps = [], []
            with torch.no_grad():
                for _ in range(steps):
                    o = model(cur, past_key_values=cache, use_cache=True)
                    logits = o.logits[:, -1].float()
                    lps.append(torch.log_softmax(logits, -1))
                    nxt = logits.argmax(-1, keepdim=True)
                    toks.append(int(nxt))
                    if quant_fn is not None:
                        for layer in cache.layers:
                            layer.keys = quant_fn(layer.keys)
                            layer.values = quant_fn(layer.values)
                    cur = nxt
            return toks, lps

        ref_t, ref_lp = run(None)
        print('prompt %d: %d tokens, BF16 first 8 %s' %
              (pi, ids.shape[1], ref_t[:8]), flush=True)
        for name, fn in (('fp8', quant_fp8), ('mxfp4', quant_mxfp4)):
            t, lp = run(fn)
            match = sum(a == b for a, b in zip(ref_t, t))
            kls = [float(torch.sum(torch.exp(a) * (a - b))) for a, b in zip(ref_lp, lp)]
            print('  %s: top1 %2d/%2d | KL mean %.5f max %.5f | tokens %s'
                  % (name, match, len(ref_t), sum(kls) / len(kls), max(kls), t[:8]),
                  flush=True)


if __name__ == '__main__':
    main()
