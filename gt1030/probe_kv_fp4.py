"""probe_kv_fp4.py — MXFP4/FP8 KV-cache quantization probe on Qwen3-1.7B.

Loads Qwen3-1.7B (bf16), builds a real KV cache for a prompt, quantizes K/V with
  * BF16 (reference),
  * FP8 E4M3 (per-16-block scale),
  * MXFP4 E2M1 (4-bit codebook, per-16-block scale),
then replaces the cache and runs one decode step. Reports KV reconstruction relL2,
next-token logits agreement, and effective bytes/token.
"""
from __future__ import annotations

import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r'D:\work\llm_env\Qwen3-1.7B'

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_mxfp4(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """Simulate MXFP4 E2M1 with per-group max scale (dequantized back to bf16)."""
    orig = x.shape
    xb = x.float().reshape(-1, group)
    maxabs = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = maxabs / 6.0
    q = xb / scale
    a = q.abs()
    idx = torch.bucketize(a, (E2M1[:-1] + E2M1[1:]) / 2.0).clamp(0, 7)
    qdq = E2M1[idx] * torch.sign(q) * scale
    return qdq.reshape(orig).to(x.dtype)


def quant_fp8(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    """FP8 E4M3 with per-16-block scale (simulated, dequantized back to bf16)."""
    orig = x.shape
    xb = x.float().reshape(-1, group)
    maxabs = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = maxabs / 448.0
    q = (xb / scale).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(orig).to(x.dtype)


def bytes_per_token(kv, dtype_bits=16):
    n = 0
    for layer in kv.layers:
        if layer.keys is not None:
            n += layer.keys.numel() + layer.values.numel()
    return n * dtype_bits / 8


def quant_cache(cache, fn):
    rels = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        kq, vq = fn(k), fn(v)
        rels.append(float((kq - k).norm() / k.norm()))
        rels.append(float((vq - v).norm() / v.norm()))
        layer.keys, layer.values = kq, vq
    return sum(rels) / len(rels)


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    prompt = ('The key idea behind KV cache compression for long-context inference is '
              'to reduce the memory footprint while preserving attention quality. ' * 8)
    ids = tok(prompt, return_tensors='pt').input_ids
    print('prompt tokens', ids.shape[1], flush=True)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(ids, use_cache=True)
        print('prefill %.2f s | logits %s' % (time.perf_counter() - t0,
                                              tuple(out.logits.shape)), flush=True)
    cache = out.past_key_values
    orig = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    next_id = out.logits[:, -1:].argmax(-1)
    with torch.no_grad():
        ref = model(next_id, past_key_values=cache, use_cache=False).logits[:, -1]
    ref_top = int(ref.argmax(-1)); ref_lp = torch.log_softmax(ref.float(), -1)

    seq = ids.shape[1]
    bytes_bf16 = bytes_per_token(cache, 16) / seq
    bytes_fp8 = bytes_bf16 * 8.5 / 16
    bytes_fp4 = bytes_bf16 * 4.5 / 16
    print('KV elements/token %.0f | BF16 %.1f KB/token | FP8(4.5bit) %.1f KB | '
          'MXFP4(4.5bit) %.1f KB'
          % (bytes_bf16 / 2, bytes_bf16 / 1024, bytes_fp8 / 1024,
             bytes_fp4 / 1024), flush=True)

    for name, fn in (('fp8', quant_fp8), ('mxfp4', quant_mxfp4)):
        for layer, (k, v) in zip(cache.layers, orig):
            layer.keys, layer.values = k.clone(), v.clone()
        rel = quant_cache(cache, fn)
        with torch.no_grad():
            logits = model(next_id, past_key_values=cache, use_cache=False).logits[:, -1]
        lp = torch.log_softmax(logits.float(), -1)
        kl = float(torch.sum(torch.exp(ref_lp) * (ref_lp - lp)))
        top1 = int(logits.argmax(-1))
        print('%s | KV relL2 %.4f | top1 %s (ref %s) | KL %.4f | max|dlogit| %.3f'
              % (name, rel, top1 == ref_top, ref_top, kl,
                 float((logits - ref).abs().max())), flush=True)


if __name__ == '__main__':
    main()
