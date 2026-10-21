"""probe_kv_fp4_storage.py — actual nibble-packed MXFP4 KV storage on Qwen3-1.7B.

Unlike the quality probe, this stores K/V as uint8 nibbles + fp16 per-16 scales and measures:
  * logical bytes vs BF16 cache (compression ratio),
  * pack/unpack time on CPU (overhead budget for decode),
  * reconstruction relL2.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r'D:\work\llm_env\Qwen3-1.7B'
# E2M1 positive magnitudes; sign is stored in bit 3 of the 4-bit code
E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def pack_mxfp4(x: torch.Tensor, group: int = 16):
    orig = x.shape
    xb = x.float().reshape(-1, group)
    maxabs = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = maxabs / 6.0
    q = xb / scale
    a = q.abs()
    idx = torch.bucketize(a, (E2M1_POS[:-1] + E2M1_POS[1:]) / 2.0).clamp(0, 7)
    sign = (q < 0).to(torch.uint8)
    codes = (idx.to(torch.uint8) | (sign << 3)).reshape(-1)
    packed = (codes[0::2] | (codes[1::2] << 4)).contiguous()
    return packed, scale, orig


def unpack_mxfp4(packed, scale, orig, group: int = 16):
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.empty(packed.shape[0] * 2, dtype=torch.uint8)
    codes[0::2] = lo
    codes[1::2] = hi
    idx = (codes & 0x07).long()
    sign = torch.where((codes >> 3).bool(), -1.0, 1.0)
    vals = E2M1_POS[idx] * sign * scale.expand(-1, group).reshape(-1)
    return vals.reshape(orig).to(torch.bfloat16)


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
    layers = out.past_key_values.layers
    orig_bytes = sum(l.keys.numel() + l.values.numel() for l in layers) * 2
    t0 = time.perf_counter()
    packed_layers = []
    for l in layers:
        pk, sk, shk = pack_mxfp4(l.keys)
        pv, sv, shv = pack_mxfp4(l.values)
        packed_layers.append((pk, sk, shk, pv, sv, shv))
    pack_s = time.perf_counter() - t0
    packed_bytes = sum(pk.numel() * pk.element_size() + sk.numel() * sk.element_size()
                       + pv.numel() * pv.element_size() + sv.numel() * sv.element_size()
                       for pk, sk, _, pv, sv, _ in packed_layers)
    t0 = time.perf_counter()
    rels = []
    for (pk, sk, shk, pv, sv, shv), l in zip(packed_layers, layers):
        k = unpack_mxfp4(pk, sk, shk); v = unpack_mxfp4(pv, sv, shv)
        rels.append(float((k - l.keys).norm() / l.keys.norm()))
        rels.append(float((v - l.values).norm() / l.values.norm()))
    unpack_s = time.perf_counter() - t0
    print('seq %d | BF16 KV %.2f MB | packed %.2f MB | ratio %.2fx'
          % (ids.shape[1], orig_bytes / 1e6, packed_bytes / 1e6,
             orig_bytes / packed_bytes), flush=True)
    print('pack %.2f s | unpack %.2f s | mean relL2 %.4f'
          % (pack_s, unpack_s, sum(rels) / len(rels)), flush=True)
    # bytes/token
    print('BF16 %.1f KB/token | packed %.1f KB/token'
          % (orig_bytes / ids.shape[1] / 1024, packed_bytes / ids.shape[1] / 1024),
          flush=True)


if __name__ == '__main__':
    main()
