"""probe_kv_fp4_cache.py — real MXFP4 KV cache decode on Qwen3-1.7B.

Implements a minimal cache that stores only nibble-packed K/V + fp16 per-16 scales, and
dequantizes one layer at a time inside update() (peak BF16 = one layer, not the whole cache).
HF Qwen3 attention only uses cache.update() and cache.get_seq_length(), so duck typing works.

Compares greedy tokens, RSS and decode throughput against a normal BF16 DynamicCache.
"""
from __future__ import annotations

import sys
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from probe_kv_fp4_storage import pack_mxfp4, unpack_mxfp4  # noqa: E402

MODEL = r'D:\work\llm_env\Qwen3-1.7B'


class QLayer:
    def __init__(self):
        self.kp = self.vp = None      # packed uint8
        self.ks = self.vs = None      # fp16 scales
        self.shape = None             # (B,H,T,D)
        self.seq = 0


class QuantizedKVCache:
    def __init__(self, mode='mxfp4', group=16):
        self.layers: list[QLayer] = []
        self.mode = mode
        self.group = group

    def _get(self, idx):
        while len(self.layers) <= idx:
            self.layers.append(QLayer())
        return self.layers[idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx >= len(self.layers) or self.layers[layer_idx] is None:
            return 0
        return self.layers[layer_idx].seq

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx=0):
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].seq + query_length, 0

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        lay = self._get(layer_idx)
        B, H, T, D = key_states.shape
        if self.mode == 'identity':
            if lay.kp is None:
                lay.kp, lay.shape, lay.seq = key_states.detach(), (B, H, T, D), T
                lay.vp = value_states.detach()
            else:
                lay.kp = torch.cat([lay.kp, key_states.detach()], dim=2)
                lay.vp = torch.cat([lay.vp, value_states.detach()], dim=2)
                lay.seq += T
            return lay.kp, lay.vp
        if self.mode == 'fp8':
            kb = key_states.detach().to(torch.float8_e4m3fn).view(torch.uint8)
            vb = value_states.detach().to(torch.float8_e4m3fn).view(torch.uint8)
            if lay.kp is None:
                lay.kp, lay.shape, lay.seq = kb, (B, H, T, D), T
                lay.vp = vb
                return key_states, value_states
            k_old = lay.kp.view(torch.float8_e4m3fn).to(torch.bfloat16)
            v_old = lay.vp.view(torch.float8_e4m3fn).to(torch.bfloat16)
            lay.kp = torch.cat([lay.kp, kb], dim=2); lay.vp = torch.cat([lay.vp, vb], dim=2)
            lay.seq += T
            return torch.cat([k_old, key_states], dim=2), torch.cat([v_old, value_states], dim=2)
        pk, sk, shk = pack_mxfp4(key_states, group=self.group)
        pv, sv, shv = pack_mxfp4(value_states, group=self.group)
        if lay.kp is None:
            lay.kp, lay.ks, lay.shape, lay.seq = pk, sk, shk, T
            lay.vp, lay.vs = pv, sv
            return key_states, value_states
        k_old = unpack_mxfp4(lay.kp, lay.ks, (B, H, lay.seq, D), group=self.group)
        v_old = unpack_mxfp4(lay.vp, lay.vs, (B, H, lay.seq, D), group=self.group)
        lay.kp = torch.cat([lay.kp, pk]); lay.ks = torch.cat([lay.ks, sk])
        lay.vp = torch.cat([lay.vp, pv]); lay.vs = torch.cat([lay.vs, sv])
        lay.seq += T
        return torch.cat([k_old, key_states], dim=2), torch.cat([v_old, value_states], dim=2)


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    prompt = ('The key idea behind KV cache compression for long-context inference is '
              'to reduce the memory footprint while preserving attention quality. ' * 4)
    ids = tok(prompt, return_tensors='pt').input_ids

    def run(cache, label, steps=8):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(ids, past_key_values=cache, use_cache=True)
            pre = time.perf_counter() - t0
            cur = out.logits[:, -1:].argmax(-1)
            toks = []
            t0 = time.perf_counter()
            for _ in range(steps):
                o = model(cur, past_key_values=cache, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1)
                toks.append(int(cur))
            gen_s = time.perf_counter() - t0
        print('%s | prefill %.2f s | decode %.3f s/token | RSS %.2f GB | tokens %s'
              % (label, pre, gen_s / steps, rss_gb(), toks[:6]), flush=True)
        return toks

    base = run(DynamicCache(), 'BF16')
    ident = run(QuantizedKVCache(mode='identity'), 'identity-cache')
    print('identity match', sum(a == b for a, b in zip(base, ident)), '/', len(base), flush=True)
    fp8c = QuantizedKVCache(mode='fp8')
    f8 = run(fp8c, 'FP8')
    print('fp8 match', sum(a == b for a, b in zip(base, f8)), '/', len(base), flush=True)
    qcache = QuantizedKVCache(mode='mxfp4')
    q = run(qcache, 'MXFP4')
    match = sum(a == b for a, b in zip(base, q))
    print('token match %d/%d' % (match, len(base)), flush=True)
    layers = qcache.layers
    packed = sum(l.kp.numel() + l.ks.numel() * 2 + l.vp.numel() + l.vs.numel() * 2
                 for l in layers)
    bf16 = sum(l.shape[0] * l.shape[1] * l.seq * l.shape[3] * 2 * 2 for l in layers)
    print('KV packed %.2f MB vs BF16 %.2f MB | ratio %.2fx'
          % (packed / 1e6, bf16 / 1e6, bf16 / packed), flush=True)


if __name__ == '__main__':
    main()
