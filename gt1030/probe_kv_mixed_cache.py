"""probe_kv_mixed_cache.py — mixed-precision KV cache on Qwen3-1.7B.

K is attention-score sensitive, V is not. Test:
  * k8v4: K stored FP8 E4M3, V stored MXFP4 E2M1
  * k4v8: K stored MXFP4, V stored FP8
against BF16 cache over 8 greedy decode tokens.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from probe_kv_fp4_storage import pack_mxfp4, unpack_mxfp4  # noqa: E402

MODEL = r'D:\work\llm_env\Qwen3-1.7B'


class _L:
    def __init__(self):
        self.kf = self.vf = None
        self.ks = self.vs = None      # fp4 scales (None for fp8)
        self.k_fp8 = self.v_fp8 = True
        self.shape = None
        self.seq = 0


class MixedCache:
    def __init__(self, mode='k8v4'):
        self.mode = mode
        self.layers = []

    def _layer(self, idx):
        while len(self.layers) <= idx:
            self.layers.append(_L())
        return self.layers[idx]

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].seq if layer_idx < len(self.layers) else 0

    def get_query_offset(self, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length, layer_idx=0):
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].seq + query_length, 0

    @staticmethod
    def _quant(x, fp8):
        if fp8:
            return x.detach().reshape(-1).to(torch.float8_e4m3fn).view(torch.uint8), None
        p, s, _ = pack_mxfp4(x)
        return p, s

    @staticmethod
    def _dequant(flat, scale, shape, fp8):
        if fp8:
            return flat.view(torch.float8_e4m3fn).to(torch.bfloat16).reshape(shape)
        return unpack_mxfp4(flat, scale, shape)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        lay = self._layer(layer_idx)
        B, H, T, D = key_states.shape
        k_fp8 = self.mode == 'k8v4'
        kq, ks = self._quant(key_states, k_fp8)
        vq, vs = self._quant(value_states, not k_fp8)
        if lay.kf is None:
            lay.kf, lay.ks, lay.k_fp8 = kq, ks, k_fp8
            lay.vf, lay.vs, lay.v_fp8 = vq, vs, not k_fp8
            lay.shape, lay.seq = (B, H, T, D), T
            return key_states, value_states
        k_old = self._dequant(lay.kf, lay.ks, (B, H, lay.seq, D), lay.k_fp8)
        v_old = self._dequant(lay.vf, lay.vs, (B, H, lay.seq, D), lay.v_fp8)
        lay.kf = torch.cat([lay.kf, kq])
        lay.vf = torch.cat([lay.vf, vq])
        if lay.ks is not None:
            lay.ks = torch.cat([lay.ks, ks])
        if lay.vs is not None:
            lay.vs = torch.cat([lay.vs, vs])
        lay.seq += T
        return torch.cat([k_old, key_states], dim=2), torch.cat([v_old, value_states], dim=2)


def main():
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cpu',
        low_cpu_mem_usage=True, local_files_only=True).eval()
    prompt = ('The key idea behind KV cache compression for long-context inference is '
              'to reduce the memory footprint while preserving attention quality. ' * 4)
    ids = tok(prompt, return_tensors='pt').input_ids

    def run(cache):
        with torch.no_grad():
            o = model(ids, past_key_values=cache, use_cache=True)
            cur = o.logits[:, -1:].argmax(-1)
            toks = []
            for _ in range(8):
                o = model(cur, past_key_values=cache, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1)
                toks.append(int(cur))
        return toks

    base = run(DynamicCache())
    print('BF16      ', base, flush=True)
    for mode in ('k8v4', 'k4v8'):
        toks = run(MixedCache(mode))
        match = sum(a == b for a, b in zip(base, toks))
        print('%-5s     ' % mode, toks, 'match %d/%d' % (match, len(base)), flush=True)


if __name__ == '__main__':
    main()
