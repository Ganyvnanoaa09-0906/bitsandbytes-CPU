"""probe_kv_fp8_scale.py — 4K/16K synthetic KV cache memory + per-layer unpack cost.

Qwen3-1.7B cache: 28 layers x 2 (K,V) x 8 heads x 128 dim. Real CPU prefill at 4K would
take ~10 min per mode, so first measure the storage/decode-overhead sides synthetically:
  * logical BF16 vs FP8 bytes,
  * process RSS while holding a packed FP8 cache,
  * per-layer unpack time (decode must pay this per layer per step).
"""
from __future__ import annotations

import time

import psutil
import torch


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(4)
    L, H, D = 28, 8, 128
    for T in (4096, 16384):
        bf16_bytes = L * 2 * H * T * D * 2
        fp8_bytes = L * 2 * H * T * D
        r0 = rss_gb()
        # Build FP8-packed cache layer by layer; keep only packed bytes.
        packed = []
        for _ in range(L):
            k = torch.randn(1, H, T, D, dtype=torch.bfloat16)
            v = torch.randn(1, H, T, D, dtype=torch.bfloat16)
            packed.append((k.to(torch.float8_e4m3fn).view(torch.uint8),
                           v.to(torch.float8_e4m3fn).view(torch.uint8)))
        r1 = rss_gb()
        # Unpack all layers sequentially (peak = one layer at a time).
        t0 = time.perf_counter()
        acc = 0.0
        for kb, vb in packed:
            k = kb.view(torch.float8_e4m3fn).to(torch.bfloat16)
            v = vb.view(torch.float8_e4m3fn).to(torch.bfloat16)
            acc += float(k[0, 0, 0, 0]) + float(v[0, 0, 0, 0])
        unpack_s = time.perf_counter() - t0
        r2 = rss_gb()
        print('T=%5d | BF16 %.2f GB | FP8 %.2f GB | RSS +%.2f GB (peak %.2f) | '
              'unpack all %.3f s (%.3f s/layer) | %.1f GB/s'
              % (T, bf16_bytes / 1e9, fp8_bytes / 1e9, r1 - r0, r2,
                 unpack_s, unpack_s / L, fp8_bytes / unpack_s / 1e9), flush=True)
        del packed, k, v
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    main()
