"""probe_igpu_memory_breakdown.py — where does the iGPU path's extra ~3.6GB live?

Loads the 4-bit transformer, measures RSS, wraps all 232 linears for the iGPU fused path,
then measures RSS again. If RSS jumps by ~3.4GB at wrap time, the GPU buffers are charged
to the process working set (WDDM shared memory); if not, the extra appears during forward.
"""
from __future__ import annotations

import gc
import sys

import psutil
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer  # noqa: E402
from qwen4bit_igpu_fused import to_igpu_fused  # noqa: E402


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(4)
    print('before load  RSS %.2f GB | sys free %.2f GB'
          % (rss_gb(), psutil.virtual_memory().available / 1e9), flush=True)
    model = load_transformer()
    gc.collect()
    r1 = rss_gb()
    print('after load   RSS %.2f GB | sys free %.2f GB'
          % (r1, psutil.virtual_memory().available / 1e9), flush=True)
    n = to_igpu_fused(model)
    gc.collect()
    r2 = rss_gb()
    print('after wrap   RSS %.2f GB (delta %+.2f) | sys free %.2f GB | %d linears'
          % (r2, r2 - r1, psutil.virtual_memory().available / 1e9, n), flush=True)
    # Count host-side storages still referenced by the model's wrappers.
    host_bytes = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ == 'GPU4BitLinearFused':
            for attr in ('absmax_host', 'code_host'):
                v = getattr(mod, attr, None)
                if v is not None:
                    host_bytes += v.nbytes
    print('host-side wrapper arrays %.2f MB' % (host_bytes / 1e6), flush=True)


if __name__ == '__main__':
    main()
