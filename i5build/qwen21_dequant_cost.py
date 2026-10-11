"""Which half of the 4-bit tax is the expensive one: the unpacking, or the matmul?

The tax measured at the model's shapes is 5.5-14.5%, and it has two possible sources
with completely different fixes:

  * the dequantise kernel is slow (a scalar gather over a 16-entry LUT would run at
    ~1 GB/s where the machine can stream ~14 GB/s) -> fixable in the C kernel
  * the dense matmul after it is slower than the pure-fp32 benchmark (oneDNN packing a
    freshly written 200 MB weight again, cache pressure from a just-written buffer)
    -> not fixable by touching the dequantise

So this times the pieces separately on the model's own shapes:
    dequant only        F.dequantize_4bit(q, state)
    mm on a warm weight torch.mm(x, w32.t()) where w32 already exists
    both, in that order dequant then mm -- what the layer actually does
A dequantise that is much slower than 226 MB / bandwidth would be the actionable one.
"""
import time

import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F

torch.set_num_threads(6)
torch.set_grad_enabled(False)

SHAPES = [(4126, 4096, 12288), (4126, 12288, 4096), (1054, 12288, 4096)]


def timed(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
print(f"{'shape (M,N,K)':<22} {'dequant ms':>10} {'GB/s out':>9} {'mm ms':>9} "
      f"{'both ms':>9} {'layer ms':>9}")
for (m, n, k) in SHAPES:
    x = torch.randn(m, k)
    g = torch.Generator().manual_seed(0)
    w = torch.randn(n, k, generator=g) * 0.02
    q, state = F.quantize_4bit(w, quant_type="nf4", blocksize=64,
                               compress_statistics=True)
    w32 = F.dequantize_4bit(q, state).contiguous()          # a warm fp32 weight

    t_dq = timed(lambda: F.dequantize_4bit(q, state))
    t_mm = timed(lambda: torch.mm(x, w32.t()))
    t_both = timed(lambda: torch.mm(x, F.dequantize_4bit(q, state).t()))

    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float32,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(w.clone(), requires_grad=False, quant_type="nf4",
                                     compress_statistics=True)
    layer = layer.to("cpu")
    t_layer = timed(lambda: layer(x))

    out_bytes = n * k * 4
    print(f"  {m}x{n}x{k:<10} {t_dq*1000:>10.1f} {out_bytes/t_dq/1e9:>9.2f} "
          f"{t_mm*1000:>9.1f} {t_both*1000:>9.1f} {t_layer*1000:>9.1f}")
    del x, w, q, state, w32, layer
