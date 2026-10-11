"""The 4-bit tax at the shapes this model actually calls.

The probe measured `bitsandbytes::dequantize_4bit` at 5% of the step and the
`gemm_4bit` dispatcher at another 3%. Both are pure overhead of using packed weights
for a large-M matmul: the weight is unpacked to fp32 and then multiplied densely, so
the arithmetic is identical to keeping fp32 weights and only the memory footprint
improves. This measures that tax directly, one layer at a time, so the ceiling on what
a fused kernel could recover is a number rather than an impression.

Three configurations per shape:
    fp32        a plain fp32 weight, torch.mm -- the arithmetic alone
    dequant     bnb dequantize_4bit + torch.mm -- what the model does today
    Linear4bit  the module, end to end -- adds bnb's Python dispatch and bookkeeping

Usage:
    python qwen21_4bit_tax.py
"""
import time

import torch
import torch.nn as nn

import bitsandbytes as bnb

torch.set_num_threads(6)
torch.set_grad_enabled(False)

SHAPES = [(1054, 12288, 4096), (1054, 4096, 12288), (4126, 12288, 4096)]


def timed(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def make_4bit(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.02
    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float32,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                     compress_statistics=True)
    return layer.to("cpu"), w


print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
print(f"{'shape (M,N,K)':<24} {'fp32 ms':>9} {'dequant+mm':>11} {'Linear4bit':>11} "
      f"{'tax':>7}  {'GFLOPS 4bit':>11}")
for (m, n, k) in SHAPES:
    x = torch.randn(m, k)
    layer, w = make_4bit(n, k)
    w32 = w.clone()

    t_fp32 = timed(lambda: torch.mm(x, w32.t()))

    wq = layer.weight
    # what the module does internally, spelled out
    from bitsandbytes import functional as F
    q, state = F.quantize_4bit(w32, quant_type="nf4", blocksize=64,
                               compress_statistics=True)
    t_dq = timed(lambda: torch.mm(x, F.dequantize_4bit(q, state).t()))

    t_mod = timed(lambda: layer(x))

    flops = 2.0 * m * n * k
    print(f"  {m}x{n}x{k:<12} {t_fp32*1000:>9.1f} {t_dq*1000:>11.1f} "
          f"{t_mod*1000:>11.1f} {100*(t_mod-t_fp32)/t_mod:>6.1f}% "
          f"{flops/t_mod/1e9:>11.1f}")
    del x, layer, w, w32, q, state
