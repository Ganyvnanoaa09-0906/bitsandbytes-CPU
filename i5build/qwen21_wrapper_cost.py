"""How much of the 4-bit tax is bnb's Python wrapper rather than the arithmetic?

At (4126, 4096, 12288) the layer took 2024 ms where "dequantise + mm" took 1953 ms, so
about 70 ms per call is neither. Over the 224 quantised layers of this model that is
~16 s of a 350 s step -- small, but it is the only part of the step that is not already
at the machine's arithmetic ceiling, and it costs nothing to check whether it is
recoverable.

Three timings per shape:
    layer(x)              the module as the model calls it
    minimal forward       dequantize_4bit + F.linear, no checks, no casts
    mm on a warm weight    the arithmetic alone, as the floor
"""
import time

import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F

torch.set_num_threads(6)
torch.set_grad_enabled(False)

SHAPES = [(4126, 4096, 12288), (4126, 12288, 4096)]


def timed(fn, reps=4, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def build(k, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g) * 0.02
    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float32,
                              quant_type="nf4", compress_statistics=True)
    layer.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                     compress_statistics=True)
    return layer.to("cpu"), w


print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
print(f"{'shape (M,N,K)':<22} {'layer ms':>9} {'minimal ms':>11} {'warm mm ms':>11} "
      f"{'wrapper':>8}  {'agree':>6}")
for (m, n, k) in SHAPES:
    x = torch.randn(m, k)
    layer, w = build(k, n)
    w32 = w.clone()

    def minimal():
        wq = layer.weight
        return torch.nn.functional.linear(
            x, F.dequantize_4bit(wq.data, wq.quant_state), layer.bias)

    t_layer = timed(lambda: layer(x))
    t_min = timed(minimal)
    t_mm = timed(lambda: torch.mm(x, w32.t()))

    y1, y2 = layer(x), minimal()
    agree = float((y1 - y2).abs().max())

    print(f"  {m}x{n}x{k:<10} {t_layer*1000:>9.1f} {t_min*1000:>11.1f} {t_mm*1000:>11.1f} "
          f"{100*(t_layer-t_min)/t_layer:>7.1f}%  {agree:>6.1e}")
    del x, layer, w, w32
