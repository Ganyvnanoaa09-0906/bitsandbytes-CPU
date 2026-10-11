"""Should a quantized diffusion model compute in 4 bits, or unpack and compute in fp32?

The report recorded a result that is easy to forget when the model is a diffusion
transformer rather than an LLM: this fork's 4-bit GEMM is fast where an LLM lives and
slow where a diffusion model lives. At M=1 (decode) it beats fp32 by ~2x, and at M=512
it loses by 6.3x, because the kernel is built to stream weights against a single token.

Diffusion is the opposite regime. Qwen-Image-2.1 at 1024x1024 is 16384 tokens per
forward, so every GEMM is enormous in M and the weight traffic that 4-bit saves is not
what the machine is waiting on.

That raises a concrete question with a large payoff: if the weights are stored 4-bit
only to fit in memory, is it faster to unpack one layer at a time and run an ordinary
fp32 GEMM through oneDNN? The memory argument is unaffected -- one layer is unpacked at
a time -- and the compute argument may flip entirely.

Measured against the real layer shapes from Qwen-Image-2.1's transformer config.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn import Linear4bit, Params4bit

torch.set_grad_enabled(False)


def bench(fn, reps=3, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--shapes", default="4096x4096,4096x12288,12288x4096,4096x2048")
    ap.add_argument("--ms", default="1088,4096,16384")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    shapes = []
    for s in a.shapes.split(","):
        o, i = s.lower().split("x")
        shapes.append((int(o), int(i)))
    ms = [int(x) for x in a.ms.split(",")]

    print(f"Qwen-Image-2.1 transformer shapes, threads={a.threads}")
    print(f"{'shape':>12} {'M':>6} {'4bit ms':>9} {'unpack+fp32':>12} {'fp32 ms':>9} "
          f"{'4bit/unpack':>12} {'4bit/fp32':>10} {'exact':>7}")
    for out_f, in_f in shapes:
        ref = nn.Linear(in_f, out_f, bias=True)
        with torch.no_grad():
            ref.weight.normal_(0, 0.02)
        q = Linear4bit(in_f, out_f, bias=True, compute_dtype=torch.float32,
                       quant_type="nf4", compress_statistics=True)
        with torch.no_grad():
            q.weight = Params4bit(ref.weight.data, requires_grad=False,
                                  quant_type="nf4", compress_statistics=True)
            q.weight = q.weight.to("cpu")
            q.bias.data.copy_(ref.bias.data)
        # the packed weight and its state, which is what actually has to be held
        wq, st = q.weight.data, q.weight.quant_state

        for M in ms:
            x = torch.randn(M, in_f)
            t4 = bench(lambda: q(x))
            # unpack this one layer back to fp32 and let oneDNN do the GEMM
            def unpack_fp32():
                w = dequantize_4bit(wq, st)
                return F.linear(x, w.float(), ref.bias)
            try:
                tup = bench(unpack_fp32)
                up_ok = True
            except Exception as exc:  # noqa: BLE001
                tup = float("nan")
                up_ok = False
                up_err = f"{type(exc).__name__}: {str(exc)[:60]}"
            tf = bench(lambda: ref(x))

            exact = ""
            if up_ok:
                y4 = q(x)
                yf = ref(x)
                rel = ((y4 - yf).norm() / yf.norm()).item()
                exact = f"{rel:.3f}"
            print(f"{out_f:5d}x{in_f:<6d} {M:6d} {t4*1e3:9.1f} "
                  f"{tup*1e3:12.1f} {tf*1e3:9.1f} {t4/tup:12.2f} {t4/tf:10.2f} "
                  f"{exact:>7}" + ("" if up_ok else f"   unpack FAILED: {up_err}"))
        print(flush=True)

    print("\nratio > 1 means the 4-bit kernel is SLOWER than the alternative.")
    print("'4bit/fp32' is the speed the same computation would run at if weights")
    print("were fp32 -- which they cannot be, since the model does not fit -- so it")
    print("bounds what a faster dequant path could recover, not a real option.")


if __name__ == "__main__":
    main()
