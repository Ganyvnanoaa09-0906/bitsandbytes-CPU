"""bf16_decode.py -- can bnb quantize a bf16 tensor directly, and does bf16 input
still get the fast decode path?

Two questions that decide the streaming quantizer's memory ceiling.

1. quantize_4bit needs to not require a float32 input. The reference checks it does:
   `_dequantize_4bit_compute` accepts bf16 and `cquantize_blockwise_cpu_4bit_bf16`
   is exported. If quantizing straight from bf16 works, a 1.16 GB bf16 tensor never
   needs a 2.31 GB float32 twin, which is the difference between fitting the 9.28 GB
   transformer shard in 8.5 GB of free RAM and not fitting it.

2. the FP32_DEQUANT path in backends/cpu/ops.py does the arithmetic in double
   ("与 quantize_blockwise 默认用的同一个 => 数值一致"). Whether a bf16 input still
   takes the fast C kernel or falls back to that float64 path decides whether
   streaming quantization is minutes or hours.

Usage:
    python bf16_decode.py
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402

BLOCKSIZE = 64
QUANT_TYPE = "nf4"


def main():
    print(f"torch {torch.__version__}, bnb {bnb.__version__}")
    print()

    # The tensor that actually matters: the text encoder's token embedding, which is
    # the largest single tensor in the whole checkpoint.
    shape = (151936, 4096)
    print(f"timing on {shape} = {shape[0]*shape[1]/1e9:.3f}e9 weights "
          f"({shape[0]*shape[1]*2/1e9:.2f} GB in bf16)")
    torch.manual_seed(0)
    w_bf16 = torch.randn(*shape, dtype=torch.bfloat16)
    w_f32 = w_bf16.to(torch.float32)

    for label, w in (("bf16 input", w_bf16), ("fp32 input", w_f32)):
        try:
            t0 = time.time()
            wq, qs = bnb.functional.quantize_4bit(w, quant_type=QUANT_TYPE,
                                                  blocksize=BLOCKSIZE,
                                                  compress_statistics=True)
            dt = time.time() - t0
            n = w.numel()
            print(f"  {label:12s}: {dt:6.2f} s  ({n/dt/1e6:6.1f} M weights/s)  "
                  f"packed {wq.dtype} {tuple(wq.shape)}  state dtype {qs.dtype} "
                  f"nested={qs.nested}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:12s}: FAILED {type(e).__name__}: {str(e)[:150]}")
    print()

    # Do the two inputs give the same answer? bf16 has 8 explicit mantissa bits, so
    # quantizing from bf16 cannot equal quantizing from fp32 in general; the question
    # is whether the difference is within what bf16 storage already costs.
    try:
        wq_b, qs_b = bnb.functional.quantize_4bit(w_bf16, quant_type=QUANT_TYPE,
                                                  blocksize=BLOCKSIZE,
                                                  compress_statistics=True)
        wq_f, qs_f = bnb.functional.quantize_4bit(w_f32, quant_type=QUANT_TYPE,
                                                  blocksize=BLOCKSIZE,
                                                  compress_statistics=True)
        dq_b = bnb.functional.dequantize_4bit(wq_b, qs_b).float()
        dq_f = bnb.functional.dequantize_4bit(wq_f, qs_f).float()
        ref = w_f32
        def rel(a, b):
            return ((a - b).norm() / b.norm()).item()
        print("quantization error against the fp32 original:")
        print(f"  from fp32 : {rel(dq_f, ref):.6f}")
        print(f"  from bf16 : {rel(dq_b, ref):.6f}")
        print(f"  bf16 storage alone (no quantization): "
              f"{rel(w_bf16.float(), ref):.6f}")
        print()
        print("  if 'from bf16' is close to 'bf16 storage alone', the bf16 input is")
        print("  not what dominates the error and quantizing from bf16 is fine.")
    except Exception as e:  # noqa: BLE001
        print(f"comparison FAILED {type(e).__name__}: {str(e)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
