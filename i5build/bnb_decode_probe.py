"""Why is 4-bit LLM decode SLOWER than bf16 on this CPU?

The full comparison reported 2.28 tok/s in bf16 against 0.67 tok/s with bnb's 4-bit
Linear on the same Qwen3-1.7B, which is backwards: at batch 1 a 4-bit weight is a
quarter of the bytes to stream, and this project's own report records the fused CPU GEMV
beating fp32 by ~2x at M=1. Either the command line is not reaching the fused kernel, or
something else in the step is paying for the quantisation.

This measures one layer, at M=1, three ways -- fp32, bf16 and bnb Linear4bit -- at the
two shapes that dominate a decode step:

    q_proj   2048 -> 2048     the per-layer cost, 28 of these per token
    lm_head  2048 -> 151936   once per token, and the largest GEMM in the model

The LM head matters twice over: it is the biggest single matmul in a decode step, and
`replace_linears` quantises every nn.Linear it finds, so if the head is included the
comparison is measuring that too.

Usage:
    python bnb_decode_probe.py
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)
torch.set_num_threads(6)


def bench(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    print(f"torch {torch.__version__}  bnb {bnb.__version__}  "
          f"threads {torch.get_num_threads()}")

    shapes = [("q_proj  ", 2048, 2048), ("lm_head ", 151936, 2048),
              ("mlp down", 2048, 6144)]
    print(f"\n{'layer':<10} {'shape':>16} {'bf16 us':>10} {'4bit us':>10} "
          f"{'ratio':>7} {'4bit B/w':>9}")
    for label, out_f, in_f in shapes:
        x = torch.randn(1, in_f, dtype=torch.bfloat16)
        lin = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            lin.weight.normal_(0, 0.02)
        w_bf16 = lin.weight.detach().to(torch.bfloat16)

        q = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.bfloat16,
                       quant_type="nf4", compress_statistics=True)
        q.weight = Params4bit(lin.weight.data, requires_grad=False,
                              quant_type="nf4", compress_statistics=True)
        q.weight = q.weight.to("cpu")

        t_bf16 = bench(lambda: F.linear(x, w_bf16))
        t_4bit = bench(lambda: q(x))
        bpw = q.weight.numel() / lin.weight.numel()
        print(f"{label:<10} {f'{out_f}x{in_f}':>16} {t_bf16*1e6:10.1f} "
              f"{t_4bit*1e6:10.1f} {t_4bit/t_bf16:7.2f} {bpw:9.3f}")
        del q, lin, w_bf16
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n=== is the fused GEMV kernel actually registered and taken? ===")
    try:
        from bitsandbytes.backends.cpu.ops import _get_cpu_kernel  # noqa: F401
    except Exception:
        pass
    import bitsandbytes.backends.cpu.ops as cpuops  # noqa: F401
    print("  gemv_4bit registered for cpu:",
          hasattr(torch.ops.bitsandbytes, "gemv_4bit"))
    try:
        d = torch.ops.bitsandbytes.gemv_4bit.default
        print("  gemv_4bit default:", d)
    except Exception as exc:  # noqa: BLE001
        print(f"  no gemv_4bit default: {type(exc).__name__}: {str(exc)[:60]}")

    print("\n=== the same layer, one token, one call: where does the time go ===")
    in_f = out_f = 2048
    x = torch.randn(1, in_f, dtype=torch.bfloat16)
    lin = nn.Linear(in_f, out_f, bias=False)
    q = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.bfloat16,
                   quant_type="nf4", compress_statistics=True)
    q.weight = Params4bit(lin.weight.data, requires_grad=False, quant_type="nf4",
                          compress_statistics=True)
    q.weight = q.weight.to("cpu")

    from bitsandbytes.functional import dequantize_4bit
    t_deq = bench(lambda: dequantize_4bit(q.weight.data, q.weight.quant_state), reps=10)
    print(f"  dequantize_4bit alone : {t_deq*1e6:9.1f} us")
    print(f"  full 4-bit forward    : {bench(lambda: q(x))*1e6:9.1f} us")
    print("  if the forward costs about the dequantise plus a dense GEMM, the fused")
    print("  GEMV is not being used and the packed weight is only saving memory.")


if __name__ == "__main__":
    main()
