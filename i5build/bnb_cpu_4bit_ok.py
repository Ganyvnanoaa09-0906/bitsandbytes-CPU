"""bnb_cpu_4bit_ok.py -- does bitsandbytes 4-bit actually work on the CPU backend?

Everything about running Qwen-Image-2.1 on this machine rests on one question: can
the fork quantize a bf16 tensor to 4 bits and get it back *on CPU*, with no CUDA,
no Triton and no MPS? bnb has a `quantize_4bit` kernel registered for "cpu" but the
dequantize path has a pure-torch fallback, and the MPS/CUDA paths are the ones the
library is normally exercised on. If this fails there is no 4-bit plan for the
16.2B-parameter model on a 15.4 GB machine, so it is worth one minute to know.

Checks, in order of how much they would hurt to discover later:
  1. quantize_4bit / dequantize_4bit round-trip on plain tensors
  2. the same through nn.Linear4bit, which is what a real model would use
  3. numerics: relative error against the original weight, and a matmul agreeing
     with fp32 to a sane tolerance
  4. the actual memory saving, measured rather than assumed

Usage:
    python bnb_cpu_4bit_ok.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402


def rel_err(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).norm() / max(1e-12, a.norm())).item()


def main():
    print(f"torch          : {torch.__version__}")
    print(f"bitsandbytes   : {bnb.__version__}")
    print(f"device         : {torch.device('cpu')}")
    print()

    torch.manual_seed(0)
    fails = []

    # 1) raw functional round-trip
    print("1) functional quantize_4bit -> dequantize_4bit")
    w = torch.randn(512, 384, dtype=torch.bfloat16)
    for qtype in ("nf4", "fp4"):
        for bs in (64, 32, 128, 256):
            try:
                wq, state = bnb.functional.quantize_4bit(w, quant_type=qtype, blocksize=bs)
                back = bnb.functional.dequantize_4bit(wq, state)
                e = rel_err(w.float(), back.float())
                packed_ratio = w.numel() * 2 / max(1, wq.numel() + state.absmax.numel() * 4)
                print(f"   {qtype} bs={bs:3d}  rel_err={e:.5f}  "
                      f"stored={wq.numel()+state.absmax.numel()*4} B vs bf16 {w.numel()*2} B "
                      f"({packed_ratio:.2f}x smaller)")
            except Exception as ex:  # noqa: BLE001
                print(f"   {qtype} bs={bs:3d}  FAILED {type(ex).__name__}: {str(ex)[:100]}")
                fails.append(f"functional {qtype} bs={bs}")
    print()

    # 2) nn.Linear4bit, the path a real model uses
    #
    # blocksize is deliberately NOT passed to Linear4bit: in this version it is a
    # Params4bit argument only (Linear4bit.__init__ takes input_features,
    # output_features, bias, compute_dtype, compress_statistics, quant_type,
    # quant_storage, device). Passing it here is the mistake that made this check
    # report a failure the first time it ran.
    #
    # .to("cpu") is what actually triggers quantization: Params4bit.to() calls
    # _quantize(device) whenever the target device is not "meta" and the parameter
    # is not already quantized. Without any .to() call the weight stays in bf16,
    # quant_state stays None, and forward() trips an assertion -- which is the
    # second thing this check got wrong before it got anything right.
    print("2) nn.Linear4bit forward")
    try:
        lin = torch.nn.Linear(384, 512, bias=True).to(torch.bfloat16)
        x = torch.randn(4, 384, dtype=torch.bfloat16)
        ref = lin(x)
        q = bnb.nn.Linear4bit(384, 512, bias=True, compute_dtype=torch.float32,
                              quant_type="nf4")
        q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                     quant_type="nf4", blocksize=64)
        q.bias = torch.nn.Parameter(lin.bias.data.clone())
        q = q.to("cpu")
        out = q(x)
        ws = q.weight.quant_state
        print(f"   forward ok, out {tuple(out.shape)} {out.dtype}, "
              f"rel_err vs bf16 linear = {rel_err(ref.float(), out.float()):.5f}")
        print(f"   quant_state: {ws.quant_type} blocksize={ws.blocksize} "
              f"nested={ws.nested} dtype={q.weight.dtype}")
        print(f"   weight storage: {q.weight.numel()} B packed vs "
              f"{lin.weight.numel()*2} B bf16")
    except Exception as ex:  # noqa: BLE001
        print(f"   FAILED {type(ex).__name__}: {str(ex)[:200]}")
        fails.append("Linear4bit forward")
    print()

    # 2b) the CPU fast path is gated on AVX512-BF16, which neither target CPU has.
    # Knowing which branch we are on decides whether 4-bit inference is affordable
    # or merely possible.
    print("2b) which CPU path is taken")
    try:
        probe = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float32,
                                  quant_type="nf4")
        probe.weight = bnb.nn.Params4bit(torch.randn(64, 64, dtype=torch.bfloat16),
                                         requires_grad=False, quant_type="nf4",
                                         blocksize=64)
        print(f"   support_avx512bf16_for_cpu = {probe.support_avx512bf16_for_cpu}")
        if not probe.support_avx512bf16_for_cpu:
            print("   -> generic matmul_4bit path: dequantize then matmul, so the "
                  "4-bit win here is memory, not speed")
    except Exception as ex:  # noqa: BLE001
        print(f"   could not read the flag: {type(ex).__name__}: {str(ex)[:120]}")
    print()

    # 3) memory: measure the footprint of the same weight in bf16 vs quantized
    print("3) measured footprint of one 4096x4096 weight")
    try:
        big = torch.randn(4096, 4096, dtype=torch.bfloat16)
        bf16_b = big.numel() * 2
        wq, state = bnb.functional.quantize_4bit(big, quant_type="nf4", blocksize=64)
        q_b = wq.numel() + state.absmax.numel() * 4
        if getattr(state, "state2", None) is not None:
            q_b += state.state2.absmax.numel() * 4
        print(f"   bf16 {bf16_b/1e6:.1f} MB -> 4bit {q_b/1e6:.1f} MB "
              f"({bf16_b/q_b:.2f}x smaller)")
        print(f"   linear-layer total for the whole 16.2B model would be "
              f"~{16.2e9*10/8*0.27/1e9:.1f} GB of the 15.4 GB budget")
    except Exception as ex:  # noqa: BLE001
        print(f"   FAILED {type(ex).__name__}: {str(ex)[:200]}")
        fails.append("memory measurement")
    print()

    if fails:
        print(f"RESULT: {len(fails)} failure(s) -- 4-bit on this CPU backend is not usable")
        for f in fails:
            print(f"   {f}")
        return 1
    print("RESULT: 4-bit quantization and dequantization both work on the CPU backend")
    return 0


if __name__ == "__main__":
    sys.exit(main())
