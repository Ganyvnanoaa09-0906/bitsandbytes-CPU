"""chunked_quant_test.py -- is chunked quantization bit-identical to whole-tensor?

quantize_4bit_chunked() exists to get the fp32 fast path (64.9 M weights/s) without
the fp32 memory cost (a 2.49 GB allocation, which failed outright when measured).
The claim it has to earn is narrow and testable: quantizing a tensor in row-chunks,
converting each chunk to fp32 in turn, produces EXACTLY the same packed bytes and
the same absmax as quantizing the whole tensor at once.

That should hold because blocks never straddle a chunk boundary -- blocksize divides
the row length for every tensor in this model -- and because the blockwise quantizer
is elementwise-local to each block. "Should" is not a result, so this checks the
packed bytes and the absmax with torch.equal, and additionally dequantizes both and
compares the outputs. A chunked path that is merely close would silently change the
checkpoint.

It also compares against the bf16 path, which is the one the first quantizer used:
if chunked fp32 matches whole-tensor fp32 but not bf16, that is expected; if it
matches neither, the chunking is broken.

Usage:
    python chunked_quant_test.py
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402
import qwen21_quant as Q  # noqa: E402

BLOCKSIZE = 64


def compare(label, a, b):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        print(f"    {label}: SHAPE DIFFERS {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    same = torch.equal(a, b)
    print(f"    {label}: {'IDENTICAL' if same else 'DIFFERS'}"
          + ("" if same else f"  max|d|={(a.float()-b.float()).abs().max().item():.6e}"))
    return same


def case(rows, cols, budget_bytes, label):
    print(f"  {label}: {rows}x{cols} bf16 = {rows*cols*2/1e6:.2f} MB, "
          f"budget {budget_bytes/1e6:.0f} MB")
    torch.manual_seed(0)
    w = torch.randn(rows, cols, dtype=torch.bfloat16)

    q_whole, s_whole = bnb.functional.quantize_4bit(
        w.to(torch.float32), quant_type="nf4", blocksize=BLOCKSIZE,
        compress_statistics=True)
    q_bf16, s_bf16 = bnb.functional.quantize_4bit(
        w, quant_type="nf4", blocksize=BLOCKSIZE, compress_statistics=True)

    q_ch, s_ch, path = Q.quantize_4bit_chunked(w, BLOCKSIZE, budget_bytes)
    print(f"    path taken: {path}")

    ok = True
    ok &= compare("packed vs whole-fp32", q_ch, q_whole) is not False
    ok &= compare("absmax vs whole-fp32", s_ch.absmax, s_whole.absmax) is not False
    ok &= compare("code   vs whole-fp32", s_ch.code, s_whole.code) is not False
    print(f"    shape: chunked={tuple(s_ch.shape)} whole={tuple(s_whole.shape)} "
          f"-> {'ok' if tuple(s_ch.shape) == tuple(s_whole.shape) else 'MISMATCH'}")
    ok &= tuple(s_ch.shape) == tuple(s_whole.shape)

    d_ch = bnb.functional.dequantize_4bit(q_ch, s_ch)
    d_wh = bnb.functional.dequantize_4bit(q_whole, s_whole)
    ok &= compare("dequantized vs whole-fp32", d_ch, d_wh) is not False

    # The bf16 path is a different input dtype, so it is not required to match; it is
    # printed to show how much the dtype choice moves the result.
    d_bf = bnb.functional.dequantize_4bit(q_bf16, s_bf16)
    e_ch = ((d_ch.float() - w.float()).norm() / w.float().norm()).item()
    e_bf = ((d_bf.float() - w.float()).norm() / w.float().norm()).item()
    print(f"    rel err vs bf16 source: chunked-fp32 {e_ch:.6f}   whole-bf16 {e_bf:.6f}")
    print(f"    => {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main():
    print(f"torch {torch.__version__}")
    print(f"blocksize {BLOCKSIZE}, nf4, compress_statistics=True")
    print()
    allok = True

    # A budget large enough for everything: must take the whole-tensor path.
    allok &= case(512, 4096, 1 << 40, "budget exceeds the tensor")
    # A budget forcing several chunks.
    allok &= case(512, 4096, 40 << 20, "budget forces chunking")
    # Real tensor shapes from the checkpoint.
    allok &= case(12288, 4096, 300 << 20, "img_mlp.proj shape, small budget")
    allok &= case(4096, 4096, 120 << 20, "to_q shape, small budget")
    # A row length that is not a multiple of the blocksize would break the
    # no-straddling assumption, so it is exercised to see what happens.
    print("  adversarial: row length NOT a multiple of blocksize")
    try:
        w = torch.randn(256, 100, dtype=torch.bfloat16)
        q, s, path = Q.quantize_4bit_chunked(w, BLOCKSIZE, 20 << 20)
        print(f"    path {path}, packed {tuple(q.shape)}, "
              f"dequantizes to {tuple(bnb.functional.dequantize_4bit(q, s).shape)}")
        print("    (a row length of 100 with blocksize 64 means blocks DO straddle "
              "rows; the chunked path is not valid for such shapes and the quantizer "
              "must not be pointed at one)")
    except Exception as e:  # noqa: BLE001
        print(f"    raised {type(e).__name__}: {str(e)[:120]}")
    print()

    # Speed and peak memory, which is the whole point of the exercise.
    print("speed and memory on a 151936x4096 tensor (the largest in the checkpoint)")
    torch.manual_seed(0)
    big = torch.randn(151936, 4096, dtype=torch.bfloat16)
    for label, fn in (
            ("bf16 whole", lambda: bnb.functional.quantize_4bit(
                big, quant_type="nf4", blocksize=BLOCKSIZE, compress_statistics=True)),
            ("fp32 whole", lambda: bnb.functional.quantize_4bit(
                big.to(torch.float32), quant_type="nf4", blocksize=BLOCKSIZE,
                compress_statistics=True)),
            ("fp32 chunked 256 MB", lambda: Q.quantize_4bit_chunked(
                big, BLOCKSIZE, 256 << 20)),
            ("fp32 chunked 64 MB", lambda: Q.quantize_4bit_chunked(
                big, BLOCKSIZE, 64 << 20)),
    ):
        try:
            t0 = time.time()
            r = fn()
            dt = time.time() - t0
            path = r[2] if len(r) > 2 else "-"
            print(f"  {label:22s}: {dt:6.2f} s  "
                  f"({big.numel()/dt/1e6:6.1f} M weights/s)   {path}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:22s}: FAILED {type(e).__name__}: {str(e)[:90]}")
    print()
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
