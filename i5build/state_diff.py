"""state_diff.py -- do Embedding4bit and quantize_4bit agree on the quant state?

probe_bisect.py found the two paths produce byte-identical packed weights but
different absmax, and that the Embedding4bit table is *closer* to bf16 (6.0% error)
than the plain quantize_4bit table (9.2%). NF4 at blocksize 64 costs about 9%, so
the 6% table is the suspect one: a path that is more accurate than the format
allows is usually a path that is not quantizing what it claims to.

This prints the two states field by field and, decisively, dequantizes the packed
bytes with each state and measures both against the original. Whichever table the
original quantizer's own state reproduces is the correct interpretation of those
packed bytes; if Embedding4bit's state reproduces something closer to the original
than quantization can explain, its state does not belong to those bytes.

Usage:
    python state_diff.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402

VOCAB = 4096
HIDDEN = 256
BLOCKSIZE = 64
QUANT_TYPE = "nf4"


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / max(1e-12, a.norm())).item()


def describe(qs, label):
    print(f"{label}:")
    print(f"  quant_type={qs.quant_type} blocksize={qs.blocksize} dtype={qs.dtype}")
    print(f"  shape={tuple(qs.shape) if qs.shape is not None else None} "
          f"nested={qs.nested}")
    print(f"  absmax  shape={tuple(qs.absmax.shape)} dtype={qs.absmax.dtype} "
          f"numel={qs.absmax.numel()} "
          f"min={qs.absmax.min().item():.6g} max={qs.absmax.max().item():.6g}")
    print(f"  code    shape={tuple(qs.code.shape)} "
          f"first3={qs.code[:3].tolist()}")
    if qs.state2 is not None:
        print(f"  state2.absmax shape={tuple(qs.state2.absmax.shape)} "
              f"blocksize={qs.state2.blocksize} "
              f"min={qs.state2.absmax.min().item():.6g} "
              f"max={qs.state2.absmax.max().item():.6g}")
        print(f"  offset={qs.offset.item() if qs.offset is not None else None}")


def main():
    torch.manual_seed(0)
    ref = torch.nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)
    w = ref.weight.data

    # path 1: a plain functional call
    packed, qs_plain = bnb.functional.quantize_4bit(
        w.to(torch.float32), quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)

    # path 2: Embedding4bit's own .to("cpu")
    emb = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
    emb.weight = bnb.nn.Params4bit(w.clone(), requires_grad=False,
                                   quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
    emb = emb.to("cpu")
    qs_emb = emb.weight.quant_state
    packed_emb = emb.weight.data

    print(f"packed identical        : {torch.equal(packed, packed_emb)}")
    print(f"absmax identical        : {torch.equal(qs_plain.absmax, qs_emb.absmax)}")
    print(f"code identical          : {torch.equal(qs_plain.code, qs_emb.code)}")
    print(f"nested flags            : plain={qs_plain.nested} emb={qs_emb.nested}")
    print()

    # Are the absmax tensors the same length? A different length means one of them
    # is the nested (compressed) form and the other is fully dequantized, and
    # comparing them directly would be meaningless.
    print("absmax lengths:")
    print(f"  plain {qs_plain.absmax.numel()}   emb {qs_emb.absmax.numel()}")
    nblocks = (VOCAB * HIDDEN + BLOCKSIZE - 1) // BLOCKSIZE
    print(f"  blocks for the full table: {nblocks}")
    print()

    describe(qs_plain, "plain quantize_4bit state")
    print()
    describe(qs_emb, "Embedding4bit state")
    print()

    # Decisive test: dequantize the SAME packed bytes with each state.
    print("dequantizing the same packed bytes with each state:")
    t_plain = bnb.functional.dequantize_4bit(packed, qs_plain)
    try:
        t_emb = bnb.functional.dequantize_4bit(packed, qs_emb)
    except Exception as e:  # noqa: BLE001
        print(f"  with emb state: FAILED {type(e).__name__}: {str(e)[:150]}")
        t_emb = None
    print(f"  with plain state: {tuple(t_plain.shape)} rel_err vs bf16 = "
          f"{rel(t_plain, w):.6f}")
    if t_emb is not None:
        print(f"  with emb state  : {tuple(t_emb.shape)} rel_err vs bf16 = "
              f"{rel(t_emb, w):.6f}")
        print(f"  the two tables identical: {torch.equal(t_plain, t_emb)}")
    print()

    # What does the forward pass actually return, and which state explains it?
    idx = torch.tensor([1, 17, 4095, 1234], dtype=torch.long)
    with torch.no_grad():
        fwd = emb(idx).float()
    print("Embedding4bit forward vs candidates:")
    for label, table in (("bf16 original", w),
                         ("plain-state dequant", t_plain),
                         ("emb-state dequant", t_emb)):
        if table is None:
            continue
        rows = table[idx].float() if table.dim() == 2 else None
        if rows is None:
            continue
        print(f"  vs {label:22s}: rel_err = {rel(fwd, rows):.6f}")
    print()
    print(f"forward dtype: {emb(idx).dtype}, table dtype: {t_plain.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
