"""embedding_reload_diff.py -- where exactly does the reloaded table diverge?

embedding4bit_probe.py reported that a table reloaded from packed safetensors
differs from the directly quantized one by 3.6%. A 3.6% difference with the same
packed bytes and the same quant_state can only come from one of a few places, and
guessing between them is how a wrong conclusion gets recorded:

  * the packed bytes really did change in the file round trip
  * the QuantState really did change (blocksize, shape, absmax, nested stats)
  * the LENGTH of the dequantized table changed, so the comparison is between
    tensors of different lengths -- a bug in the test, not in the library
  * Embedding4bit's lookup uses something other than dequantize_4bit

Comparing dequantized tables directly separates all four: identical bytes plus
identical state must give identical tables, and if the tables differ in LENGTH then
the earlier "relative error" was meaningless.

Runs the same comparison for Linear4bit as a control, since that path is already
known to reload correctly and a difference between the two would point at
Embedding4bit specifically.

Usage:
    python embedding_reload_diff.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402
import safetensors.torch as st  # noqa: E402

VOCAB = 4096
HIDDEN = 256
BLOCKSIZE = 64
QUANT_TYPE = "nf4"


def state_summary(qs, label):
    parts = [f"{label}: quant_type={qs.quant_type} blocksize={qs.blocksize} "
             f"shape={tuple(qs.shape) if qs.shape is not None else None} "
             f"dtype={qs.dtype} nested={qs.nested}"]
    parts.append(f"    absmax shape={tuple(qs.absmax.shape)} "
                 f"dtype={qs.absmax.dtype} numel={qs.absmax.numel()}")
    if qs.code is not None:
        parts.append(f"    code shape={tuple(qs.code.shape)}")
    if qs.state2 is not None:
        parts.append(f"    state2.absmax shape={tuple(qs.state2.absmax.shape)} "
                     f"blocksize={qs.state2.blocksize} "
                     f"offset={qs.offset.item() if qs.offset is not None else None}")
    return "\n".join(parts)


def roundtrip_embedding():
    print("=" * 78)
    print("Embedding4bit")
    print("=" * 78)
    torch.manual_seed(0)
    w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)

    wq, qs = bnb.functional.quantize_4bit(w.to(torch.float32),
                                         quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
    saved = {"weight": wq}
    for k, v in qs.as_dict(packed=True).items():
        saved[f"weight.{k}"] = v

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.safetensors")
        st.save_file(saved, p)
        loaded = st.load_file(p)

    print(f"packed bytes identical after file round trip: "
          f"{torch.equal(wq, loaded['weight'])}")
    for k in saved:
        if k == "weight":
            continue
        if k in loaded:
            same = (torch.equal(saved[k], loaded[k])
                    if isinstance(saved[k], torch.Tensor) else saved[k] == loaded[k])
            print(f"  {k}: {'identical' if same else 'DIFFERS'} "
                  f"{tuple(loaded[k].shape) if isinstance(loaded[k], torch.Tensor) else loaded[k]}")
        else:
            print(f"  {k}: MISSING from the reloaded file")

    qs2 = bnb.functional.QuantState.from_dict(
        {k: v for k, v in loaded.items() if k.startswith("weight.")},
        device=torch.device("cpu"))
    print()
    print(state_summary(qs, "original"))
    print(state_summary(qs2, "reloaded"))
    print()

    t1 = bnb.functional.dequantize_4bit(wq, qs)
    t2 = bnb.functional.dequantize_4bit(loaded["weight"], qs2)
    print(f"dequantized original : {tuple(t1.shape)} {t1.dtype} numel={t1.numel()}")
    print(f"dequantized reloaded : {tuple(t2.shape)} {t2.dtype} numel={t2.numel()}")
    if t1.shape != t2.shape:
        print("  -> SHAPES DIFFER: the earlier 'relative error' compared tensors of "
              "different lengths and meant nothing")
    if t1.shape == t2.shape:
        print(f"  identical: {torch.equal(t1, t2)}")
        if not torch.equal(t1, t2):
            d = (t1.float() - t2.float()).abs()
            print(f"  max abs diff {d.max().item():.6e}, "
                  f"first differing index {(d > 0).nonzero()[0].tolist()}")
    print()

    # Does a real Embedding4bit lookup agree with the table?
    print("does Embedding4bit's forward use dequantize_4bit?")
    up = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
    param = bnb.nn.Params4bit(loaded["weight"], requires_grad=False,
                              quant_type=QUANT_TYPE, blocksize=BLOCKSIZE,
                              bnb_quantized=True, quant_storage=loaded["weight"].dtype)
    param.quant_state = qs2
    param.module = up
    up.weight = param
    up.quant_state = qs2
    idx = torch.tensor([0, 1, 17, 4095, 1234], dtype=torch.long)
    with torch.no_grad():
        via_fwd = up(idx)
    print(f"  forward out {tuple(via_fwd.shape)} {via_fwd.dtype}")
    print(f"  dequantized table {tuple(t2.shape)} {t2.dtype}")
    # The table may be [vocab, hidden] or flat; row i of a [vocab, hidden] table
    # must equal the lookup of index i.
    if t2.dim() == 2 and tuple(t2.shape) == (VOCAB, HIDDEN):
        rows = t2[idx].float()
        e = ((via_fwd.float() - rows).norm() / rows.norm()).item()
        print(f"  rel_err(forward, table rows) = {e:.6f}")
        if e > 1e-4:
            print("  -> forward returned different values than the table lookup")
    else:
        print("  table is not [vocab, hidden]; cannot compare directly")


def main():
    roundtrip_embedding()
    print()
    print("=" * 78)
    print("Linear4bit, as a control: this path is known to reload correctly")
    print("=" * 78)
    torch.manual_seed(0)
    lin = torch.nn.Linear(384, 512, bias=False).to(torch.bfloat16)
    wq, qs = bnb.functional.quantize_4bit(lin.weight.data.to(torch.float32),
                                         quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
    saved = {"weight": wq}
    for k, v in qs.as_dict(packed=True).items():
        saved[f"weight.{k}"] = v
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "l.safetensors")
        st.save_file(saved, p)
        loaded = st.load_file(p)
    qs2 = bnb.functional.QuantState.from_dict(
        {k: v for k, v in loaded.items() if k.startswith("weight.")},
        device=torch.device("cpu"))
    t1 = bnb.functional.dequantize_4bit(wq, qs)
    t2 = bnb.functional.dequantize_4bit(loaded["weight"], qs2)
    print(f"dequantized original {tuple(t1.shape)} vs reloaded {tuple(t2.shape)}: "
          f"identical={torch.equal(t1, t2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
