"""probe_bisect.py -- find why two probes disagreed by 3.6%.

embedding4bit_probe.py claimed a reloaded Embedding4bit table differs from a
directly quantized one by 3.6%; embedding_reload_diff.py then showed the packed
bytes, the absmax, the code, the serialized quant_state and the dequantized table
are all bit-identical, and that the forward pass reproduces the table exactly.

Both cannot be true, so the earlier probe measured something other than what it
claimed. Candidates, in the order they are tested here:

  A. index 0 is a padding row that is all zeros, so including it in a batch
     relative-error metric inflates the error without any weight being wrong
  B. Embedding4bit's forward is called with the wrong quantization state and only
     the zero row hides it
  C. the two tables really are different and the table comparison was comparing a
     table against itself

Each is settled by a measurement rather than by argument.

Usage:
    python probe_bisect.py
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


def main():
    torch.manual_seed(0)
    ref = torch.nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)

    # A: is index 0 a zero row?
    row0 = ref.weight.data[0]
    print("A) is row 0 special?")
    print(f"   row 0 all zero          : {bool((row0 == 0).all())}")
    print(f"   row 0 norm              : {row0.float().norm().item():.6e}")
    nz = int((ref.weight.data.abs().sum(dim=1) > 0).sum())
    print(f"   non-zero rows           : {nz} of {VOCAB}")
    print(f"   embedding padding_idx   : {ref.padding_idx}")
    print()

    # The exact comparison the first probe made, with and without index 0.
    emb = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
    emb.weight = bnb.nn.Params4bit(ref.weight.data.clone(), requires_grad=False,
                                   quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
    emb = emb.to("cpu")
    qs_direct = emb.weight.quant_state

    packed, qs = bnb.functional.quantize_4bit(ref.weight.data.to(torch.float32),
                                              quant_type=QUANT_TYPE,
                                              blocksize=BLOCKSIZE)

    print("B) is the direct Embedding4bit path quantizing the same table?")
    print(f"   packed identical to a plain quantize_4bit call : "
          f"{torch.equal(emb.weight.data, packed)}")
    print(f"   absmax identical                               : "
          f"{torch.equal(qs_direct.absmax, qs.absmax)}")
    print()

    idx_all = torch.tensor([0, 1, 17, 4095, 1234], dtype=torch.long)
    idx_nz = torch.tensor([1, 17, 4095, 1234], dtype=torch.long)
    with torch.no_grad():
        e_all = rel(emb(idx_all), ref(idx_all))
        e_nz = rel(emb(idx_nz), ref(idx_nz))
    print("C) the metric the first probe used, with and without index 0")
    print(f"   rel_err over {idx_all.tolist()}  = {e_all:.6f}   <- what was reported")
    print(f"   rel_err over {idx_nz.tolist()}   = {e_nz:.6f}   <- same table, no zero row")
    if e_nz < 1e-9:
        print("   -> the table is reproduced EXACTLY; the whole 9.9% was the zero row")
    print()

    # And the reload comparison, done properly this time.
    saved = {"weight": packed}
    for k, v in qs.as_dict(packed=True).items():
        saved[f"weight.{k}"] = v
    import tempfile

    import safetensors.torch as st
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "e.safetensors")
        st.save_file(saved, p)
        loaded = st.load_file(p)
    qs2 = bnb.functional.QuantState.from_dict(
        {k: v for k, v in loaded.items() if k.startswith("weight.")},
        device=torch.device("cpu"))
    up = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
    param = bnb.nn.Params4bit(loaded["weight"], requires_grad=False,
                              quant_type=QUANT_TYPE, blocksize=BLOCKSIZE,
                              bnb_quantized=True,
                              quant_storage=loaded["weight"].dtype)
    param.quant_state = qs2
    param.module = up
    up.weight = param
    up.quant_state = qs2

    with torch.no_grad():
        got = up(idx_nz)
        dir_ = emb(idx_nz)
        e_reload = rel(got, dir_)
        e_vs_ref = rel(got, ref(idx_nz))
    print("D) reloaded table vs directly quantized table, no zero row involved")
    print(f"   rel_err(reloaded, direct) = {e_reload:.9f}")
    print(f"   rel_err(reloaded, bf16)   = {e_vs_ref:.6f}")
    print()

    # The honest number: what does 4-bit cost on a real embedding table?
    print("E) the number that actually matters, excluding padding rows")
    mask = ref.weight.data.abs().sum(dim=1) > 0
    with torch.no_grad():
        table_direct = bnb.functional.dequantize_4bit(emb.weight.data, qs_direct)
        table_reload = bnb.functional.dequantize_4bit(loaded["weight"], qs2)
    real_q = rel(table_direct[mask], ref.weight.data[mask])
    real_r = rel(table_reload[mask], ref.weight.data[mask])
    print(f"   quantized vs bf16, non-padding rows only : {real_q:.6f}")
    print(f"   reloaded  vs bf16, non-padding rows only : {real_r:.6f}")
    print(f"   direct and reloaded tables identical     : "
          f"{torch.equal(table_direct, table_reload)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
