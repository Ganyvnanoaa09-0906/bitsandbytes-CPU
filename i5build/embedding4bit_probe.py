"""embedding4bit_probe.py -- does Embedding4bit work on CPU, and can it reload?

The token embedding is the single largest tensor in the checkpoint: 151936 x 4096
bf16 = 1.16 GB. Quantizing it is worth 0.83 GB, so it is not something to skip --
but Embedding4bit has its own forward and its own quantization entry points, and
nothing so far has exercised them on the CPU backend. Linear4bit working says
nothing about Embedding4bit.

Two things have to hold:
  1. Embedding4bit.forward returns the right rows on CPU
  2. a packed weight plus QuantState.as_dict(packed=True) reloads into
     Embedding4bit via Params4bit(bnb_quantized=True), the same way the loader will
     do it -- including the lookup, which must use the dequantized table

Usage:
    python embedding4bit_probe.py
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


def main():
    torch.manual_seed(0)
    ref = torch.nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)
    idx = torch.tensor([0, 1, 17, 4095, 1234], dtype=torch.long)
    with torch.no_grad():
        expect = ref(idx)

    print(f"reference: nn.Embedding({VOCAB}, {HIDDEN}) bf16, "
          f"{ref.weight.numel()*2/1e6:.2f} MB")
    print(f"           lookup {idx.tolist()} -> {tuple(expect.shape)}")
    print()

    # 1) Embedding4bit built the normal way
    print("1) Embedding4bit constructed and quantized via .to('cpu')")
    try:
        emb = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
        emb.weight = bnb.nn.Params4bit(ref.weight.data.clone(), requires_grad=False,
                                       quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
        emb = emb.to("cpu")
        with torch.no_grad():
            got = emb(idx)
        qs = emb.weight.quant_state
        err = ((got.float() - expect.float()).norm()
               / expect.float().norm()).item()
        print(f"   ok: {tuple(got.shape)} {got.dtype}, rel_err={err:.5f}")
        print(f"   quant_state: {qs.quant_type} blocksize={qs.blocksize} "
              f"shape={tuple(qs.shape)}")
        print(f"   packed {emb.weight.numel()} B vs bf16 {ref.weight.numel()*2} B")
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED {type(e).__name__}: {str(e)[:200]}")
        return 1
    print()

    # 2) serialize and reload the way the loader will
    print("2) packed weight + as_dict -> Params4bit(bnb_quantized=True) -> lookup")
    try:
        wq, qstate = bnb.functional.quantize_4bit(
            ref.weight.data.to(torch.float32), quant_type=QUANT_TYPE,
            blocksize=BLOCKSIZE)
        saved = {"weight": wq}
        for k, v in qstate.as_dict(packed=True).items():
            saved[f"weight.{k}"] = v
        # Round-trip through a real file, since the loader reads from safetensors.
        import tempfile

        import safetensors.torch as st
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "emb.safetensors")
            st.save_file(saved, p)
            loaded = st.load_file(p)

        up = bnb.nn.Embedding4bit(VOCAB, HIDDEN, quant_type=QUANT_TYPE)
        param = bnb.nn.Params4bit(loaded["weight"], requires_grad=False,
                                  quant_type=QUANT_TYPE, blocksize=BLOCKSIZE,
                                  bnb_quantized=True,
                                  quant_storage=loaded["weight"].dtype)
        param.quant_state = bnb.functional.QuantState.from_dict(
            {k: v for k, v in loaded.items() if k.startswith("weight.")},
            device=torch.device("cpu"))
        param.module = up
        up.weight = param
        up.quant_state = param.quant_state
        setattr(up, "quant_state", param.quant_state)
        with torch.no_grad():
            got2 = up(idx)
        err2 = ((got2.float() - got.float()).norm() / got.float().norm()).item()
        print(f"   ok: {tuple(got2.shape)}, rel_err vs the direct Embedding4bit "
              f"= {err2:.6f}")
        print(f"   (0.000000 means the reload reproduced the same table exactly)")
        if err2 > 1e-4:
            print(f"   FAILED: reloaded table differs by {err2:.6f}")
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED {type(e).__name__}: {str(e)[:300]}")
        return 1

    print()
    print("RESULT: PASS -- Embedding4bit runs and reloads on CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
