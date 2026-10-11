"""norm_zip.py -- are the qk-norm scales zero in the CHECKPOINT, or only after load?

The quantized transformer's forward pass returns finite values, but every
transformer_blocks.*.attn.norm_q.weight and norm_k.weight comes back all zero. Two
explanations with opposite consequences:

  * the checkpoint stores zeros, so the model really does zero its QK-norm scales
    and that is how it was trained. Nothing to fix, but attention degenerates to
    mean pooling and the quantization is not the reason for any softness.
  * the checkpoint stores values and the load loses them, so attention is running
    without its normalization scales and any image produced would be wrong.

The distinction is decided by reading the source safetensors directly and by
reading it back through the same Params4bit path the loader uses, so a failure in
either stage shows up separately. Also checks the bf16 source rather than the
quantized copy, because the quantizer keeps these tensors verbatim.

Usage:
    python norm_zip.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import safetensors.torch as st  # noqa: E402

SRC = r"D:\work\qwen_env\Qwen-Image-2.1"
QUANT = r"D:\work\qwen_env\Qwen-Image-2.1-4bit"

KEYS = [
    "transformer_blocks.0.attn.norm_q.weight",
    "transformer_blocks.0.attn.norm_k.weight",
    "transformer_blocks.1.attn.norm_q.weight",
    "txt_in.text_norm.weight",
    "transformer_blocks.0.attn.to_q.weight",
]


def report(label, getter):
    print(f"{label}:")
    for k in KEYS:
        try:
            t = getter(k)
        except KeyError:
            print(f"  {k}: absent")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  {k}: ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        if t is None:
            print(f"  {k}: absent")
            continue
        f = t.detach().float().reshape(-1)
        print(f"  {k}: shape {tuple(t.shape)} dtype {t.dtype} "
              f"min {f.min():+.6g} max {f.max():+.6g} zeros {int((f == 0).sum())}/{f.numel()}")
    print()


def main():
    # --- 1. straight from the bf16 source shards ---
    with open(os.path.join(SRC, "transformer",
                           "diffusion_pytorch_model.safetensors.index.json"),
              encoding="utf-8") as fh:
        wm = json.load(fh)["weight_map"]

    # safe_open gives lazy per-tensor access. The first version of this function
    # called st.load_file on the whole shard, returned a tensor from the resulting
    # dict, and then deleted the dict -- so the tensor's storage was freed while the
    # reference was still used, and the process died with an access violation
    # (0xC0000005) rather than any Python exception. Streaming one tensor at a time
    # both fixes that and avoids pulling 9.3 GB into RAM to look at a 128-element
    # vector.
    from safetensors import safe_open

    def from_source(k):
        if k not in wm:
            return None
        path = os.path.join(SRC, "transformer", wm[k])
        with safe_open(path, framework="pt", device="cpu") as fh:
            return fh.get_tensor(k).clone()

    report("1) bf16 SOURCE shards (the ground truth)", from_source)

    # --- 2. from the quantized checkpoint, same way the loader reads it ---
    import qwen21_load as L

    tensors, meta = L.load_checkpoint(QUANT, "transformer")

    def from_quant(k):
        return tensors.get(k)

    report("2) quantized checkpoint, raw tensors", from_quant)

    # --- 3. after the loader has built the model ---
    with L.meta_device_context():
        qmodel = L.build_transformer(QUANT)
    L.replace_quantized(qmodel, meta, tensors, verbose=False)
    n, nb, still = L.materialize_rest(qmodel, meta, tensors)
    named = dict(qmodel.named_parameters())

    def from_model(k):
        return named.get(k)

    report("3) in the loaded model", from_model)

    print(f"materialized {n} parameters, {len(still)} still meta")
    print()
    print("reading:")
    print("  if (1) already shows zeros, the model is trained that way and the")
    print("  loader is innocent. If (1) is non-zero and (2) or (3) is zero, the")
    print("  loss happens in the quantizer or the loader respectively, and the")
    print("  weights that matter are the ones in (2), because that is what the")
    print("  loader is handed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
