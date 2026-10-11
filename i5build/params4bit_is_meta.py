"""params4bit_is_meta.py -- does Params4bit report is_meta correctly?

The loader replaces 232 Linear modules with Linear4bit and reports, separately, that
65 parameters are still on the meta device. The restore loop copies 65 unquantized
parameters and skips none, so the two counts cannot both describe the same thing:
65 + 65 is not 232 + 65.

The likely explanation is that `is_meta` lies for Params4bit. It is a torch.Tensor
subclass holding a real uint8 tensor, and `torch.Tensor.is_meta` is documented for
regular tensors -- a subclass with custom attributes may not answer it the way the
loader assumes. If so, the loader's check is wrong and the model may be perfectly
fine, which is worth knowing before rewriting anything.

This settles it on a small quantized layer rather than by reasoning.

Usage:
    python params4bit_is_meta.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402


def describe(label, t):
    parts = [f"{label}: type={type(t).__name__}"]
    for attr in ("is_meta", "is_cuda", "device", "dtype"):
        try:
            parts.append(f"{attr}={getattr(t, attr)}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"{attr}=<{type(e).__name__}>")
    try:
        parts.append(f"data.device={t.data.device}")
        parts.append(f"data.is_meta={t.data.is_meta}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"data=<{type(e).__name__}>")
    print("  " + "  ".join(parts))


def main():
    print(f"torch {torch.__version__}")
    print()

    print("1) a plain tensor on cpu:")
    describe("cpu tensor", torch.zeros(4, 4))
    print("2) a plain tensor on the meta device:")
    describe("meta tensor", torch.zeros(4, 4, device="meta"))
    print()

    print("3) Params4bit holding a real CPU tensor:")
    packed = torch.zeros(2048, 1, dtype=torch.uint8)
    p = bnb.nn.Params4bit(packed, requires_grad=False, quant_type="nf4",
                          blocksize=64, bnb_quantized=True,
                          quant_storage=torch.uint8)
    describe("Params4bit", p)
    print()

    print("4) Params4bit as a module's weight, which is what the loader walks:")
    lin = bnb.nn.Linear4bit(128, 256, bias=False, compute_dtype=torch.float32,
                            quant_type="nf4")
    w = torch.randn(256, 128, dtype=torch.bfloat16)
    lin.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4",
                                   blocksize=64)
    lin = lin.to("cpu")
    describe("Linear4bit.weight", lin.weight)
    print(f"     quant_state is None? {lin.weight.quant_state is None}")
    print(f"     packed numel {lin.weight.numel()}  dtype {lin.weight.dtype}")
    print()

    print("5) how many parameters of a real quantized model report is_meta?")
    try:
        sys.path.insert(0, HERE)
        import qwen21_load as L

        tensors, meta = L.load_checkpoint(r"D:\work\qwen_env\Qwen-Image-2.1-4bit",
                                          "transformer")
        with L.meta_device_context():
            m = L.build_transformer(r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
        L.replace_quantized(m, meta, tensors, verbose=False)
        L.materialize_rest(m, meta, tensors)
        quantized = set(meta["layers"])
        real_meta = [(n, p) for n, p in m.named_parameters()
                     if p.is_meta and n not in quantized]
        reported = [(n, p) for n, p in m.named_parameters() if p.is_meta]
        print(f"     parameters total              : {len(list(m.named_parameters()))}")
        print(f"     report is_meta                : {len(reported)}")
        print(f"     of those, quantized (replaced): "
              f"{len([1 for n, _ in reported if n in quantized])}")
        print(f"     of those, NOT quantized       : {len(real_meta)}")
        for n, p in real_meta[:5]:
            print(f"       genuinely meta: {n} {tuple(p.shape)}")
        # And the decisive practical question: can it run?
        print()
        print("     can the weight be read back?")
        for n, p in reported[:3]:
            try:
                if hasattr(p, "quant_state") and p.quant_state is not None:
                    dq = bnb.functional.dequantize_4bit(p.data, p.quant_state)
                    print(f"       {n}: dequantized to {tuple(dq.shape)} "
                          f"{dq.dtype}, finite={bool(torch.isfinite(dq).all())}")
                else:
                    print(f"       {n}: no quant_state (unquantized parameter)")
            except Exception as e:  # noqa: BLE001
                print(f"       {n}: FAILED {type(e).__name__}: {str(e)[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"     FAILED {type(e).__name__}: {str(e)[:250]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
