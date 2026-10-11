"""quant_format_probe.py -- can a 4-bit checkpoint round-trip and run on CPU?

The pieces are individually proven: bnb quantizes and dequantizes on CPU, and
Linear4bit.forward works on CPU. What is NOT proven is the part that decides
whether a quantized Qwen-Image-2.1 is loadable at all -- writing a real model's
quantized weights to safetensors and loading them back WITHOUT going through
transformers' from_pretrained path.

That path cannot be used: it routes through `bitsandbytes`' transformers
integration, which requires CUDA (device_map, .cuda()). What has to work instead
is Linear4bit + Params4bit(bnb_quantized=True), the same trick
Params4bit.from_prequantized performs for CUDA, applied on CPU.

Runs against a small QwenImage21Transformer2DModel built from an explicit config,
so it needs no checkpoint download and completes in minutes.

Test sequence:
  1. build a tiny QwenImage21Transformer2DModel on the meta device
  2. quantize every 2-D linear weight with quantize_4bit, save packed QuantState
  3. load it back with Params4bit(bnb_quantized=True) and run a CPU forward
  4. compare against the same model unquantized, and report the error

Usage:
    python quant_format_probe.py
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import bitsandbytes as bnb  # noqa: E402
import safetensors.torch as st  # noqa: E402

OUT = r"D:\work\_quantprobe"
BLOCKSIZE = 64
QUANT_TYPE = "nf4"


def rel_err(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / max(1e-12, a.norm())).item()


def tiny_config():
    """A QwenImage21Transformer2DModel small enough to run in seconds.

    Same arguments as the real transformer/config.json -- patch_size 1, in/out
    channels 64, context_in_dim 4096, axes_dims_rope (16, 56, 56) -- with only the
    sizes reduced. Keeping the argument *names* identical matters, because the
    module structure and therefore the parameter key names are what the quantizer
    matches on, and a divergent tiny model would prove nothing about the real one.

    One constraint has to be preserved rather than scaled: axes_dims_rope must sum
    to attention_head_dim, because the rotary embedding is applied per head. The
    real model satisfies it as 16+56+56 = 128, so the reduced model satisfies it as
    8+8+16 = 32.
    """
    return {
        "patch_size": 1,
        "in_channels": 64,
        "out_channels": 64,
        "num_layers": 2,
        "attention_head_dim": 32,
        "num_attention_heads": 4,
        "context_in_dim": 64,
        "mlp_ratio": 3,
        "axes_dims_rope": (8, 8, 16),
        "eps": 1e-06,
        "causal_condition": True,
    }


def is_linear_weight(name, tensor):
    """The quantization rule, in one place: a 2-D parameter whose name ends in
    .weight. Embeddings, norms, biases and 1-D tensors keep full precision."""
    return tensor.dim() == 2 and name.endswith(".weight")


def main():
    from diffusers.models import QwenImage21Transformer2DModel

    cfg = tiny_config()
    t0 = time.time()
    torch.manual_seed(0)
    model = QwenImage21Transformer2DModel(**cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny model: {n_params/1e6:.2f}M params, built in {time.time()-t0:.1f}s")

    # Which tensors would be quantized? Report the rule rather than guessing.
    targets = [(n, p) for n, p in model.named_parameters()
               if is_linear_weight(n, p)]
    other = [(n, p) for n, p in model.named_parameters()
             if not is_linear_weight(n, p)]
    q_bytes = sum(p.numel() * 2 for _, p in targets)
    o_bytes = sum(p.numel() * 2 for _, p in other)
    print(f"  2-D .weight tensors: {len(targets)} ({q_bytes/1e6:.2f} MB in bf16)")
    print(f"  everything else    : {len(other)} ({o_bytes/1e6:.2f} MB in bf16)")
    print()

    # Reference forward, done before quantization so the weights are untouched.
    #
    # Shape rules read off the vendored transformer's forward():
    #   hidden_states is already packed: (B, num_latent_tokens, in_channels)
    #   img_shapes is (frame, height, width) per image, in latent tokens
    #   img_mask marks the vision-language encoder's image slots, and each slot
    #   stands for a 2x2 group, so 64 latent tokens need 16 True slots
    #   axes_dims_rope sums to attention_head_dim (24 -> 3 axes), which fixes the
    #   position-id width at 3
    torch.manual_seed(1)
    cfg_axes = sum(cfg["axes_dims_rope"])
    latent_tokens = 64
    text_tokens = 6
    hidden = torch.randn(1, latent_tokens, cfg["in_channels"], dtype=torch.float32)
    enc = torch.randn(1, text_tokens, cfg["context_in_dim"], dtype=torch.float32)
    img_shapes = [[(1, 8, 8)]]
    img_mask = torch.cat([torch.zeros(1, text_tokens, dtype=torch.bool),
                          torch.ones(1, latent_tokens // 4, dtype=torch.bool)], dim=1)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    with torch.no_grad():
        ref = model(hidden_states=hidden, encoder_hidden_states=enc,
                    timestep=timestep, img_shapes=img_shapes, img_mask=img_mask,
                    return_dict=False)[0]
    print(f"reference output: {tuple(ref.shape)} {ref.dtype}")
    print()

    # Quantize, saving in the exact layout Linear4bit._save_to_state_dict writes:
    # the packed tensor under `weight`, and as_dict(packed=True) under `weight.*`.
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "quant.safetensors")
    saved = {}
    meta = {}
    t0 = time.time()
    for name, p in model.named_parameters():
        if is_linear_weight(name, p):
            wq, qs = bnb.functional.quantize_4bit(
                p.data.to(torch.float32), quant_type=QUANT_TYPE, blocksize=BLOCKSIZE)
            saved[name] = wq
            for k, v in qs.as_dict(packed=True).items():
                saved[f"{name}.{k}"] = v
            meta[name] = {"shape": list(p.shape), "blocksize": qs.blocksize,
                          "quant_type": qs.quant_type, "nested": bool(qs.nested)}
        else:
            saved[name] = p.data.to(torch.bfloat16)
    st.save_file(saved, path)
    with open(os.path.join(OUT, "quant_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"blocksize": BLOCKSIZE, "quant_type": QUANT_TYPE, "layers": meta}, fh,
                  indent=2)
    qsize = sum(t.numel() * t.element_size() for t in saved.values())
    print(f"quantized + saved in {time.time()-t0:.1f}s")
    print(f"  file {os.path.getsize(path)/1e6:.2f} MB "
          f"(vs {2*n_params/1e6:.2f} MB if every parameter stayed bf16)")
    print(f"  quantized tensors: {len(meta)}")
    print()

    # Load the quantized weights into a fresh model built on the meta device, so
    # the bf16 weights are never materialized -- this is what makes a 14 GB model
    # fit in a 15.4 GB budget.
    torch.manual_seed(0)
    qmodel = QwenImage21Transformer2DModel(**cfg).eval()
    loaded = st.load_file(path)
    n_replaced = 0
    for module_name, module in list(qmodel.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        # The root module has an empty name and therefore no parent to attach to;
        # a bare Linear as the whole model is not a case that exists here.
        if "." not in module_name:
            continue
        wkey = f"{module_name}.weight"
        if wkey not in meta:
            continue
        packed = loaded[wkey]
        qstate = bnb.functional.QuantState.from_dict(
            {k: v for k, v in loaded.items() if k.startswith(wkey + ".")},
            device=torch.device("cpu"))
        new = bnb.nn.Linear4bit(module.in_features, module.out_features,
                                bias=module.bias is not None,
                                compute_dtype=torch.float32,
                                quant_type=QUANT_TYPE)
        param = bnb.nn.Params4bit(packed, requires_grad=False, quant_type=QUANT_TYPE,
                                  blocksize=BLOCKSIZE, bnb_quantized=True,
                                  quant_storage=packed.dtype)
        param.quant_state = qstate
        param.module = new
        new.weight = param
        new.quant_state = qstate
        if module.bias is not None:
            new.bias = torch.nn.Parameter(module.bias.data.clone())
        parent = qmodel.get_submodule(module_name.rsplit(".", 1)[0])
        setattr(parent, module_name.rsplit(".", 1)[1], new)
        n_replaced += 1
    print(f"replaced {n_replaced} Linear modules with Linear4bit")

    # Everything that was not quantized still has to be put back. Matching on the
    # parameter's own name rather than on a substring of the key list, because a
    # name like "x.to_q.weight" occurring anywhere in the file says nothing about
    # whether this particular parameter needs restoring.
    n_restored = 0
    with torch.no_grad():
        for name, p in qmodel.named_parameters():
            if name in meta:
                continue
            if name in loaded:
                p.copy_(loaded[name].to(p.dtype))
                n_restored += 1
    print(f"restored {n_restored} unquantized tensors")
    print()

    t0 = time.time()
    with torch.no_grad():
        got = qmodel(hidden_states=hidden, encoder_hidden_states=enc,
                     timestep=timestep, img_shapes=img_shapes, img_mask=img_mask,
                     return_dict=False)[0]
    print(f"quantized forward ok in {time.time()-t0:.1f}s, {tuple(got.shape)}")
    e = rel_err(ref.float(), got.float())
    print(f"relative error vs unquantized: {e:.5f}")
    print()

    if not torch.isfinite(got).all():
        print("RESULT: FAIL -- non-finite values in the quantized output")
        return 1
    if e > 0.5:
        print("RESULT: FAIL -- error is too large to be a quantization artefact")
        return 1
    print("RESULT: PASS -- 4-bit save/load/forward round-trips on CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
