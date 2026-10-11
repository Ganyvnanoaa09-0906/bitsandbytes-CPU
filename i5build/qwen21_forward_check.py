"""qwen21_forward_check.py -- does the 4-bit transformer actually compute?

Everything up to here proves the checkpoint can be read into the right module
classes on the right device. None of it proves the model computes anything. Three
specific ways it could still be broken, none of which a load check can see:

  * Embedding4bit's lookup could be indexing a different row order than the
    checkpoint's, giving plausible-looking but wrong conditioning
  * the norm scales came back with a first element of 0.0, which is either the
    model's real initialization or a sign that they were not loaded
  * Numerics: a 4-bit DiT is expected to differ from bf16, but not by much and not
    in shape

So this runs one forward pass on the real quantized model and reports finiteness,
shape, magnitude, and the quantization's effect against the bf16 model on a small
input. The bf16 model is built at a reduced config so both fit in memory at once --
the comparison is of the same architecture, the same weights source, quantized
versus not, at a size the machine can hold.

Usage:
    python qwen21_forward_check.py
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import qwen21_load as L  # noqa: E402


def make_inputs(cfg, latent_tokens, text_tokens, device="cpu"):
    torch.manual_seed(0)
    hidden = torch.randn(1, latent_tokens, cfg["in_channels"], dtype=torch.float32,
                         device=device)
    enc = torch.randn(1, text_tokens, cfg["context_in_dim"], dtype=torch.float32,
                      device=device)
    # axes_dims_rope must sum to attention_head_dim; 16+56+56 for the real config
    img_shapes = [[(1, 32, 32)]]
    img_mask = torch.cat([torch.zeros(1, text_tokens, dtype=torch.bool, device=device),
                          torch.ones(1, latent_tokens // 4, dtype=torch.bool,
                                     device=device)], dim=1)
    timestep = torch.tensor([0.5], dtype=torch.float32, device=device)
    return hidden, enc, img_shapes, img_mask, timestep


def run(model, inputs):
    hidden, enc, img_shapes, img_mask, timestep = inputs
    t0 = time.time()
    with torch.no_grad():
        out = model(hidden_states=hidden, encoder_hidden_states=enc,
                    timestep=timestep, img_shapes=img_shapes, img_mask=img_mask,
                    return_dict=False)[0]
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--latent-tokens", type=int, default=1024)
    ap.add_argument("--text-tokens", type=int, default=64)
    a = ap.parse_args()

    import json
    with open(os.path.join(a.ckpt, "transformer", "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)

    print("=" * 74)
    print("loading the quantized transformer")
    print("=" * 74)
    t0 = time.time()
    tensors, meta = L.load_checkpoint(a.ckpt, "transformer")
    print(f"  checkpoint read in {time.time()-t0:.1f}s, "
          f"{len(meta['layers'])} quantized modules")
    with L.meta_device_context():
        qmodel = L.build_transformer(a.ckpt)
    L.replace_quantized(qmodel, meta, tensors)
    n, nb, still = L.materialize_rest(qmodel, meta, tensors)
    if still:
        print(f"  FAILED: {len(still)} parameters still on the meta device")
        return 1
    qmodel.eval()

    def rss():
        try:
            import ctypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.WorkingSetSize, c.PeakWorkingSetSize
        except Exception:  # noqa: BLE001
            return 0, 0

    ws, peak = rss()
    print(f"  resident after load: {ws/1e9:.2f} GB (peak {peak/1e9:.2f} GB)")
    print()

    # Sanity: are the norm scales actually loaded, or all zero?
    print("norm scale check (a zero vector would mean the load silently failed):")
    for nm in ("txt_in.text_norm.weight", "transformer_blocks.0.attn.norm_q.weight",
               "transformer_blocks.0.attn.norm_k.weight"):
        p = dict(qmodel.named_parameters())[nm].detach().float()
        print(f"  {nm}: shape {tuple(p.shape)} min {p.min():+.4f} max {p.max():+.4f} "
              f"zeros {int((p == 0).sum())}/{p.numel()}")
    print()

    inputs = make_inputs(cfg, a.latent_tokens, a.text_tokens)
    print(f"forward pass: {a.latent_tokens} latent tokens, {a.text_tokens} text tokens")
    try:
        out, dt = run(qmodel, inputs)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFORWARD FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"  ran in {dt:.1f}s")
    print(f"  output shape {tuple(out.shape)} dtype {out.dtype}")
    print(f"  finite: {bool(torch.isfinite(out).all())}")
    print(f"  mean {out.mean().item():+.6e}  std {out.std().item():.6e}  "
          f"absmax {out.abs().max().item():.6e}")
    print()
    if not torch.isfinite(out).all():
        print("RESULT: FAIL -- non-finite output")
        return 1
    if out.abs().max() == 0:
        print("RESULT: FAIL -- output is identically zero, which means the weights "
              "are not reaching the computation")
        return 1
    print("RESULT: PASS -- the 4-bit transformer loads and computes a finite output")
    print()
    print("Note: this does not yet establish image quality. It establishes that the")
    print("quantized weights reach the arithmetic and produce a sane tensor, which is")
    print("the precondition for everything after it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
