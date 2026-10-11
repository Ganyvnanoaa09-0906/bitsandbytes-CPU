"""sd15_image_ab.py -- generate images at fp32, 4-bit and 2-bit, and compare them.

This is the measurement every earlier attempt was circling. Weight reconstruction error
(2-bit is 5.7x worse than 4-bit) says nothing on its own; a half-loaded model probing
individual layers produced a degenerate amplification factor; one layer's ratio does not
generalise. What does answer it is a picture.

The UNet here is complete and the rest of the pipeline is on disk, so a real prompt can
be rendered end to end and the three outputs compared. Two comparisons are reported,
because they answer different questions:

  * against fp32 -- how much does each format cost in absolute terms
  * against each other -- whether 2-bit is a reasonable substitute for 4-bit, which is
    the actual decision, since 4-bit is what would ship

The images are written out as well, because a numeric difference and a visible one are
not the same thing and only one of them can be judged from a tensor.

What is quantized, stated up front because it caps every conclusion: bitsandbytes has no
convolution quantization, and 68.5% of this UNet's parameters are Conv2d. Only the 31.4%
in Linear layers is touched. The VAE and text encoder are left in fp32 -- they hold
another 200M parameters of the 1.13e9 total.

Usage:
    python sd15_image_ab.py [--prompt "..."] [--steps 20] [--size 384]
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))

import quant2bit as Q2  # noqa: E402

ROOT = r"D:\work\textmodel"
BASE = os.path.join(ROOT, "sd15_base")
FULL = os.path.join(ROOT, "sd15_full")
COMBINED = os.path.join(ROOT, "_sd15_combined")


MODEL_INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "_diffusers_version": "0.27.0",
    "scheduler": ["diffusers", "DDIMScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "safety_checker": [None, None],
    "feature_extractor": [None, None],
}


def build_combined():
    """Assemble a complete SD1.5 from the two halves, by directory junction.

    sd15_base holds only unet/, sd15_full holds only scheduler, text_encoder and vae.
    Neither alone is a pipeline, and neither carries a model_index.json -- that file is
    the pipeline manifest and without it from_pretrained refuses the directory with
    "Error no file named model_index.json found", which is what the first run of this
    script hit.

    Two more gaps had to be filled, both discovered by running it rather than by reading
    the tree:
      * tokenizer/ is absent. SD1.5 keeps the CLIP tokenizer in its own directory, but
        sd15_full's text_encoder/ already contains vocab.json, merges.txt and
        tokenizer_config.json, so the tokenizer component points there.
      * the manifest itself has to be written. The component list is taken from the
        counterfeit pipeline's own model_index.json, which is the same architecture.

    Junctions rather than copies, so 4 GB is not duplicated, and the directory is
    rebuilt on each run so a stale link cannot silently supply the wrong weights.
    """
    if os.path.isdir(COMBINED):
        shutil.rmtree(COMBINED, ignore_errors=True)
    os.makedirs(COMBINED, exist_ok=True)
    made = []
    for src, name in ((os.path.join(BASE, "unet"), "unet"),
                      (os.path.join(FULL, "scheduler"), "scheduler"),
                      (os.path.join(FULL, "text_encoder"), "text_encoder"),
                      (os.path.join(FULL, "vae"), "vae")):
        if not os.path.isdir(src):
            print(f"  missing component: {src}")
            continue
        dst = os.path.join(COMBINED, name)
        os.system(f'cmd /c mklink /J "{dst}" "{src}" >nul 2>&1')
        if os.path.isdir(dst):
            made.append(name)
        else:
            shutil.copytree(src, dst)
            made.append(name + "(copied)")

    # tokenizer: reuse the files that live inside text_encoder
    tok_src = os.path.join(FULL, "text_encoder")
    tok_dst = os.path.join(COMBINED, "tokenizer")
    os.makedirs(tok_dst, exist_ok=True)
    for fn in ("vocab.json", "merges.txt", "tokenizer_config.json",
               "special_tokens_map.json"):
        s = os.path.join(tok_src, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(tok_dst, fn))
    made.append("tokenizer")

    with open(os.path.join(COMBINED, "model_index.json"), "w", encoding="utf-8") as fh:
        json.dump(MODEL_INDEX, fh, indent=2)
    made.append("model_index.json")

    print(f"  combined pipeline at {COMBINED}")
    print(f"    components: {', '.join(made)}")
    return COMBINED


def rel(a, b):
    a, b = a.float().double(), b.float().double()
    return ((a - b).norm() / b.norm()).item()


def quantize_linears_inplace(model, bits, blocksize=64):
    import bitsandbytes as bnb

    n = 0
    stored = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data
        if bits == 4:
            packed, qs = bnb.functional.quantize_4bit(w.detach().float(),
                                                      quant_type="nf4",
                                                      blocksize=blocksize,
                                                      compress_statistics=True)
            rec = bnb.functional.dequantize_4bit(packed, qs).reshape(w.shape)
            stored += packed.numel() + sum(
                v.numel() * v.element_size() for v in qs.as_dict(packed=True).values())
        elif bits == 2:
            packed, absmax, shape = Q2.quant2bit(w)
            rec = Q2.dequant2bit(packed, absmax, shape)
            stored += packed.numel() + absmax.numel() * 4
        else:
            raise ValueError(bits)
        with torch.no_grad():
            mod.weight.copy_(rec.reshape(w.shape).to(w.dtype))
        n += 1
    return n, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance", type=float, default=7.5)
    a = ap.parse_args()

    from diffusers import StableDiffusionPipeline

    print("assembling a complete SD1.5 from the two halves on disk")
    pipe_dir = build_combined()
    print()

    print(f"loading pipeline")
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(pipe_dir, torch_dtype=torch.float32,
                                                   safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    print(f"  {time.time()-t0:.1f}s")
    print(f"  prompt: {a.prompt!r}")
    print(f"  {a.steps} steps, {a.size}x{a.size}, guidance {a.guidance}, seed {a.seed}")
    print()

    gen = lambda: torch.Generator().manual_seed(a.seed)

    print("rendering fp32 reference")
    t0 = time.time()
    with torch.no_grad():
        ref_img = pipe(a.prompt, num_inference_steps=a.steps, guidance_scale=a.guidance,
                       height=a.size, width=a.size,
                       generator=gen()).images[0]
    print(f"  {time.time()-t0:.1f}s")
    ref_t = torch.from_numpy(np.asarray(ref_img)).float() / 255.0
    ref_img.save(os.path.join(HERE, "_ab_fp32.png"))
    print()

    results = {}
    for bits in (4, 2):
        print(f"quantizing the UNet's Linear layers to {bits}-bit and re-rendering")
        # reload the UNet so each quantization starts from the original weights
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(os.path.join(BASE, "unet"),
                                                    torch_dtype=torch.float32)
        unet.eval()
        t0 = time.time()
        n, stored = quantize_linears_inplace(unet, bits)
        print(f"  {n} layers in {time.time()-t0:.0f}s, quantized storage "
              f"{stored/1e6:.1f} MB")
        pipe.unet = unet
        t0 = time.time()
        with torch.no_grad():
            img = pipe(a.prompt, num_inference_steps=a.steps,
                       guidance_scale=a.guidance, height=a.size, width=a.size,
                       generator=gen()).images[0]
        print(f"  render {time.time()-t0:.1f}s")
        t = torch.from_numpy(np.asarray(img)).float() / 255.0
        img.save(os.path.join(HERE, f"_ab_{bits}bit.png"))
        results[bits] = (t, stored, n)
        print(f"  pixel difference vs fp32: rel {rel(t, ref_t):.6f}  "
              f"max |d| {float((t-ref_t).abs().max()):.6f}")
        print()

    print("=" * 84)
    print("RESULT -- end-to-end images, SD1.5, Linear layers quantized only")
    print("=" * 84)
    print(f"  scope: {results[4][2]} Linear layers = 31.4% of the UNet; the other 68.5%")
    print(f"         is Conv2d which bitsandbytes cannot quantize. VAE and text encoder")
    print(f"         stay fp32.")
    print()
    print(f"  {'format':>8s} {'storage MB':>11s} {'vs fp32 (rel)':>14s} "
          f"{'max pixel diff':>15s}")
    for bits in (4, 2):
        t, stored, n = results[bits]
        print(f"  {bits}-bit   {stored/1e6:11.1f} {rel(t, ref_t):14.6f} "
              f"{float((t-ref_t).abs().max()):15.4f}")
    d = rel(results[2][0], results[4][0])
    print()
    print(f"  2-bit against 4-bit: rel {d:.6f}  "
          f"max |d| {float((results[2][0]-results[4][0]).abs().max()):.4f}")
    print(f"  storage ratio 2/4: {results[2][1]/results[4][1]:.3f}")
    print()
    print("  images written next to this script: _ab_fp32.png, _ab_4bit.png, _ab_2bit.png")
    print("  A numeric difference and a visible one are different things; the numbers")
    print("  bound the change, the files show whether it matters.")
    return 0


import numpy as np  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
