"""Encode prompts with the 4-bit text encoder, and nothing else.

Why this is a separate program rather than a flag: the three components of
Qwen-Image-2.1 need 8.99 GB resident together, and the i5-10400 has 11.8 GB of which
about 7.4 GB is free. Splitting the work by lifetime rather than by convenience puts
each half well inside that budget, because the text encoder is used exactly once per
prompt and the transformer is used once per step:

    this program        text encoder only            4.22 GB resident
    qwen21_render.py    transformer + vae            4.77 GB resident

The interface between them is a small tensor file -- 30 tokens x 4096 floats -- so
nothing large has to survive the hand-off. Encoding a batch of prompts once and
rendering them many times also becomes the cheap path rather than an optimisation.

Usage:
    python qwen21_encode.py --ckpt <4bit> --src <original> --out embeds.pt \
        --prompt "..." --prompt "..."
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_load as L  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--prompt", action="append", required=True,
                    help="repeat for each prompt; the order defines the keys")
    ap.add_argument("--negative-prompt", default=None,
                    help="encoded into a separate 'negative' section; it only has any "
                         "effect when the render is called with --cfg > 1, because "
                         "Qwen-Image 2.1 is guidance-distilled and defaults to no CFG")
    ap.add_argument("--out", default="qwen21_embeds.pt")
    ap.add_argument("--image", action="append", default=None,
                    help="a reference image the text encoder reads as vision context. "
                         "The third return value (image_pad_mask) is saved too, because "
                         "the generation process cannot re-derive it without the text "
                         "encoder resident -- and both together are 9 GB on a 16 GB box")
    ap.add_argument("--image-long-side", type=int, default=768,
                    help="downscale a reference image so its long side is at most this. "
                         "The processor does NOT do it for you: its max_pixels is null and "
                         "longest_edge is 16777216, so a 2400x1080 screenshot arrives as "
                         "150x68 = 10200 patches and the 4-bit vision tower dies with "
                         "'mat1 and mat2 shapes cannot be multiplied (10200x4 and "
                         "1152x2304)'. 768 gives about 1056 patches")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"encoding {len(a.prompt)} prompt(s) with the 4-bit text encoder")
    for i, p in enumerate(a.prompt, 1):
        print(f"  [{i}] {p[:76]}")

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    print("[1/3] text encoder (4-bit)", flush=True)
    text_encoder = L.load_quantized(a.ckpt, "text_encoder")
    print("[2/3] processor + scheduler", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))

    # vae=None and transformer=None: encode_prompt reads only the processor, the text
    # encoder and the class-level prompt template, so neither is touched here. The
    # pipeline's own fallback for a missing VAE is latent_channels=64, which is this
    # checkpoint's z_dim, so building it this way does not change any number.
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None,
                               text_encoder=text_encoder, processor=processor,
                               transformer=None)
    ws, peak = L.rss_gb()
    print(f"[3/3] resident before encoding {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    images = None
    if a.image:
        from PIL import Image as PILImage
        images = []
        for p in a.image:
            im = PILImage.open(p).convert("RGB")
            longest = max(im.size)
            if longest > a.image_long_side:
                scale = a.image_long_side / longest
                new = (max(1, round(im.size[0] * scale)),
                       max(1, round(im.size[1] * scale)))
                print(f"  downscale {im.size} -> {new}  ({os.path.basename(p)})",
                      flush=True)
                im = im.resize(new, PILImage.LANCZOS)
            images.append(im)
        patches = [(im.size[0] // 16) * (im.size[1] // 16) for im in images]
        print(f"  {len(images)} reference image(s): {[im.size for im in images]}  "
              f"patches={patches}", flush=True)
        # Save the PROCESSED image, because the generation call has to see exactly the
        # same one. Handing the pipeline the original 2400x1080 while the embeds carry
        # tokens from a 768x346 copy is a layout mismatch, and the sampler answers it
        # with NaN -- a pure white decode, 20 steps of compute thrown away.
        ref_dir = os.path.abspath(a.out) + ".refs"
        os.makedirs(ref_dir, exist_ok=True)
        ref_paths = []
        for i, im in enumerate(images):
            fp = os.path.join(ref_dir, f"{i}.png")
            im.save(fp)
            ref_paths.append(fp)
        print(f"  saved processed reference(s) -> {ref_dir}", flush=True)

    out = {}
    for p in a.prompt:
        t0 = time.time()
        embeds, mask, image_pad_mask = pipe.encode_prompt(
            p, image=images, device="cpu", num_images_per_prompt=1)
        # encode_prompt deliberately returns mask=None when every position is valid
        # ("without padding there is nothing to mask"), which is the normal case for a
        # single prompt. Treating None as a failure would reject the common path.
        out[p] = {"embeds": embeds.clone(),
                  "mask": None if mask is None else mask.clone(),
                  "image_pad_mask": (None if image_pad_mask is None
                                     else image_pad_mask.clone())}
        print(f"    {tuple(embeds.shape)} mask="
              f"{'all-valid (None)' if mask is None else tuple(mask.shape)} "
              f"image_pad_mask="
              f"{'None' if image_pad_mask is None else tuple(image_pad_mask.shape)} "
              f"in {time.time()-t0:.1f}s", flush=True)

    negatives = {}
    if a.negative_prompt:
        t0 = time.time()
        embeds, mask, _ = pipe.encode_prompt(a.negative_prompt, device="cpu",
                                             num_images_per_prompt=1)
        negatives[a.negative_prompt] = {
            "embeds": embeds.clone(),
            "mask": None if mask is None else mask.clone()}
        print(f"    negative {tuple(embeds.shape)} in {time.time()-t0:.1f}s", flush=True)

    blob = {"prompts": out, "src": a.src, "ckpt": a.ckpt}
    if a.image and 'ref_paths' in dir():
        # The generation side must condition on these exact files, not on whatever path
        # the caller passes it later.
        blob["ref_images"] = ref_paths
    if negatives:
        # Its own top-level section, NOT a key inside "prompts": that dict is keyed by the
        # prompt text, so a nested entry is indistinguishable from a prompt that happens
        # to be named "negative".
        blob["negative"] = negatives
    torch.save(blob, a.out)
    ws, peak = L.rss_gb()
    print(f"\nwrote {a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")
    print(f"peak resident {peak:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
