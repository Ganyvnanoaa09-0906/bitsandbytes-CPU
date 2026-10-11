"""Decode a saved latent to a PNG, the way the pipeline itself does it.

Splitting the decode out of the render is not cosmetic. Three renders in a row finished
their denoising loop (11-18 minutes of CPU each) and then died in the decode, because the
decode was written from memory instead of read off the pipeline:

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(vae.dtype)
    latents = latents * latents_std.view(1, z, 1, 1, 1) + latents_mean.view(1, z, 1, 1, 1)
    image = vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")

Three things were wrong, and each one raises a different error:
  * the denoise output is PACKED (1, T, C) and has to go through _unpack_latents
  * this VAE normalises with latents_mean/latents_std, not config.scaling_factor (which
    does not exist here -> FrozenDict AttributeError)
  * it is a 5D video-style VAE, so the image is frame 0 of the decode

Usage:
    python qwen21_decode.py --latent qwen21_ab/latent_euler20.pt --size 384 --out qwen21_ab/render_euler20.png
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
    ap.add_argument("--latent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from diffusers.models import AutoencoderKLQwenImage21
    from transformers import Qwen3VLProcessor

    # _unpack_latents needs the transformer config, and the pipeline constructor builds a
    # system prompt through the processor, so both have to be real -- processor=None dies
    # inside __init__ on apply_chat_template, before a single latent is touched.
    transformer = L.load_quantized(a.ckpt, "transformer")
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, text_encoder=None,
                               processor=processor, transformer=transformer)

    print(f"[decode] vae (fp32) ...", flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(a.src, "vae"))
    vae.eval()
    pipe.vae = vae
    try:
        vae.enable_tiling()
        print("[decode] vae tiling enabled", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[decode] vae tiling unavailable: {type(exc).__name__}: {exc}", flush=True)

    lat = torch.load(a.latent, weights_only=False)
    print(f"[decode] latent {tuple(lat.shape)} from {a.latent}", flush=True)
    lat = pipe._unpack_latents(lat, a.size, a.size, pipe.vae_scale_factor)
    lat = lat.to(vae.dtype)
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    lat = lat * std + mean
    print(f"[decode] unpacked to {tuple(lat.shape)}", flush=True)

    t0 = time.perf_counter()
    img = vae.decode(lat, return_dict=False)[0][:, :, 0]
    dt = time.perf_counter() - t0
    pil = pipe.image_processor.postprocess(img, output_type="pil")[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pil.save(a.out)
    print(f"[decode] {dt:.1f} s -> {a.out}  size {pil.size}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
