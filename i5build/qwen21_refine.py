"""Second stage of a two-stage render: upscale the latent and refine it, instead of
denoising the whole image at the target resolution from scratch.

The arithmetic this is meant to dodge: cost per step is set by the token count, and this
VAE compresses 16x spatially, so 512x512 is 1024 tokens and 1024x1024 is 4096 -- four
times the work per step, sixteen times for the attention term. Generating a picture at
512 and then spending only a few steps at 1024 should land close to the native 1024
result for a fraction of the cost:

    512@20 steps            31 min
    + 1024@4..6 refine      23..35 min        total 54..66 min
    native 1024@20 steps    117 min

The pieces, all of them borrowed from the pipeline rather than reinvented (three wrong
decodes in a row taught that lesson):

  * `_unpack_latents` to get (1, 64, 1, h, w) out of the packed (1, T, 64) the sampler
    produces, `_pack_latents` to put it back
  * the refine schedule is a TAIL of the normal one -- the sigmas from `--init-sigma`
    down to zero, resampled to the requested number of steps -- so the model is asked to
    denoise a partially noised image rather than to start over
  * the starting latent is `sigma * noise + (1 - sigma) * upscaled`, the flow-matching
    interpolation at that sigma, using the SAME generator seed as the first stage so the
    comparison against a native run keeps everything but the schedule fixed

Usage:
    python qwen21_refine.py --init qwen21_ab/latent_hard20.pt --tag hard_up \
        --from-size 512 --size 1024 --steps 6 --init-sigma 0.55
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_load as L  # noqa: E402

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="packed latent from the first stage")
    ap.add_argument("--from-size", type=int, required=True,
                    help="pixel size the first stage rendered at")
    ap.add_argument("--size", type=int, required=True, help="target pixel size")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--init-sigma", type=float, default=0.55,
                    help="sigma to re-noise the upscaled latent to; 0.55 keeps a bit "
                         "more than half of the low-res structure intact")
    ap.add_argument("--mode", choices=("bilinear", "nearest"), default="bilinear")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--embeds", required=True,
                    help="the same embeds file the first stage used; the refine has to be "
                         "conditioned on the same prompt or it is a different picture")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    from transformers import Qwen3VLProcessor

    transformer = L.load_quantized(a.ckpt, "transformer")
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    euler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=euler, vae=None, text_encoder=None,
                               processor=processor, transformer=transformer)

    low = torch.load(a.init, weights_only=False)
    print(f"[refine] stage-1 latent {tuple(low.shape)} from {a.init}", flush=True)
    x = pipe._unpack_latents(low, a.from_size, a.from_size, pipe.vae_scale_factor)
    print(f"[refine] unpacked {tuple(x.shape)}", flush=True)
    # This VAE is 5D (B, C, T, H, W) with T=1 for a still image, and torch's bilinear
    # interpolation refuses 5D input ("Got 5D input, but bilinear mode needs 4D input").
    # Folding T into the batch interpolates the spatial axes only -- a trilinear call
    # would also resample T and turn one frame into two.
    b, c, t, h, w = x.shape
    scale = a.size // a.from_size
    x4 = x.reshape(b * t, c, h, w).float()
    x4 = F.interpolate(x4, scale_factor=scale, mode=a.mode)
    x2 = x4.reshape(b, c, t, h * scale, w * scale)
    print(f"[refine] upscaled {tuple(x2.shape)} ({a.mode} x{scale})", flush=True)
    packed = pipe._pack_latents(x2, 1, x2.shape[1], x2.shape[2], x2.shape[3])
    print(f"[refine] repacked {tuple(packed.shape)}", flush=True)

    # The refine schedule: the tail of the schedule a normal run would use, resampled to
    # --steps points between --init-sigma and 0.
    seq_len = (a.size // (pipe.vae_scale_factor * 2)) ** 2
    mu = calculate_shift(seq_len,
                         euler.config.get("base_image_seq_len", 256),
                         euler.config.get("max_image_seq_len", 4096),
                         euler.config.get("base_shift", 0.5),
                         euler.config.get("max_shift", 1.15))
    full = np.linspace(euler.sigma_max, euler.sigma_min, 41)
    tail = full[full <= a.init_sigma]
    sigmas = np.linspace(tail[0], 0.0, a.steps + 1)[:-1]
    euler.set_timesteps(a.steps, device="cpu", sigmas=sigmas, mu=mu)
    print(f"[refine] sigma {float(sigmas[0]):.4f} -> 0 in {a.steps} steps "
          f"(mu={mu:.4f})", flush=True)

    gen = torch.Generator().manual_seed(a.seed)
    noise = torch.randn(packed.shape, generator=gen, dtype=torch.float32)
    s0 = float(sigmas[0])
    latents = s0 * noise + (1.0 - s0) * packed
    print(f"[refine] start latent: {s0:.3f}*noise + {1-s0:.3f}*upscaled", flush=True)

    blob = torch.load(a.embeds, weights_only=False)
    if a.prompt not in blob["prompts"]:
        raise SystemExit(f"prompt not found in {a.embeds}")
    rec = blob["prompts"][a.prompt]

    t0 = time.perf_counter()
    out = pipe(true_cfg_scale=1.0, height=a.size, width=a.size,
               num_inference_steps=a.steps, output_type="latent",
               latents=latents, sigmas=euler.sigmas.clone(),
               generator=torch.Generator().manual_seed(a.seed),
               prompt_embeds=rec["embeds"], prompt_embeds_mask=rec.get("mask"))
    dt = time.perf_counter() - t0
    lat = out.images if torch.is_tensor(out.images) else out.images
    path = os.path.join(a.outdir, f"latent_{a.tag}.pt")
    torch.save(lat, path)
    print(f"[refine] {a.steps} steps in {dt:.1f} s ({dt/a.steps:.1f} s/step); "
          f"latent {tuple(lat.shape)} std {lat.float().std():.4f}", flush=True)
    print(f"[refine] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
