"""Generate with a reference image, without holding the text encoder in memory.

Qwen-Image-2.1 conditions on a reference picture twice over:

  * the text encoder is a vision model, so it reads the reference as visual context and
    the prompt embeds come back longer (the reference's tokens are in there);
  * the generation call also VAE-encodes it, because the condition image is part of the
    latent layout the transformer sees, with its own entry in the per-region RoPE.

The second half cannot be pre-computed away -- the sequence layout depends on the image
being present at call time. The first half can: `qwen21_encode.py --image` writes the
embeds, the attention mask and the `image_pad_mask` that `encode_prompt` returns third.

That matters because the three components together are ~9 GB resident, and this box has
16 GB with 7 free while anything else runs. So the encoder runs alone once, and this
script replaces `pipe.encode_prompt` with the saved triple so the generation process only
needs the transformer (3.4 GB) and the VAE (1.35 GB).

Usage:
    python qwen21_ref.py --embeds qwen21_embeds_ref.pt --prompt "<same prompt>" \
        --image ref_keqing.png --tag ref512 --size 512 --steps 20
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
    ap.add_argument("--embeds", required=True, help="written by qwen21_encode.py --image")
    ap.add_argument("--prompt", required=True, help="must match a key in that file")
    ap.add_argument("--image", default=None,
                    help="used only when the embeds file does not record its own processed "
                         "references; the two must be the SAME image, or the vision tokens "
                         "and the VAE latents describe different pictures and the sampler "
                         "returns NaN")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=1.0)
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
    from diffusers.models import AutoencoderKLQwenImage21
    from PIL import Image as PILImage
    from transformers import Qwen3VLProcessor

    blob = torch.load(a.embeds, weights_only=False)
    if a.prompt not in blob["prompts"]:
        raise SystemExit(f"prompt not found in {a.embeds}")
    rec = blob["prompts"][a.prompt]
    if rec.get("image_pad_mask") is None:
        print("[ref] note: image_pad_mask is None in the embeds; if the render looks "
              "wrong, re-encode with a current qwen21_encode.py --image", flush=True)

    transformer = L.load_quantized(a.ckpt, "transformer")
    print("[ref] vae (fp32, needed for the condition image latents)", flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(a.src, "vae"))
    vae.eval()
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=None,
                               processor=processor, transformer=transformer)

    # The whole point: the pipeline asks for the embeds itself, and would need the text
    # encoder to answer. Hand it the saved triple instead.
    captured = (rec["embeds"], rec.get("mask"), rec.get("image_pad_mask"))

    def fake_encode_prompt(*args, **kwargs):
        return captured

    pipe.encode_prompt = fake_encode_prompt

    ref_paths = list(blob.get("ref_images") or ([] if a.image is None else [a.image]))
    if not ref_paths:
        raise SystemExit("no reference image: neither blob['ref_images'] nor --image")
    refs = [PILImage.open(p).convert("RGB") for p in ref_paths]
    for p, im in zip(ref_paths, refs):
        patches = (im.size[0] // 16) * (im.size[1] // 16)
        print(f"[ref] reference {im.size}  patches={patches}  ({os.path.basename(p)})",
              flush=True)
    print(f"[ref] output {a.size}px {a.steps} steps cfg={a.cfg}", flush=True)
    ws, peak = L.rss_gb()
    print(f"[ref] resident {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    t0 = time.perf_counter()
    out = pipe(prompt=a.prompt, image=refs, output_resolution=a.size,
               height=a.size, width=a.size, num_inference_steps=a.steps,
               true_cfg_scale=a.cfg, output_type="latent",
               generator=torch.Generator().manual_seed(a.seed))
    dt = time.perf_counter() - t0
    lat = out.images if torch.is_tensor(out.images) else out.images
    path = os.path.join(a.outdir, f"latent_{a.tag}.pt")
    torch.save(lat, path)
    print(f"[ref] {a.steps} steps in {dt:.1f} s ({dt/a.steps:.1f} s/step); "
          f"latent {tuple(lat.shape)} std {lat.float().std():.4f}", flush=True)
    print(f"[ref] saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
