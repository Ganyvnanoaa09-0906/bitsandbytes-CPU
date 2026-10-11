"""qwen21_render.py -- render an image with the 4-bit Qwen-Image-2.1 pipeline.

This is the step the earlier work stopped short of: the quantizer produces 4-bit
shards, the loader rebuilds module classes on the meta device, and the forward check
proves a single forward computes -- but none of that produces an image, and an image is
the only thing that says whether the model is intact.

The pipeline is assembled from pre-built components rather than from_pretrained,
because two of the five are ours: the transformer and the text encoder come back as
quantized modules, and from_pretrained would try to load ordinary ones.

`--staged` encodes the prompt, releases the text encoder, and only then samples. The
text encoder is 4.2 GB of the ~9 GB the three components need, and it is used exactly
once per image, so on a 16 GB machine that is the difference between fitting and not.

Usage:
    python qwen21_render.py --prompt "a cat" --height 1024 --width 1024 --steps 20
"""
import argparse
import gc
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_load as L  # noqa: E402

torch.set_grad_enabled(False)

# Both live in qwen21_load so the encoder and the renderer cannot disagree about how a
# component is rebuilt -- they now run as separate processes.
rss_gb = L.rss_gb
load_quantized = L.load_quantized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--prompt", default="a red panda astronaut floating in space, "
                                        "photorealistic, cinematic lighting")
    ap.add_argument("--negative", default="")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "qwen21_out"))
    ap.add_argument("--staged", action="store_true",
                    help="release the text encoder before sampling (saves ~4.2 GB)")
    ap.add_argument("--embeds", default="",
                    help="prompt embeddings written by qwen21_encode.py; the text "
                         "encoder is then never loaded, which is what lets an 11.8 GB "
                         "machine render at 1024x1024")
    ap.add_argument("--no-vae-tiling", action="store_true",
                    help="decode the VAE in one pass; costs several hundred MB of "
                         "peak memory at 1024x1024")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    print(f"Qwen-Image-2.1  4-bit")
    print(f"  prompt   : {a.prompt[:70]}")
    print(f"  size     : {a.width}x{a.height}, {a.steps} steps, cfg {a.cfg}, "
          f"seed {a.seed}, threads {a.threads}")
    print(f"  ckpt     : {a.ckpt}")
    print(flush=True)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    # AutoencoderKLQwenImage21 is exported from diffusers.models but not from the
    # top-level namespace, which is where the other four components live.
    from diffusers.models import AutoencoderKLQwenImage21
    from transformers import Qwen3VLProcessor

    print("[1/5] transformer (4-bit)", flush=True)
    transformer = load_quantized(a.ckpt, "transformer")

    text_encoder = None
    embeds_blob = None
    if a.embeds:
        print(f"[2/5] using pre-encoded embeddings from {a.embeds}", flush=True)
        embeds_blob = torch.load(a.embeds, weights_only=False)
        if a.prompt not in embeds_blob["prompts"]:
            print(f"    the file holds {len(embeds_blob['prompts'])} prompt(s); "
                  f"this one is not among them:")
            for p in embeds_blob["prompts"]:
                print(f"      {p[:70]}")
            return 2
    else:
        print("[2/5] text encoder (4-bit)", flush=True)
        text_encoder = load_quantized(a.ckpt, "text_encoder")

    print("[3/5] processor + scheduler", flush=True)
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))

    # The VAE is loaded LAST, after the prompt has been encoded and the text encoder
    # released. Loading all three up front peaks at 3.42 + 4.22 + 1.35 = 8.99 GB, which
    # does not fit an 11.8 GB machine that is already running Windows; deferring it puts
    # the peak at transformer + text encoder = 7.64 GB and the sampling footprint at
    # 3.42 + 1.35 = 4.77 GB. The pipeline accepts vae=None and its `latent_channels`
    # fallback for the missing VAE is 64, which is exactly this checkpoint's z_dim, so
    # nothing downstream sees a different number.
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None,
                               text_encoder=text_encoder, processor=processor,
                               transformer=transformer)
    ws, _ = rss_gb()
    print(f"    resident with transformer loaded: {ws:.2f} GB", flush=True)

    prompt_embeds = prompt_embeds_mask = None
    if embeds_blob is not None:
        rec = embeds_blob["prompts"][a.prompt]
        prompt_embeds, prompt_embeds_mask = rec["embeds"], rec["mask"]
        print(f"    embeds {tuple(prompt_embeds.shape)}", flush=True)
    elif a.staged:
        print("[*] staged: encoding the prompt, then releasing the text encoder",
              flush=True)
        t0 = time.time()
        prompt_embeds, prompt_embeds_mask, _ = pipe.encode_prompt(
            a.prompt, device="cpu", num_images_per_prompt=1)
        print(f"    encoded in {time.time()-t0:.1f}s, embeds "
              f"{tuple(prompt_embeds.shape)}", flush=True)
        pipe.text_encoder = None
        del text_encoder
        gc.collect()
        ws, _ = rss_gb()
        print(f"    resident after release: {ws:.2f} GB", flush=True)

    print("[4/5] vae (fp32; it has no Linear layers to quantize)", flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(a.src, "vae"),
                                                   torch_dtype=torch.float32)
    vae.eval()
    pipe.vae = vae
    # Tiled decoding keeps the decoder from materialising full-resolution feature maps:
    # at 1024x1024 a single (1, 128, 1024, 1024) fp32 activation is 537 MB, and the
    # decoder holds several. On the 4500U there is room; on the 11.8 GB i5 there is not,
    # and the whole render would be lost at the last step -- after two hours of sampling
    # -- rather than at the start.
    if not a.no_vae_tiling:
        try:
            vae.enable_tiling()
            print("    vae tiling enabled", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    vae tiling unavailable: {type(exc).__name__}: {exc}",
                  flush=True)
    ws, _ = rss_gb()
    print(f"    resident {ws:.2f} GB", flush=True)

    state = {"t": [], "last": None}

    def cb(pipeline, step, timestep, kwargs):
        now = time.perf_counter()
        if state["last"] is not None:
            state["t"].append(now - state["last"])
        state["last"] = now
        n = len(state["t"])
        if n and (n % 2 == 0 or n == a.steps):
            avg = sum(state["t"]) / len(state["t"])
            left = (a.steps - n) * avg
            print(f"      step {n:3d}/{a.steps}  {avg:6.2f} s/step  "
                  f"eta {left/60:5.1f} min", flush=True)
        return kwargs

    print(f"[5/5] sampling", flush=True)
    t0 = time.time()
    kw = dict(negative_prompt=a.negative or None,
              true_cfg_scale=a.cfg, height=a.height, width=a.width,
              num_inference_steps=a.steps, output_type="pil",
              generator=torch.Generator().manual_seed(a.seed),
              callback_on_step_end=cb)
    # Branched on whether embeddings exist, not on which flag produced them. The first
    # version keyed this off `--staged`, so `--embeds` alone left `prompt` in the call
    # and the pipeline went to encode it with a text encoder that was deliberately
    # never loaded.
    if prompt_embeds is not None:
        kw["prompt_embeds"] = prompt_embeds
        if prompt_embeds_mask is not None:
            kw["prompt_embeds_mask"] = prompt_embeds_mask
    else:
        kw["prompt"] = a.prompt
    out = pipe(**kw)
    img = out.images[0]
    dt = time.time() - t0
    ws, peak = rss_gb()
    name = f"qwen21_{a.width}x{a.height}_{a.steps}s_seed{a.seed}{a.tag}.png"
    path = os.path.join(a.out, name)
    img.save(path)
    print(f"\n  {dt/60:.1f} min total ({dt/a.steps:.1f} s/step), "
          f"resident {ws:.2f} GB, peak {peak:.2f} GB")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
