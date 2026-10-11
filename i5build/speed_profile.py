"""Where does the time actually go, and is bit width even the right question?

The complaint is that CPU diffusion is slow. Before optimizing anything it is worth
knowing whether the cost is arithmetic, memory traffic, or simply the number of times
the network is evaluated -- because those have completely different levers, and only
one of them is quantization.

This measures, on the real SD1.5 pipeline:
  * the end-to-end split: text encoder, the UNet loop, the VAE decoder
  * the UNet's internal split by module type (conv / linear / attention / norm)
  * the same prompt at 20 steps against 4 steps with LCM-LoRA, which is the
    few-step distillation the ecosystem actually uses to make sampling affordable

The point of the last one: if 4 steps match 20, sampling cost is a step-count problem
and no amount of bit-shaving competes with it.
"""
import argparse
import os
import time

import torch

torch.set_grad_enabled(False)

MODEL = r"D:\work\textmodel\counterfeit-v3-diffusers"
LCM_SD15 = (r"C:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5")
PROMPT = "a photo of an astronaut riding a horse on the moon"


def profile_unet(pipe, steps, label, res=512, seed=1234):
    """Time the UNet loop and break the time down by module class."""
    totals = {}
    counts = {}
    hooks = []

    def mk(cls):
        def hook(mod, inp, out):
            t0 = time.perf_counter()
            mod._t0 = t0
            # accumulate in the module's own forward is not possible from a hook, so
            # measure with pre/post hooks instead
        return hook

    pre, post = [], []

    def mkpre(cls):
        def hook(mod, inp):
            mod.__dict__["_t"] = time.perf_counter()
        return hook

    def mkpost(cls):
        def hook(mod, inp, out):
            t = getattr(mod, "_t", None)
            if t is None:
                return
            totals[cls] = totals.get(cls, 0.0) + (time.perf_counter() - t)
            counts[cls] = counts.get(cls, 0) + 1
        return hook

    import torch.nn as nn
    for mod in pipe.unet.modules():
        cls = type(mod).__name__
        if cls in ("Conv2d", "Linear", "GroupNorm", "Attention", "GEGLU", "SiLU",
                   "Upsample", "Downsample", "FeedForward"):
            pre.append(mod.register_forward_pre_hook(mkpre(cls)))
            post.append(mod.register_forward_hook(mkpost(cls)))

    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, 4, res // 8, res // 8, generator=g)
    ctx = pipe.encode_prompt(PROMPT, device="cpu", num_images_per_prompt=1,
                             do_classifier_free_guidance=True)[0]

    t0 = time.perf_counter()
    pipe.unet(lat, 500.0, encoder_hidden_states=ctx)
    warm = time.perf_counter() - t0

    totals.clear()
    counts.clear()
    t0 = time.perf_counter()
    for i in range(steps):
        pipe.unet(lat, float(999 - i * (1000 // steps)), encoder_hidden_states=ctx)
    dt = time.perf_counter() - t0
    for h in pre + post:
        h.remove()

    print(f"\n  [{label}] {steps} steps in {dt:6.2f} s   "
          f"{dt/steps:5.2f} s/step   (warmup forward {warm:.2f} s)")
    print(f"    {label} internal breakdown over {steps} steps:")
    for cls, secs in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"      {cls:<14} {secs:7.2f} s  {100*secs/dt:5.1f}%  "
              f"({counts[cls]} calls)")
    return dt, dt / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--lcm", default=LCM_SD15)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lcm-steps", type=int, default=4)
    a = ap.parse_args()

    from diffusers import StableDiffusionPipeline, LCMScheduler

    print(f"model: {a.model}")
    print(f"res  : {a.res}x{a.res}   torch threads: {torch.get_num_threads()}")

    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False).to("cpu")
    print(f"load : {time.perf_counter()-t0:.1f} s")

    # --- text encoder -------------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(3):
        pipe.encode_prompt(PROMPT, device="cpu", num_images_per_prompt=1,
                           do_classifier_free_guidance=True)
    te = (time.perf_counter() - t0) / 3
    print(f"text encoder: {te*1000:.0f} ms")

    # --- baseline UNet ------------------------------------------------------------
    dt, per_step = profile_unet(pipe, a.steps, "fp32 SD1.5", a.res)

    # --- VAE ----------------------------------------------------------------------
    g = torch.Generator().manual_seed(7)
    lat = torch.randn(1, 4, a.res // 8, a.res // 8, generator=g)
    t0 = time.perf_counter()
    pipe.vae.decode(lat / pipe.vae.config.scaling_factor)
    vae = time.perf_counter() - t0
    print(f"VAE decode  : {vae:.2f} s")

    base20 = per_step * 20
    print(f"\n=== baseline: one {a.res}x{a.res} image ===")
    print(f"  text {te:.2f} + unet {per_step:.2f}x20 = {base20:.1f} + vae {vae:.2f} "
          f"-> {te+base20+vae:.1f} s/image")

    # --- LCM ----------------------------------------------------------------------
    if not os.path.isdir(a.lcm):
        print(f"\nno LCM-LoRA at {a.lcm}; skipping the few-step comparison")
        return
    print(f"\nloading LCM-LoRA from {a.lcm} ...")
    pipe.load_lora_weights(a.lcm)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    t0 = time.perf_counter()
    img = pipe(PROMPT, num_inference_steps=a.lcm_steps, guidance_scale=1.0,
               height=a.res, width=a.res,
               generator=torch.Generator().manual_seed(1234)).images[0]
    lcm_total = time.perf_counter() - t0
    print(f"  LCM {a.lcm_steps} steps, whole pipeline: {lcm_total:.1f} s/image")
    print(f"  speedup vs 20-step baseline: {(te+base20+vae)/lcm_total:.1f}x")

    base = pipe(PROMPT, num_inference_steps=20, guidance_scale=7.5,
                height=a.res, width=a.res,
                generator=torch.Generator().manual_seed(1234)).images[0]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed_cmp")
    os.makedirs(out, exist_ok=True)
    base.save(os.path.join(out, f"_20step_fp32_{a.res}.png"))
    img.save(os.path.join(out, f"_lcm{a.lcm_steps}step_fp32_{a.res}.png"))
    print(f"  wrote {out}\\_20step_fp32_{a.res}.png and _lcm{a.lcm_steps}step_...")


if __name__ == "__main__":
    main()
