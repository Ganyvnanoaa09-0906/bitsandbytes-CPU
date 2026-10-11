"""Render one image and save it, so a sampler or step-count change can be looked at.

Numbers do not settle a quality question on a diffusion model: the sampler is chaotic, so
a change can be negligible in every pixel statistic and still be visible, or large in
statistics and invisible. This renders one configuration to a PNG and stops.

The VAE is loaded last and only for the decode, because it is fp32 (1.35 GB) and nothing
before the decode needs it.

Usage:
    python qwen21_render_one.py --tag euler20 --steps 20 --size 384
    python qwen21_render_one.py --tag dpm12   --steps 12 --size 384 --solver dpm++
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_load as L  # noqa: E402

torch.set_grad_enabled(False)


def make_matched_dpm(euler, steps, mu):
    """A second-order solver denoising along the SAME sigma schedule as the Euler run.

    Two obstacles, both from the pipeline's calling convention rather than from the maths:

      * it always calls `scheduler.set_timesteps(...)`, passing `mu=` for the dynamic time
        shift, and DPMSolverMultistepScheduler accepts neither `mu` nor `sigmas`. The
        schedule is therefore built here first and the pipeline's call is made a no-op.
      * the dynamic shift has to be translated. The Euler scheduler applies
        `shift = exp(mu)` to a linear sigma ramp when `time_shift_type == "exponential"`,
        which is exactly what `flow_shift` does for DPM++, so `flow_shift = exp(mu)` lands
        on the same schedule. The two are printed side by side so the claim is checkable
        rather than assumed.
    """
    import math

    from diffusers import DPMSolverMultistepScheduler

    shift = math.exp(mu)

    class MatchedDPM(DPMSolverMultistepScheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None,
                          mu=None, **kw):
            # Only the pipeline's call carries sigmas=/mu=; swallowing that one is the
            # point. The first version swallowed every call, including the one that
            # builds the schedule below, and `step()` then reported "Number of inference
            # steps is 'None'".
            if sigmas is not None or mu is not None:
                return None
            return super().set_timesteps(num_inference_steps, device=device, **kw)

    dpm = MatchedDPM(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
        beta_schedule="linear", prediction_type="flow_prediction",
        algorithm_type="dpmsolver++", solver_order=2, use_flow_sigmas=True,
        flow_shift=shift, final_sigmas_type="zero", lower_order_final=True)
    dpm.set_timesteps(steps, device="cpu")
    return dpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--embeds", default=os.path.join(HERE, "test_embeds.pt"))
    ap.add_argument("--prompt", default="a red panda astronaut floating in space, "
                                        "photorealistic, cinematic lighting, earth in "
                                        "the background, detailed fur")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--solver", choices=("euler", "dpm++"), default="euler")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="true_cfg_scale. The model is guidance-distilled and the "
                         "pipeline default is 1.0 (no CFG, one forward per step); above "
                         "1 it runs a second forward on the negative prompt, so the step "
                         "costs about twice as much")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--no-vae-tiling", action="store_true")
    ap.add_argument("--latent-only", action="store_true",
                    help="stop after the denoising loop; the latent is saved either way, "
                         "and the decode is where a wrong VAE import classically bites "
                         "after eleven minutes of denoising")
    a = ap.parse_args()
    # Create it here rather than trusting the caller: the first two attempts at this A/B
    # each lost a completed denoise (11 and 18 minutes of CPU) to
    # "Parent directory qwen21_ab does not exist" from the torch.save that follows it.
    os.makedirs(a.outdir, exist_ok=True)

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    transformer = L.load_quantized(a.ckpt, "transformer")
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, text_encoder=None,
                               processor=processor, transformer=transformer)
    blob = torch.load(a.embeds, weights_only=False)
    rec = blob["prompts"][a.prompt]
    neg = None
    if a.negative_prompt:
        # Negative prompts live in their own section, and they have lived in two places:
        # qwen21_encode.py first filed them under prompts["negative"], which is easy to
        # mistake for a prompt actually called "negative". Both are accepted so the
        # embeds files already on disk keep working.
        negs = dict(blob.get("negative", {}))
        negs.update(blob.get("prompts", {}).get("negative", {}))
        if a.negative_prompt not in negs:
            raise SystemExit(f"negative prompt not in {a.embeds}; encode it with "
                             f"qwen21_encode.py --negative-prompt ...")
        neg = negs[a.negative_prompt]

    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    seq_len = (a.size // (pipe.vae_scale_factor * 2)) ** 2
    mu = calculate_shift(seq_len,
                         scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    sigmas = np.linspace(scheduler.sigma_max, scheduler.sigma_min, a.steps + 1)[:-1]
    scheduler.set_timesteps(a.steps, device="cpu", sigmas=sigmas, mu=mu)
    sigmas_used = scheduler.sigmas.clone()

    if a.solver == "dpm++":
        dpm = make_matched_dpm(scheduler, a.steps, mu)
        print(f"[render] euler sigmas : {[f'{float(s):.4f}' for s in sigmas_used[:4]]}"
              f" ... {float(sigmas_used[-1]):.4f}")
        print(f"[render] dpm++ sigmas : "
              f"{[f'{float(s):.4f}' for s in dpm.sigmas[:4]]} ... "
              f"{float(dpm.sigmas[-1]):.4f}   (flow_shift={np.exp(mu):.4f})")
        pipe.scheduler = dpm

    print(f"[render] {a.tag}: {a.size}px {a.steps} steps solver={a.solver} "
          f"cfg={a.cfg}{' +negative' if neg is not None else ''} "
          f"mu={mu:.4f} seq_len={seq_len}", flush=True)
    t0 = time.perf_counter()
    call = dict(true_cfg_scale=a.cfg, height=a.size, width=a.size,
                num_inference_steps=a.steps, output_type="latent",
                generator=torch.Generator().manual_seed(a.seed),
                prompt_embeds=rec["embeds"], prompt_embeds_mask=rec.get("mask"),
                sigmas=sigmas_used)
    if neg is not None:
        call["negative_prompt_embeds"] = neg["embeds"]
        call["negative_prompt_embeds_mask"] = neg.get("mask")
    if a.cfg > 1.0 and neg is None:
        print("[render] WARNING: true_cfg_scale > 1 without a negative prompt; the "
              "pipeline ignores guidance in that case and you pay nothing for it",
              flush=True)
    out = pipe(**call)
    dt = time.perf_counter() - t0
    lat = out.images if torch.is_tensor(out.images) else out.images
    print(f"[render] denoise {dt:.1f} s ({dt/a.steps:.1f} s/step); "
          f"latent {tuple(lat.shape)} std {lat.float().std():.4f}", flush=True)

    lat_saved = os.path.join(a.outdir, f"latent_{a.tag}.pt")
    torch.save(lat, lat_saved)
    print(f"[render] latent saved {lat_saved}", flush=True)
    print(f"[render] decode with: python qwen21_decode.py --latent {lat_saved} "
          f"--size {a.size} --out {os.path.join(a.outdir, f'render_{a.tag}.png')}",
          flush=True)
    # The decode lives in qwen21_decode.py, not here. Three runs in a row finished their
    # denoising loop and then died in a decode written from memory (packed latents,
    # latents_mean/std instead of scaling_factor, and a 5D video-style VAE). Splitting it
    # out means a wrong decode costs 30 s, not the 11-18 minutes that came before it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
