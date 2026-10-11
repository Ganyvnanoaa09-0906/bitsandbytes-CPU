"""Replace the first-order Euler sampler with a second-order multistep one, same model.

The step count is not the only knob. The pipeline ships with
FlowMatchEulerDiscreteScheduler, which is first order: its local error is O(h^2) per step
and it pays one model evaluation per step. DPMSolverMultistepScheduler is second order
and *also* pays one evaluation per step (it reuses the previous steps' derivatives), so
for the same number of model calls it should land closer to the same image -- which is
the training-free version of "fewer steps", as opposed to distilling a new model.

Two things have to be handled for the swap to be legal:

  * the pipeline computes a dynamic time shift and passes `mu=` to
    scheduler.set_timesteps(); DPM++ has no such argument. The exact sigma schedule the
    Euler scheduler produces is therefore read off it and handed to DPM++ verbatim, so
    the two runs denoise along the same schedule and only the solver differs.
  * DPM++ must be told the model predicts a flow (`prediction_type="flow_prediction"`),
    otherwise it interprets the output as epsilon and the step math is wrong.

Usage:
    python qwen21_solver_ab.py --steps 12 --size 384 --tag dpm12
    python qwen21_solver_ab.py --steps 20 --size 384 --tag euler20 --solver euler
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


def build(a):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    transformer = L.load_quantized(a.ckpt, "transformer")
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    euler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=euler, vae=None, text_encoder=None,
                               processor=processor, transformer=transformer)
    return pipe, euler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--embeds", default=os.path.join(HERE, "test_embeds.pt"))
    ap.add_argument("--prompt", default="a red panda astronaut floating in space, "
                                        "photorealistic, cinematic lighting, earth in "
                                        "the background, detailed fur")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--solver", choices=("euler", "dpm++"), default="dpm++")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)

    pipe, euler = build(a)
    blob = torch.load(a.embeds, weights_only=False)
    rec = blob["prompts"][a.prompt]

    # The schedule the pipeline itself would use, so both runs share it exactly.
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    seq_len = (a.size // (pipe.vae_scale_factor * 2)) ** 2
    mu = calculate_shift(seq_len,
                         euler.config.get("base_image_seq_len", 256),
                         euler.config.get("max_image_seq_len", 4096),
                         euler.config.get("base_shift", 0.5),
                         euler.config.get("max_shift", 1.15))
    sigmas = np.linspace(euler.sigma_max, euler.sigma_min, a.steps + 1)[:-1]
    euler.set_timesteps(a.steps, device="cpu", sigmas=sigmas, mu=mu)
    sigmas_used = euler.sigmas.clone()

    if a.solver == "dpm++":
        from diffusers import DPMSolverMultistepScheduler
        dpm = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
            prediction_type="flow_prediction",
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_flow_sigmas=True,
            flow_shift=1.0,
            final_sigmas_type="zero",
            lower_order_final=True,
        )
        dpm.set_timesteps(a.steps, device="cpu", sigmas=sigmas_used[:-1])
        pipe.scheduler = dpm
        print(f"solver dpm++ order={dpm.order} sigmas {len(dpm.sigmas)}", flush=True)
    else:
        print(f"solver euler sigmas {len(sigmas_used)}", flush=True)

    print(f"size {a.size} steps {a.steps} seq_len {seq_len} mu {mu:.4f} "
          f"tag {a.tag}", flush=True)
    t0 = time.perf_counter()
    out = pipe(true_cfg_scale=1.0, height=a.size, width=a.size,
               num_inference_steps=a.steps, output_type="latent",
               generator=torch.Generator().manual_seed(a.seed),
               prompt_embeds=rec["embeds"], prompt_embeds_mask=rec.get("mask"),
               sigmas=sigmas_used)
    dt = time.perf_counter() - t0
    lat = out.images if not torch.is_tensor(out.images) else out.images
    lat = lat.detach().float()
    print(f"\n{a.steps} steps in {dt:.1f} s -> {dt/a.steps:.1f} s/step")
    print(f"latent: shape {tuple(lat.shape)} mean {lat.mean():.4f} "
          f"std {lat.std():.4f} finite {bool(torch.isfinite(lat).all())}")
    path = os.path.join(HERE, f"latent_{a.tag}.pt")
    torch.save(lat, path)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
