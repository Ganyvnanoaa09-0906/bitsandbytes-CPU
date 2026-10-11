"""What does the TEACHER cost per training sample, at the STUDENT's resolution?

The distillation plan kept stalling on "the teacher takes 1.9 h per image", which is the
number for a 1024x1024 render. That is the wrong denominator: a rendered image is only
needed at the resolution the student will work at. This VAE compresses 16x and the
transformer's cost per step is (2 x 7.1e9 params x tokens), so the teacher's price for a
128x128 sample is roughly 1/40 of the price for a 1024x1024 one.

Two effects are measured rather than assumed:

  * tokens -> time. Predicted 0.085 s per (step . token) from the 512x512 measurement.
  * batch -> time per sample. The per-layer dequantise is a FIXED cost per forward
    (about 30 ms x 224 layers = 6.7 s), and small M leaves the GEMM well below the
    machine's ceiling. Both amortise across a batch, so the per-sample price should fall
    faster than 1/batch.

Runs output_type="latent", so no VAE is loaded; the teacher's product for latent-space
distillation is the final latent anyway.

Usage:
    python qwen21_teacher_cost.py --steps 2
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
    ap.add_argument("--embeds", default=os.path.join(HERE, "qwen21_embeds_free.pt"))
    ap.add_argument("--prompt", default="a beautiful woman, photorealistic, masterpiece, "
                                        "best quality, highly detailed")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--sizes", default="128,256,512")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

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
    embeds, mask = rec["embeds"], rec.get("mask")

    sizes = [int(s) for s in a.sizes.split(",")]
    batches = [int(b) for b in a.batches.split(",")]

    print(f"steps per cell {a.steps}   prompt: {a.prompt[:60]}")
    print(f"{'size':>6} {'tokens':>7} {'batch':>6} {'s/step':>8} {'s/sample':>9} "
          f"{'GFLOPS':>8} {'12 steps/img':>13} {'1000 imgs':>10}")

    for size in sizes:
        tokens = (size // 16) ** 2
        for batch in batches:
            kw = dict(true_cfg_scale=1.0, height=size, width=size,
                      num_inference_steps=a.steps, output_type="latent",
                      num_images_per_prompt=batch,
                      generator=torch.Generator().manual_seed(1234),
                      prompt_embeds=embeds, prompt_embeds_mask=mask)
            pipe(**{**kw, "num_inference_steps": 1})            # warm this shape
            t0 = time.perf_counter()
            pipe(**kw)
            dt = time.perf_counter() - t0
            per_step = dt / a.steps
            per_sample = per_step / batch
            rows = batch * tokens                      # rows through the MLP per step
            flops = 2.0 * 7.115e9 * rows
            gflops = flops / per_step / 1e9
            print(f"{size:>6} {tokens:>7} {batch:>6} {per_step:>8.1f} "
                  f"{per_sample:>9.1f} {gflops:>8.1f} {per_sample*12/60:>12.1f}m "
                  f"{per_sample*12*1000/3600:>9.1f}h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
