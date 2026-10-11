"""Where does a Qwen-Image-2.1 step actually spend its time?

The i5-10400 has twice the threads of the R5-4500U and renders a 1024x1024 step in 225 s
against 328 s -- 1.46x, not the 2x the core count suggests. Something in the step does
not scale with threads, and guessing which part it is would be wasteful: the two
candidates pull in opposite directions.

  * if it is the dense GEMMs, the fix is arithmetic (fewer steps, smaller MLP)
  * if it is the dequantise step, the fix is a different one entirely -- the 4-bit path
    unpacks every weight to fp32 before each matmul, so each forward moves 3.42 GB of
    packed weights and writes 6.9 GB of fp32, and that traffic does not shrink when
    more cores are added

There is also a specific hazard this fork has already been bitten by: the CPU kernels
run their own OpenMP region, so `torch.set_num_threads()` does not govern them, and the
report records a case where using every core was SLOWER than using four. A probe that
only varied torch's thread count would have missed it.

Runs the pipeline with output_type="latent" so the VAE is never needed and the numbers
are the transformer's alone.

Usage:
    python qwen21_probe.py --mode profile --steps 3
    python qwen21_probe.py --mode time --steps 2
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


def build(a):
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
    return pipe, rec["embeds"], rec.get("mask")


def run(a, pipe, embeds, mask, steps, tag=""):
    kw = dict(true_cfg_scale=a.cfg, height=a.height, width=a.width,
              num_inference_steps=steps, output_type="latent",
              generator=torch.Generator().manual_seed(a.seed),
              prompt_embeds=embeds)
    if mask is not None:
        kw["prompt_embeds_mask"] = mask
    t0 = time.perf_counter()
    pipe(**kw)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--embeds", default=os.path.join(HERE, "test_embeds.pt"))
    ap.add_argument("--prompt", default="a red panda astronaut floating in space, "
                                        "photorealistic, cinematic lighting, earth in "
                                        "the background, detailed fur")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mode", choices=("profile", "time"), default="profile")
    a = ap.parse_args()

    import bitsandbytes  # noqa: F401  (report what the library picked up)
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"host {os.environ.get('COMPUTERNAME','?')}  "
          f"torch threads {torch.get_num_threads()}  "
          f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS','<unset>')}")
    print(f"size {a.width}x{a.height}  steps {a.steps}  mode {a.mode}")
    print(flush=True)

    pipe, embeds, mask = build(a)
    ws, peak = L.rss_gb()
    print(f"loaded: resident {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    if a.mode == "time":
        run(a, pipe, embeds, mask, 1, "warm")          # warm oneDNN kernels
        dt = run(a, pipe, embeds, mask, a.steps)
        print(f"\n{a.steps} steps in {dt:.1f} s -> {dt/a.steps:.1f} s/step")
        return 0

    from torch.profiler import ProfilerActivity, profile

    run(a, pipe, embeds, mask, 1, "warm")
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        dt = run(a, pipe, embeds, mask, a.steps)
    print(f"\nprofiled {a.steps} steps in {dt:.1f} s ({dt/a.steps:.1f} s/step)\n")
    print("=== top ops by self CPU time ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))

    print("=== time by operator category ===")
    cats = {}
    for ev in prof.key_averages():
        name = ev.key
        if "dequantize" in name or "quantize" in name:
            cat = "quantize/dequantize"
        elif "addmm" in name or "mm" == name or "linear" in name.lower() or "gemm" in name:
            cat = "gemm"
        elif "scaled_dot_product" in name or "attention" in name.lower():
            cat = "attention"
        elif "layer_norm" in name or "rms" in name.lower() or "norm" in name.lower():
            cat = "norm"
        elif "copy" in name or "cat" in name or "contiguous" in name:
            cat = "memory movement"
        elif "mul" in name or "add" in name or "silu" in name or "gelu" in name:
            cat = "elementwise"
        else:
            cat = "other"
        cats[cat] = cats.get(cat, 0.0) + ev.self_cpu_time_total / 1e6
    total = sum(cats.values()) or 1.0
    for cat, ms in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<20} {ms/1000:8.2f} s  {100*ms/total:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
