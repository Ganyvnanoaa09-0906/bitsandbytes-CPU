"""How much of a Qwen-Image-2.1 trajectory could be reused instead of recomputed?

Distillation is not available for this model (no LCM/Lightning weights for
Qwen-Image-2.1, and its architecture differs from Qwen-Image-1), so the training-free
alternative is step-level caching as used on FLUX and Wan: adjacent denoising steps often
produce nearly the same transformer output, and TeaCache decides whether to reuse the
previous one from the change in the timestep modulation vector.

Whether that holds here is a measurement. This probe records, once per denoising step:

  * the modulation vector -- the signal a cache would threshold on
  * the model output the scheduler actually consumes, and its relative change against the
    previous step, which is the error a cache would inject by skipping the recomputation

WHY THE SCHEDULER IS HOOKED RATHER THAN THE TRANSFORMER
The pipeline runs the first step as a prefill (image and text tokens together) and the
rest as decode against a cached text KV (`use_kv_cache=True`), so the first transformer
output has 30 more rows than the others. Comparing transformer outputs across that
boundary compares two different objects; the scheduler sees one consistent tensor per
step, which is also exactly what a cache would reuse.

WHAT THIS DOES NOT MEASURE
The perturbation of the noise prediction, not of the final image. The sampler is chaotic
(§10.166), so a small perturbation here can still move the picture; the last word has to
be an A/B of rendered images.

Usage:
    python qwen21_step_cache_probe.py --size 384 --steps 20
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

STATE = {"step": 0, "timestep": None, "mod": None}
ROWS = []


def install(pipe, transformer):
    handles = []

    # --- modulation vector: produced per step by a small module inside the transformer
    def keep_vec(mod, args, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            STATE["mod"] = t.detach().clone()

    for name, mod in transformer.named_modules():
        if name.endswith("modulation") or name.endswith("time_text_embed"):
            handles.append(mod.register_forward_hook(keep_vec))

    def pre(mod, args, kwargs):
        t = kwargs.get("timestep", args[0] if args else None)
        STATE["timestep"] = t

    handles.append(transformer.register_forward_pre_hook(pre, with_kwargs=True))

    # --- what the sampler is handed, which is what a cache would substitute
    sched = pipe.scheduler
    original_step = sched.step

    def step_wrapper(model_output, timestep, sample, *a, **kw):
        with torch.no_grad():
            mo = model_output.detach().float()
            prev = STATE.get("prev_mo")
            mod = STATE.get("mod")
            row = {"step": STATE["step"], "t": float(timestep.flatten()[0]),
                   "mo_shape": tuple(mo.shape), "mod_rel": None, "mo_rel": None}
            if prev is not None and prev.shape == mo.shape:
                row["mo_rel"] = ((mo - prev).norm() / prev.norm()).item()
            if mod is not None and STATE.get("prev_mod") is not None:
                a_, b_ = mod.float(), STATE["prev_mod"].float()
                if a_.shape == b_.shape:
                    row["mod_rel"] = ((a_ - b_).norm() / b_.norm()).item()
            ROWS.append(row)
            STATE["prev_mo"] = mo.clone()
            STATE["prev_mod"] = None if mod is None else mod.clone()
            STATE["step"] += 1
            STATE["mod"] = None
        return original_step(model_output, timestep, sample, *a, **kw)

    sched.step = step_wrapper
    return handles


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
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="output change below which a step counts as reusable")
    a = ap.parse_args()

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"size {a.size}  steps {a.steps}  threads {torch.get_num_threads()}", flush=True)

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

    handles = install(pipe, transformer)
    print(f"hooked {len(handles)} transformer module(s) + scheduler.step", flush=True)
    t0 = time.perf_counter()
    pipe(true_cfg_scale=a.cfg, height=a.size, width=a.size,
         num_inference_steps=a.steps, output_type="latent",
         generator=torch.Generator().manual_seed(a.seed),
         prompt_embeds=rec["embeds"], prompt_embeds_mask=rec.get("mask"))
    dt = time.perf_counter() - t0
    for h in handles:
        h.remove()
    pipe.scheduler.step = None  # the bound original is gone with the object

    print(f"\n{a.steps} steps in {dt:.1f} s -> {dt/a.steps:.1f} s/step\n")
    print("=== what changes between adjacent steps ===")
    print(f"  {'step':>4} {'t':>8} {'noise shape':>14} {'mod rel':>9} {'noise rel':>10}"
          f"   {'reusable':>9}")
    reusable = 0
    compared = 0
    for r in ROWS:
        rel = r["mo_rel"]
        mod = r["mod_rel"]
        if rel is not None:
            compared += 1
            ok = rel < a.threshold
            reusable += 1 if ok else 0
        else:
            ok = None
        print(f"  {r['step']:>4} {r['t']:>8.4f} {str(r['mo_shape']):>14} "
              f"{(f'{mod:>9.4f}' if mod is not None else '        -')} "
              f"{(f'{rel:>10.4f}' if rel is not None else '         -')}   "
              f"{('yes' if ok else 'no') if ok is not None else '(prefill)':>9}")
    if compared:
        rels = sorted(r["mo_rel"] for r in ROWS if r["mo_rel"] is not None)
        print(f"\n  steps compared            : {compared} of {len(ROWS)}")
        print(f"  noise change  median      : {rels[len(rels)//2]:.4f}")
        print(f"                max         : {rels[-1]:.4f}")
        print(f"  below {a.threshold:.0%} change     : {reusable}/{compared}"
              f"  -> upper bound on saving: {100*reusable/compared:.0f}% of step time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
