"""Where inside one Qwen-Image-2.1 step the time actually goes.

qwen21_probe.py answers "which aten op dominates", and that is not enough to act on:
"60% in aten::linear" does not say whether it is the image MLP, the text MLP or the
attention projections, and it never compares the achieved GFLOPS with what this machine
can actually do on these shapes. A step can be slow because it is doing a lot of
arithmetic, or because it is doing arithmetic badly, and those two have opposite fixes.

This probe adds three things:

  1. self time per module, with a stack so a parent's time excludes its children's.
     An attention block contains its own projections; attributing both double counts.
  2. every Linear call's (M, N, K) and every attention call's (B, H, T, D), so the work
     can be counted in FLOPs and divided by the wall clock -- then compared against a
     GEMM ceiling measured in the same process on the same shapes, because a ceiling
     quoted from another machine or another shape is not a denominator.
  3. the same numbers at two resolutions. Attention is O(T^2); everything else is O(T).
     Running 512 and 1024 separates the two without guessing which module is quadratic.

Usage:
    python qwen21_where.py --size 512
    python qwen21_where.py --size 1024 --steps 1
    python qwen21_where.py --size 512 --threads 12
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

# ---------------------------------------------------------------- module accounting
# Inclusive/exclusive wall-clock attribution. Every hooked module pushes on entry and
# pops on exit; the difference, minus the time its own children took inside that window,
# is its self time. Without the subtraction the transformer block and the eight Linears
# underneath it all report the block's full duration and the table sums to 10x the step.
_STACK = []
SELF = {}
CALLS = {}
SHAPES = {}


def _label(name, module):
    """A stable group name: the leaf path with its indices removed.

    transformer_blocks.17.img_mlp.net.0.proj -> img_mlp.net.0.proj
    Keeping the index would produce 56 rows of one line each; dropping it aggregates
    them, which is what the question 'which part is slow' wants.
    """
    parts = [p for p in name.split(".") if not p.isdigit()]
    joined = ".".join(parts)
    for key in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
                "attn.to_qkv", "attn.to_out"):
        if joined.endswith(key):
            return key.replace("attn.", "attn.")
    if joined.startswith("transformer_blocks."):
        joined = joined[len("transformer_blocks."):]
    if joined.startswith("single_transformer_blocks."):
        joined = joined[len("single_transformer_blocks."):]
    return joined or module.__class__.__name__


def _hookable(module):
    from torch import nn
    names = ("Linear", "Linear4bit", "Linear8bitLt", "Conv2d", "LayerNorm", "RMSNorm",
             "GroupNorm", "Embedding", "GELU", "SiLU", "Dropout", "Identity",
             "QwenImageTransformerBlock", "QwenImage21TransformerBlock",
             "QwenTimestepProjEmbeddings", "QwenEmbedRope", "QwenEmbedLayer3DRope",
             "Attention", "FeedForward", "MLP")
    return any(n in module.__class__.__name__ for n in names)


def install_hooks(model):
    handles = []

    def make(name):
        def pre(mod, args):
            _STACK.append([name, time.perf_counter(), 0.0, mod, args])

        def post(mod, args, out):
            if not _STACK:
                return
            rec = _STACK.pop()
            if rec[0] != name:                     # an exception unwound the stack
                return
            dt = time.perf_counter() - rec[1] - rec[2]
            SELF[name] = SELF.get(name, 0.0) + dt
            CALLS[name] = CALLS.get(name, 0) + 1
            if _STACK:
                _STACK[-1][2] += dt + rec[2]
            # shape bookkeeping, for Linear-like modules only. A LayerNorm created with
            # elementwise_affine=False still HAS a .weight attribute, it is just None,
            # which is how this crashed the first run at 1024.
            w = getattr(mod, "weight", None)
            inp = rec[4][0] if rec[4] else None
            is_linear = "Linear" in mod.__class__.__name__ or "Conv" in mod.__class__.__name__
            if is_linear and isinstance(inp, torch.Tensor) and torch.is_tensor(w):
                m = inp.numel() // max(1, inp.shape[-1]) if inp.dim() else 1
                k = inp.shape[-1]
                # NOT w.shape: a Params4bit holds the PACKED uint8 storage, whose first
                # dimension is numel/2 (12288*4096/2 = 25165824) and whose second is 1.
                # Using it reported 25 million output features and a FLOP count four
                # orders of magnitude out. QuantState keeps the real shape.
                qs = getattr(w, "quant_state", None)
                if qs is not None and getattr(qs, "shape", None) is not None:
                    n = int(qs.shape[0])
                else:
                    n = int(w.shape[0])
                key = (name, int(m), int(n), int(k))
                SHAPES[key] = SHAPES.get(key, 0) + 1
        return pre, post

    for name, mod in model.named_modules():
        if not name or not _hookable(mod):
            continue
        pre, post = make(_label(name, mod))
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    return handles


# ------------------------------------------------------------------ attention shape
_SDPA = torch.nn.functional.scaled_dot_product_attention
ATTN = {"flops": 0.0, "calls": 0, "shapes": {}}


def install_sdpa_hook():
    def wrapper(query, key, value, *a, **kw):
        try:
            b, h, t, d = query.shape
            t2 = key.shape[-2]
            # 2 matmuls of 2*T*T*D each per head; key/value heads may be fewer (GQA)
            ATTN["flops"] += 2.0 * 2.0 * b * h * t * t2 * d
            ATTN["calls"] += 1
            sig = (int(b), int(h), int(t), int(t2), int(d))
            ATTN["shapes"][sig] = ATTN["shapes"].get(sig, 0) + 1
        except Exception:  # noqa: BLE001
            pass
        return _SDPA(query, key, value, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = wrapper


# ------------------------------------------------------------------------ the runner
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
    return pipe, transformer, rec["embeds"], rec.get("mask")


def run(a, pipe, embeds, mask, steps):
    kw = dict(true_cfg_scale=a.cfg, height=a.height, width=a.width,
              num_inference_steps=steps, output_type="latent",
              generator=torch.Generator().manual_seed(a.seed),
              prompt_embeds=embeds)
    if mask is not None:
        kw["prompt_embeds_mask"] = mask
    t0 = time.perf_counter()
    pipe(**kw)
    return time.perf_counter() - t0


def gemm_ceiling(tokens, dim_in, dim_out, threads, reps=3):
    """What this process can do on this shape, right now. The denominator.

    A ceiling quoted from a benchmark run elsewhere is not usable: oneDNN picks
    different kernels per shape, the thermal state differs, and the packed 4-bit path
    dequantises before it multiplies, so the honest comparison is a plain fp32 GEMM at
    the same M, N, K.
    """
    x = torch.randn(tokens, dim_in)
    w = torch.randn(dim_out, dim_in)
    torch.mm(x, w.t())                       # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        torch.mm(x, w.t())
    dt = (time.perf_counter() - t0) / reps
    flops = 2.0 * tokens * dim_in * dim_out
    return dt, flops / dt / 1e9


def report(a, dt, steps, threads):
    per_step = dt / steps
    print()
    print("=" * 78)
    print(f"per-step wall clock: {per_step:.1f} s   ({steps} step(s), "
          f"{a.width}x{a.height}, {threads} torch threads)")
    print("=" * 78)

    print("\n=== self time by module group (top 22) ===")
    total = sum(SELF.values()) or 1.0
    print(f"  {'group':<34} {'calls':>7} {'self s':>9} {'%':>6}")
    for name, ms in sorted(SELF.items(), key=lambda kv: -kv[1])[:22]:
        print(f"  {name:<34} {CALLS.get(name,0):>7} {ms:>9.2f} {100*ms/per_step:>5.1f}%")
    print(f"  {'(hooked modules total)':<34} {'':>7} {total:>9.2f} "
          f"{100*total/per_step:>5.1f}%")

    print("\n=== Linear shapes (top 12 by FLOPs) ===")
    rows = []
    for (name, m, n, k), cnt in SHAPES.items():
        flops = 2.0 * m * n * k * cnt
        rows.append((flops, name, m, n, k, cnt))
    rows.sort(reverse=True)
    lin_flops = sum(r[0] for r in rows)
    print(f"  {'group':<30} {'M':>7} {'N':>6} {'K':>6} {'calls':>6} {'GFLOP':>9}")
    for flops, name, m, n, k, cnt in rows[:12]:
        print(f"  {name:<30} {m:>7} {n:>6} {k:>6} {cnt:>6} {flops/1e9:>9.1f}")
    print(f"  linear total: {lin_flops/1e9:.1f} GFLOP  ->  "
          f"{lin_flops/per_step/1e9:.1f} GFLOPS achieved")

    if ATTN["calls"]:
        print(f"\n  attention: {ATTN['calls']} SDPA calls, {ATTN['flops']/1e9:.1f} GFLOP "
              f"-> {ATTN['flops']/per_step/1e9:.1f} GFLOPS")
        for sig, cnt in sorted(ATTN["shapes"].items(), key=lambda kv: -kv[1])[:4]:
            b, h, t, t2, d = sig
            print(f"    B={b} H={h} T={t}->{t2} D={d}   x{cnt}")

    print("\n=== vs a plain fp32 GEMM at the same shape, same process ===")
    for (tokens, din, dout) in ((4096, 4096, 4096), (16384, 4096, 12288)):
        gdt, gfl = gemm_ceiling(tokens, din, dout, threads)
        print(f"  {tokens:>6} x {din} x {dout}: {gdt*1000:8.1f} ms  {gfl:6.1f} GFLOPS")

    grand = lin_flops + ATTN["flops"]
    print(f"\n  accounted FLOPs: {grand/1e9:.1f} GFLOP  ->  "
          f"{grand/per_step/1e9:.1f} GFLOPS overall")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--embeds", default=os.path.join(HERE, "test_embeds.pt"))
    ap.add_argument("--prompt", default="a red panda astronaut floating in space, "
                                        "photorealistic, cinematic lighting, earth in "
                                        "the background, detailed fur")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-hooks", action="store_true")
    a = ap.parse_args()
    a.height = a.width = a.size

    import bitsandbytes  # noqa: F401
    if a.threads:
        torch.set_num_threads(a.threads)
    threads = torch.get_num_threads()
    print(f"host {os.environ.get('COMPUTERNAME','?')}  torch threads {threads}  "
          f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS','<unset>')}")
    print(f"size {a.width}x{a.height}  steps {a.steps}", flush=True)

    pipe, transformer, embeds, mask = build(a)
    ws, peak = L.rss_gb()
    print(f"loaded: resident {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    install_sdpa_hook()
    handles = [] if a.no_hooks else install_hooks(transformer)
    print(f"hooked {len(handles)} module forwards", flush=True)

    run(a, pipe, embeds, mask, 1)                     # warm oneDNN kernels
    SELF.clear(); CALLS.clear(); SHAPES.clear()
    ATTN["flops"] = 0.0; ATTN["calls"] = 0; ATTN["shapes"].clear()
    dt = run(a, pipe, embeds, mask, a.steps)
    for h in handles:
        h.remove()
    return report(a, dt, a.steps, threads)


if __name__ == "__main__":
    sys.exit(main())
