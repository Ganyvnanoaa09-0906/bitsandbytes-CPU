"""Can this fork train a LoRA on top of a frozen 4-bit Qwen-Image-2.1, on CPU?

This is the switch the whole distillation plan hangs on. Everything else in the plan is
arithmetic we have already measured; this one is a property of the fork that has never
been exercised on the real machine. The help text advertises "4-bit base + adapters
(QLoRA)", and the reference-image path just showed what happens when an advertised path
is never actually run.

What has to be true:

  1. autograd must flow THROUGH a frozen 4-bit layer to its input. The CPU 4-bit GEMM ends
     in `nn.functional.linear(A, dequantize_4bit(B))`: that is differentiable in A no
     matter where B came from -- unless the call is wrapped in no_grad or registered as a
     custom op without a backward, either of which would silently cut the graph at every
     layer and leave the adapters with no gradient at all.
  2. the frozen weights themselves must stay grad-free (backward into 4-bit codes is not
     defined here and must never be attempted).
  3. it has to fit. Activations for backward at 128x128 are 1/3 of the 256x256 ones, which
     is the second reason to start small.

A silent cut is the failure mode that matters: it would not raise, it would just train the
adapters against zeros. So the check is not "did backward run" but "do the adapter
gradients have a non-zero norm".

Usage:
    python qwen21_train_probe.py --size 128
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import qwen21_load as L  # noqa: E402


class LoRA(nn.Module):
    """The smallest honest adapter: base(x) + B(A(x)), B zero so the start is a no-op."""

    def __init__(self, base, r=16):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def attach_lora(model, r):
    import bitsandbytes as bnb
    wrapped = 0
    trainable = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                lora = LoRA(child, r=r)
                setattr(module, child_name, lora)
                wrapped += 1
                trainable += sum(p.numel() for p in lora.parameters()
                                 if p.requires_grad)
    return wrapped, trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--embeds", default=os.path.join(HERE, "qwen21_embeds_free.pt"))
    ap.add_argument("--prompt", default="a beautiful woman, photorealistic, masterpiece, "
                                        "best quality, highly detailed")
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--src", default=r"D:\work\qwen_env\Qwen-Image-2.1")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_grad_enabled(True)

    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from transformers import Qwen3VLProcessor

    transformer = L.load_quantized(a.ckpt, "transformer")
    wrapped, trainable = attach_lora(transformer, a.rank)
    frozen = sum(p.numel() for p in transformer.parameters() if not p.requires_grad)
    print(f"[probe] wrapped {wrapped} Linear4bit layers with rank-{a.rank} LoRA", flush=True)
    print(f"[probe] trainable {trainable/1e6:.2f} M   frozen {frozen/1e9:.3f} B",
          flush=True)

    processor = Qwen3VLProcessor.from_pretrained(os.path.join(a.src, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(a.src, "scheduler"))
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=None, text_encoder=None,
                               processor=processor, transformer=transformer)
    rec = torch.load(a.embeds, weights_only=False)["prompts"][a.prompt]

    ws, peak = L.rss_gb()
    print(f"[probe] resident before forward {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    t0 = time.perf_counter()
    # `QwenImage21Pipeline.__call__` is decorated with @torch.no_grad() -- standard for
    # inference pipelines, and a silent graph cutter for training: the call succeeds, the
    # latent comes back, and the adapters get zero gradient with no error anywhere.
    # __wrapped__ is the same function without that one decorator.
    call = getattr(pipe.__call__, "__wrapped__", pipe.__call__)
    if call is pipe.__call__:
        print("[probe] warning: no __wrapped__ on __call__; if grads come back zero, the "
              "pipeline's no_grad decorator is the first thing to suspect", flush=True)
    out = call(pipe, true_cfg_scale=1.0, height=a.size, width=a.size,
               num_inference_steps=a.steps, output_type="latent",
               generator=torch.Generator().manual_seed(1234),
               prompt_embeds=rec["embeds"], prompt_embeds_mask=rec.get("mask"))
    t_fwd = time.perf_counter() - t0
    lat = out.images if torch.is_tensor(out.images) else out.images
    ws, peak = L.rss_gb()
    print(f"[probe] forward {a.steps} steps {t_fwd:.1f} s "
          f"({t_fwd/a.steps:.1f} s/step)  latent {tuple(lat.shape)} "
          f"grad_fn={type(lat.grad_fn).__name__ if lat.grad_fn else None}", flush=True)
    print(f"[probe] resident after forward {ws:.2f} GB (peak {peak:.2f} GB)", flush=True)

    if lat.grad_fn is None:
        print("[probe] FAIL: the latent carries no grad_fn, so nothing in the transformer "
              "was tracked. Training is impossible on this path.", flush=True)
        return 1

    target = torch.randn_like(lat)
    loss = nn.functional.mse_loss(lat.float(), target)
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    ws, peak = L.rss_gb()
    print(f"[probe] backward {t_bwd:.1f} s; resident {ws:.2f} GB "
          f"(peak {peak:.2f} GB)", flush=True)

    grad_norm = 0.0
    with_grad = 0
    without_grad = 0
    for name, p in transformer.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            without_grad += 1
            continue
        with_grad += 1
        grad_norm += float(p.grad.pow(2).sum())
    grad_norm = grad_norm ** 0.5
    print(f"[probe] adapter params with grad: {with_grad}   without grad: "
          f"{without_grad}", flush=True)
    print(f"[probe] adapter grad norm: {grad_norm:.6e}", flush=True)
    if grad_norm == 0.0:
        print("[probe] FAIL: backward ran but every adapter gradient is zero -- the graph "
              "is cut somewhere between the loss and the adapters.", flush=True)
        return 1
    print(f"[probe] PASS: one training iteration = fwd {t_fwd/a.steps:.1f} + "
          f"bwd {t_bwd:.1f} s = {(t_fwd/a.steps + t_bwd):.1f} s/iter at {a.size}px",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
