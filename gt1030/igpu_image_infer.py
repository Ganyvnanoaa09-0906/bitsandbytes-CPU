"""igpu_image_infer.py — run a small Stable-Diffusion model on the CPU and on the iGPU (DirectML).

This is the first step of the "image/video inference on the iGPU" backlog item: diffusion
UNets are conv-heavy, so the text-model nn.Linear offloader cannot be reused.  Here we use
torch-directml (isolated venv) to run the whole pipeline on the integrated GPU and compare
with the CPU baseline: latency, process RSS, output sanity.

Usage:
  python gt1030/igpu_image_infer.py --model E:\\models\\models\\bk-sdm-tiny --backend cpu
  python gt1030/igpu_image_infer.py --model ... --backend dml --steps 4
"""
from __future__ import annotations

import argparse
import json
import traceback
import os
import time

import psutil
import torch
from PIL import Image


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--backend', choices=['cpu', 'dml'], default='cpu')
    ap.add_argument('--prompt', default='a photo of a cat sitting on a chair')
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline

    log('loading %s (variant=fp16, dtype=float32)' % args.model)
    t0 = time.perf_counter()
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    common = dict(torch_dtype=dtype, safety_checker=None,
                  requires_safety_checker=False, local_files_only=True)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, variant='fp16', **common)
    except ValueError:
        # model ships full-precision weights only (e.g. the local SD1.5 pipeline)
        pipe = StableDiffusionPipeline.from_pretrained(args.model, **common)
    pipe.set_progress_bar_config(disable=True)
    load_s = time.perf_counter() - t0
    log('loaded in %.1f s | RSS %.2f GB | components: %s'
        % (load_s, rss_gb(), type(pipe.unet).__name__))

    device = 'cpu'
    if args.backend == 'dml':
        import torch_directml
        assert torch_directml.is_available(), 'DirectML not available'
        device = torch_directml.device()
        log('moving pipeline to %s' % device)
    pipe = pipe.to(device)
    if args.backend == 'dml':
        # torch-directml does not implement F.scaled_dot_product_attention; use the
        # matmul/softmax attention processor instead. It also rejects reshape() on a
        # permuted (non-contiguous) tensor, so make those helpers contiguous.
        from diffusers.models.attention_processor import AttnProcessor, Attention

        def _bthd(self, tensor):
            head = self.heads
            b, s, d = tensor.shape
            t = tensor.reshape(b // head, head, s, d)
            return t.permute(0, 2, 1, 3).contiguous().reshape(b // head, s, d * head)

        def _hthd(self, tensor, out_dim=3):
            head = self.heads
            if tensor.ndim == 3:
                b, s, d = tensor.shape
                extra = 1
            else:
                b, extra, s, d = tensor.shape
            t = tensor.reshape(b, s * extra, head, d // head).permute(0, 2, 1, 3).contiguous()
            if out_dim == 3:
                t = t.reshape(b * head, s * extra, d // head)
            return t

        Attention.batch_to_head_dim = _bthd
        Attention.head_to_batch_dim = _hthd

        # DirectML refuses chunk(); split() + gelu is equivalent and works.
        from diffusers.models.activations import GEGLU, SwiGLU
        import torch.nn.functional as _F

        def _geglu(self, hidden_states, *args, **kwargs):
            hs = self.proj(hidden_states)
            a, b = hs.split(hs.shape[-1] // 2, dim=-1)
            return a * _F.gelu(b)

        def _swiglu(self, hidden_states, *args, **kwargs):
            hs = self.proj(hidden_states)
            a, b = hs.split(hs.shape[-1] // 2, dim=-1)
            return a * _F.silu(b)

        GEGLU.forward = _geglu
        SwiGLU.forward = _swiglu
        pipe.unet.set_attn_processor(AttnProcessor())
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    log('pipeline on %s | RSS %.2f GB' % (device, rss_gb()))

    gen = torch.Generator(device=device).manual_seed(args.seed) if device != 'cpu' else \
        torch.Generator(device='cpu').manual_seed(args.seed)
    t0 = time.perf_counter()
    try:
        image = pipe(prompt=args.prompt, num_inference_steps=args.steps,
                     height=args.size, width=args.size, generator=gen).images[0]
    except Exception as exc:  # noqa: BLE001
        log('INFERENCE FAILED: %s: %s' % (type(exc).__name__, str(exc)[:300]))
        traceback.print_exc()
        raise SystemExit(2)
    run_s = time.perf_counter() - t0
    rss = rss_gb()
    stats = (float(torch.tensor(list(image.getdata())).float().mean()),
             float(torch.tensor(list(image.getdata())).float().std()))
    log('backend=%s | %.1f s for %d steps (%.1f ms/step) | RSS %.2f GB'
        % (args.backend, run_s, args.steps, run_s * 1000 / args.steps, rss))
    log('image: %dx%d | mean %.1f | std %.1f'
        % (image.size[0], image.size[1], stats[0], stats[1]))
    model_name = os.path.basename(os.path.normpath(args.model))
    out = args.out or os.path.join(r'D:\work\research\bench_results',
                                   'igpu_%s_%s_%s_%dsteps.png'
                                   % (model_name, args.backend, args.dtype, args.steps))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    image.save(out)
    log('saved %s' % out)
    result = {
        'model': args.model, 'backend': args.backend, 'device': str(device),
        'steps': args.steps, 'size': args.size, 'seed': args.seed,
        'load_s': load_s, 'run_s': run_s, 'ms_per_step': run_s * 1000 / args.steps,
        'rss_gb': rss, 'image': out,
        'mean': stats[0], 'std': stats[1], 'dtype': args.dtype,
    }
    log('RESULT ' + json.dumps(result))
    if args.out:
        with open(args.out + '.json', 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
