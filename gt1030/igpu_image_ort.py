"""igpu_image_ort.py — Stable-Diffusion inference with an ONNX Runtime DirectML session.

The OpenCL kit is GEMM-oriented (nn.Linear/GEMV/fused layers); diffusion UNets are
conv2d-dominated, which the OpenCL path does not implement yet.  This script is the
pragmatic iGPU image-inference path for now: export the components to ONNX once
(optimum-cli export onnx --task stable-diffusion) and run the ONNX Runtime
DmlExecutionProvider session on the iGPU.

Known limit of onnxruntime-directml 1.24.4 on the Vega 6 iGPU: 256x256 works for many
steps (~1.6 s/step, ~1.9x the CPU diffusers pipeline), 384x384 only completes the first
step, and 512x512 fails in the UNet (activation memory), while the same sessions run on
CPUExecutionProvider at 512.  The long-term OpenCL path needs an im2col+dGEMM conv
engine plus GroupNorm/SiLU/upsample kernels before it can compete.

Usage:
  python gt1030/igpu_image_ort.py --onnx-dir D:\\work\\sd15_onnx --steps 8 --size 256
  python gt1030/igpu_image_ort.py --onnx-dir D:\\work\\sd15_onnx --provider cpu
"""
from __future__ import annotations

import argparse
import json
import os
import time

import psutil
import torch


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx-dir', required=True)
    ap.add_argument('--provider', choices=['dml', 'cpu'], default='dml')
    ap.add_argument('--prompt', default='a photo of a cat sitting on a chair')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--no-mem-pattern', action='store_true')
    ap.add_argument('--no-arena', action='store_true')
    args = ap.parse_args()

    from optimum.onnxruntime import ORTStableDiffusionPipeline

    provider = 'DmlExecutionProvider' if args.provider == 'dml' else 'CPUExecutionProvider'
    log('loading %s (provider=%s)' % (args.onnx_dir, provider))
    t0 = time.perf_counter()
    import onnxruntime as ort
    so = ort.SessionOptions()
    if args.no_mem_pattern:
        so.enable_mem_pattern = False
    po = None
    if args.no_arena:
        so.enable_cpu_mem_arena = False
        po = [{'enable_memory_arena': False}]
    pipe = ORTStableDiffusionPipeline.from_pretrained(args.onnx_dir, provider=provider,
                                                      session_options=so,
                                                      provider_options=po,
                                                      local_files_only=True)
    load_s = time.perf_counter() - t0
    log('loaded in %.1f s | RSS %.2f GB' % (load_s, rss_gb()))

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    try:
        image = pipe(prompt=args.prompt, num_inference_steps=args.steps, height=args.size,
                     width=args.size, guidance_scale=7.5).images[0]
    except Exception as exc:  # noqa: BLE001
        log('INFERENCE FAILED: %s: %s' % (type(exc).__name__, str(exc)[:200]))
        log('NOTE: on this iGPU the DML EP is reliable at 256; 384 only 1 step; 512 fails '
            '(UNet activation memory). Try --provider cpu for 512.')
        raise SystemExit(2)
    run_s = time.perf_counter() - t0
    rss = rss_gb()
    model_name = os.path.basename(os.path.normpath(args.onnx_dir))
    out = args.out or os.path.join(r'D:\work\research\bench_results',
                                   'igpu_%s_%s_%dsteps_%d.png'
                                   % (model_name, args.provider, args.steps, args.size))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    image.save(out)
    log('backend=%s | %.2f s for %d steps (%.2f s/step) | RSS %.2f GB'
        % (args.provider, run_s, args.steps, run_s / args.steps, rss))
    log('image: %dx%d | saved %s' % (image.size[0], image.size[1], out))
    log('RESULT ' + json.dumps({
        'onnx_dir': args.onnx_dir, 'provider': args.provider, 'steps': args.steps,
        'size': args.size, 'seed': args.seed, 'load_s': load_s, 'run_s': run_s,
        's_per_step': run_s / args.steps, 'rss_gb': rss, 'image': out,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
