"""probe_qwen_vae_conv.py — is a Qwen VAE T=1 OpenCL port worth it?

Captures the real conv shapes from a CPU VAE decode, then benchmarks the largest convs:
CPU torch conv2d vs OpenCLFP16.conv2d_h (im2col + CLBlast Hgemm, fp16).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models import AutoencoderKLQwenImage21

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

FULL = r'D:\work\qwen_env\Qwen-Image-2.1'


def main():
    torch.set_num_threads(6)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval()
    rt = OpenCLFP16.get()
    shapes = {}
    hooks = []

    def mk(name):
        def h(module, inp, out):
            if name not in shapes:
                s = tuple(inp[0].shape)
                if len(s) == 5:
                    s = (s[0], s[1], s[3], s[4])
                shapes[name] = (s, tuple(module.weight.shape),
                                tuple(module.stride), tuple(module.padding))
        return h

    for name, module in vae.named_modules():
        if name.startswith('decoder') and module.__class__.__name__ == 'QwenImage21CausalConv3d':
            hooks.append(module.register_forward_hook(mk(name)))
    z = torch.randn(1, 64, 1, 16, 16)
    with torch.no_grad():
        vae.decode(z, return_dict=False)
    for h in hooks:
        h.remove()
    print('captured', len(shapes), 'conv shapes; largest by FLOPs:', flush=True)

    rows = []
    for name, (ishape, wshape, stride, pad) in shapes.items():
        N, C, H, W = ishape
        O = wshape[0]
        flops = 2 * N * O * (H * W) * C * wshape[2] * wshape[3]
        rows.append((flops, name, ishape, wshape, stride, pad))
    rows.sort(reverse=True)

    for flops, name, (N, C, H, W), (O, OC, KH, KW), stride, pad in rows[:6]:
        module = vae.get_submodule(name)
        x = torch.randn(N, C, H, W)
        w = module.weight.detach().float()
        b = module.bias.detach().float() if module.bias is not None else None
        pad_eff = KH // 2
        with torch.no_grad():
            y_cpu = F.conv2d(x, w, b, stride=stride, padding=pad_eff)
            t0 = time.perf_counter()
            for _ in range(3):
                y_cpu = F.conv2d(x, w, b, stride=stride, padding=pad_eff)
            cpu_s = (time.perf_counter() - t0) / 3
        # GPU: pad is the same; fp16 weights/input
        try:
            xb = rt.to_gpu(np.ascontiguousarray(x.numpy().astype(np.float16)))
            wb = rt.to_gpu(np.ascontiguousarray(w.numpy().astype(np.float16)))
            shp = (N, C, H, W)
            wshp = (O, OC, KH, KW)
            rt.conv2d_h(xb, wb, shp, wshp, stride[0], pad_eff)
            rt.queue.finish()
            t0 = time.perf_counter()
            for _ in range(3):
                ob, osh = rt.conv2d_h(xb, wb, shp, wshp, stride[0], pad_eff)
            rt.queue.finish()
            gpu_s = (time.perf_counter() - t0) / 3
            print('%-42s %s->%s k%dx%d | FLOP %.2fG | CPU %.3f s | GPU %.3f s | %.2fx'
                  % (name, C, O, KH, KW, flops / 1e9, cpu_s, gpu_s, cpu_s / gpu_s),
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(name, 'GPU failed', type(e).__name__, e, flush=True)


if __name__ == '__main__':
    main()
