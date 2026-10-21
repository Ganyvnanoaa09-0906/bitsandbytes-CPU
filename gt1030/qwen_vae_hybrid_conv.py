"""qwen_vae_hybrid_conv.py — per-conv CPU/iGPU hybrid for the Qwen VAE decoder (T=1).

ROI probe showed large-channel convs are 1.0-1.53x faster on the iGPU, while small ones are
0.5-0.6x. This wraps selected large convs with an OpenCL im2col+Hgemm path and leaves the
rest on CPU, then benchmarks/validates a full 256^2 T=1 decode.
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


class HybridCausalConv3d(torch.nn.Module):
    def __init__(self, orig, rt):
        super().__init__()
        self.orig = orig
        self.rt = rt
        self.weight = orig.weight.detach()
        self.bias = orig.bias.detach() if orig.bias is not None else None
        self.stride = orig.stride[0]
        self.pad = 0  # padding is applied explicitly by F.pad above
        self._padding = tuple(orig._padding)
        self.wbuf = rt.to_gpu(np.ascontiguousarray(
            self.weight.numpy().astype(np.float16)))
        self.bbuf = (rt.to_gpu(np.ascontiguousarray(
            self.bias.numpy().astype(np.float16))) if self.bias is not None else None)

    def forward(self, x, cache_x=None):
        # x: (B, C, 1, H, W), same contract as QwenImage21CausalConv3d
        # image-specialized VAE: cache_x is unused/None for T=1
        squeeze = x.squeeze(2)
        xp = F.pad(squeeze, self._padding)
        xh = np.ascontiguousarray(xp.detach().numpy().astype(np.float16))
        xb = self.rt.to_gpu(xh)
        O, C, KH, KW = self.wbuf_shape = tuple(self.weight.shape)
        ob, osh = self.rt.conv2d_h(xb, self.wbuf, tuple(xh.shape), (O, C, KH, KW),
                                   self.stride, self.pad)
        out = np.empty(osh, dtype=np.float16)
        import pyopencl as cl
        cl.enqueue_copy(self.rt.queue, out, ob, is_blocking=True)
        self.rt.queue.finish()
        xb.release(); ob.release()
        y = torch.from_numpy(out.astype(np.float32))
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y.unsqueeze(2)


def select_large_convs(vae, k=3):
    cand = []
    for name, m in vae.named_modules():
        if name.startswith('decoder') and m.__class__.__name__ == 'QwenImage21CausalConv3d':
            flops = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
            cand.append((flops, name, m))
    cand.sort(reverse=True)
    return [(name, m) for _, name, m in cand[:k]]


def main():
    torch.set_num_threads(6)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval()
    rt = OpenCLFP16.get()
    z = torch.randn(1, 64, 1, 16, 16)
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = vae.decode(z, return_dict=False)[0]
        cpu_s = time.perf_counter() - t0
    print('CPU decode %.2f s | out %s' % (cpu_s, tuple(ref.shape)), flush=True)

    for k in (9, 12, 18):
        vae2 = AutoencoderKLQwenImage21.from_pretrained(
            FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval()
        selected = select_large_convs(vae2, k=k)
        parent = vae2
        for name, mod in selected:
            parts = name.split('.')
            par = vae2
            for p in parts[:-1]:
                par = getattr(par, p)
            setattr(par, parts[-1], HybridCausalConv3d(mod, rt))
        with torch.no_grad():
            t0 = time.perf_counter()
            out = vae2.decode(z, return_dict=False)[0]
            hy_s = time.perf_counter() - t0
        rel = float((out - ref).norm() / ref.norm())
        print('hybrid k=%d | %.2f s | speedup %.2fx | rel %.2e'
              % (k, hy_s, cpu_s / hy_s, rel), flush=True)
        del vae2


if __name__ == '__main__':
    main()
