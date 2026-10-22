"""Linear-only fp32 hybrid wrapper for train_video_mvp.py (AnimateDiff LoRA).

Monkeypatches F.linear: large 2D/3D matmuls go to iGPU CLBlast Sgemm with a custom
autograd Function (dX, dW); small/LoRA stay on CPU. Conv/attention/norm remain CPU.

Usage: copy the usual train_video_mvp args after the script name.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\video_mvp')
import gt1030.hybrid_gpu as G

_orig_linear = F.linear
STATS = {'gpu_calls': 0, 'cpu_calls': 0, 'gpu_ms': 0.0}
MIN_M = 1024
MIN_K = 128


class GpuLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        xf = x.reshape(-1, x.shape[-1])
        y = G.sgemm(xf.detach().numpy().astype(np.float32), w.detach().numpy().astype(np.float32).T)
        out = torch.from_numpy(y)
        if b is not None:
            out = out + b
        ctx.save_for_backward(xf, w)
        ctx.shape = x.shape
        return out.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, dy):
        xf, w = ctx.saved_tensors
        df = dy.reshape(-1, dy.shape[-1]).detach().numpy().astype(np.float32)
        dx = G.sgemm(df, w.detach().numpy().astype(np.float32))
        dw = G.sgemm(df.T, xf.detach().numpy().astype(np.float32))
        return torch.from_numpy(dx).reshape(ctx.shape), torch.from_numpy(dw), None


def gpu_linear(x, w, b=None):
    if x.dim() >= 2 and w.dim() == 2 and x.shape[-1] >= MIN_K and x.numel() // x.shape[-1] >= MIN_M:
        STATS['gpu_calls'] += 1
        return GpuLinearFn.apply(x, w, b)
    STATS['cpu_calls'] += 1
    return _orig_linear(x, w, b)


F.linear = gpu_linear
# transformer internals sometimes reference torch._C._nn.linear directly
try:
    torch._C._nn.linear = gpu_linear
except Exception:
    pass

import train_video_mvp as tv  # noqa: E402

if __name__ == '__main__':
    tv.main()
    print('[hybrid-linear] gpu_calls=%d cpu_calls=%d' % (STATS['gpu_calls'], STATS['cpu_calls']), flush=True)
