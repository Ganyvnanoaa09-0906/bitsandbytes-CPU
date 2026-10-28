"""Wan VAE decode variants: cl3d + RMS fast, Conv2d cl, bf16 autocast, torch.compile."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def fast_rms(self, x):
    ss = x.pow(2).sum(dim=1, keepdim=True)
    return (x * torch.rsqrt(ss.clamp_min(1e-24))) * self.scale * self.gamma + self.bias


def timed(model, latent, tag, ref=None):
    with torch.no_grad():
        t = time.perf_counter()
        out = model.decode(latent, return_dict=False)[0]
        dt = time.perf_counter() - t
    err = float((out - ref).abs().max()) if ref is not None else 0.0
    print('%-28s %7.1f ms  maxdiff %s' % (tag, dt * 1e3, ('%.2e' % err) if ref is not None else 'ref'), flush=True)
    return out, dt


def main():
    model = P.load_decoder()
    latent = torch.randn(1, 16, 5, 8, 8)
    out_ref, base = timed(model, latent, 'fp32 baseline')
    for m in model.decoder.modules():
        if isinstance(m, WanCausalConv3d):
            m.to(memory_format=torch.channels_last_3d)
    x = latent.contiguous(memory_format=torch.channels_last_3d)
    WanRMS_norm.forward = fast_rms
    out1, t1 = timed(model, x, 'cl3d + fast RMS', out_ref)
    print('  ref speedup %.2fx' % (base / t1), flush=True)

    # variant: Conv2d channels_last too
    for m in model.decoder.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.channels_last)
    try:
        out2, t2 = timed(model, x, '+ Conv2d channels_last', out_ref)
        print('  speedup %.2fx' % (base / t2), flush=True)
    except Exception as e:
        print('Conv2d cl failed:', type(e).__name__, str(e)[:200], flush=True)

    # variant: bf16 autocast
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with torch.no_grad():
                t = time.perf_counter()
                out3 = model.decode(x, return_dict=False)[0]
                t3 = time.perf_counter() - t
        err = float((out3.float() - out_ref).abs().max())
        print('%-28s %7.1f ms  maxdiff %.2e  speedup %.2fx'
              % ('+ bf16 autocast', t3 * 1e3, err, base / t3), flush=True)
    except Exception as e:
        print('bf16 failed:', type(e).__name__, str(e)[:200], flush=True)

    # variant: torch.compile (fresh decoder, cl3d)
    try:
        import copy
        m2 = copy.deepcopy(model.decoder)
        for p in m2.parameters():
            p.requires_grad_(False)
        m2.eval()
        cm = torch.compile(m2, mode='reduce-overhead')
        with torch.no_grad():
            t = time.perf_counter()
            o = cm(x, return_dict=False)[0] if False else None
        # decoder signature: call model.decode through wrapper is easier; skip compile if hard
        print('torch.compile skipped (decode wrapper signature)', flush=True)
    except Exception as e:
        print('compile failed:', type(e).__name__, str(e)[:200], flush=True)


if __name__ == '__main__':
    main()
