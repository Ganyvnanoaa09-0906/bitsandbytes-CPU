"""Profile Wan2.1 3D VAE decoder (inference) by ATen op groups.

Loads decoder-only weights (mapping from wan_vae_decoder_map.py), runs one decode
(latent (1,16,5,8,8) -> video (1,3,17,64,64)) under torch profiler, prints groups.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import wan_vae_decoder_map as M

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def load_decoder():
    from diffusers import AutoencoderKLWan
    sd = torch.load(M.VAE_PTH, map_location='cpu', weights_only=True)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    new = {}
    for k, v in sd.items():
        nk = M.map_key(k)
        if nk is not None:
            new[nk] = v
    model = AutoencoderKLWan()
    ref = {k: v for k, v in model.state_dict().items() if k.startswith('decoder.')}
    assert all(k in ref and tuple(ref[k].shape) == tuple(new[k].shape) for k in new), 'key/shape mismatch'
    model.load_state_dict(new, strict=False)
    model.eval()
    return model


def main():
    model = load_decoder()
    latent = torch.randn(1, 16, 5, 8, 8)
    with torch.no_grad():
        t0 = time.perf_counter(); out = model.decode(latent, return_dict=False)[0]; dt = time.perf_counter() - t0
    print('decode once: %.1f ms  out=%s  finite=%s' % (dt * 1e3, tuple(out.shape), bool(torch.isfinite(out).all())), flush=True)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():
            t0 = time.perf_counter(); out = model.decode(latent, return_dict=False)[0]; dt = time.perf_counter() - t0
    print('profiled decode: %.1f ms' % (dt * 1e3), flush=True)
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35))


if __name__ == '__main__':
    main()
