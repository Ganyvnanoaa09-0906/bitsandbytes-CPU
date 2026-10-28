"""Per-module timing for Wan2.1 3D VAE decoder (inference)."""
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

import collections
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import wan_vae_decoder_map as M
import wan_vae_profile as P

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def main():
    model = P.load_decoder()
    dec = model.decoder
    counts = collections.Counter()
    leaf = []
    for name, m in dec.named_modules():
        counts[type(m).__name__] += 1
        if len(list(m.children())) == 0:
            leaf.append((name, m))
    print('module types:', dict(counts), flush=True)
    print('leaf modules:', len(leaf), flush=True)

    T = collections.defaultdict(float)
    C = collections.defaultdict(int)
    SHAPES = collections.defaultdict(list)

    def hook(name, m):
        def pre(mod, inp):
            m._t0 = time.perf_counter()
        def post(mod, inp, out):
            dt = time.perf_counter() - m._t0
            key = type(mod).__name__
            T[key] += dt; C[key] += 1
            if len(SHAPES[key]) < 8:
                try:
                    shp = tuple(inp[0].shape)
                except Exception:
                    shp = ()
                SHAPES[key].append(shp)
        return m.register_forward_pre_hook(pre), m.register_forward_hook(post)

    hs = []
    for name, m in leaf:
        hs.extend(hook(name, m))
    latent = torch.randn(1, 16, 5, 8, 8)
    with torch.no_grad():
        model.decode(latent, return_dict=False)
    T.clear(); C.clear(); SHAPES.clear()
    with torch.no_grad():
        t0 = time.perf_counter(); model.decode(latent, return_dict=False); total = time.perf_counter() - t0
    print('total decode %.1f ms' % (total * 1e3), flush=True)
    for k in sorted(T, key=lambda x: -T[x]):
        print('%-22s %8.1f ms  %6.1f%%  calls=%d  examples=%s'
              % (k, T[k] * 1e3, 100 * T[k] / total, C[k], SHAPES[k][:3]), flush=True)


if __name__ == '__main__':
    main()
