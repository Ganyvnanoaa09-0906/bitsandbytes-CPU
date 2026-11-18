# -*- coding: utf-8 -*-
"""igpu_attn_probe.py — run one FullAttn self-attention block at a real UNet shape.

    python gt1030/igpu_attn_probe.py --n 2 --c 320 --h 64 --w 64

Measures device arena delta and forward time; compares against torch SDPA.
The current FullAttn is single-head and square (self-attention); cross-attention
(K=77 text tokens) is not covered.
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import torch
import torch.nn.functional as F

import gt1030.igpu_ctx as CTX
from gt1030.igpu_attn_full import FullAttn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--c", type=int, default=320)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--arena", type=int, default=1400)
    a = ap.parse_args()
    T = a.h * a.w
    print("  N=%d C=%d T=%d rows=%d" % (a.n, a.c, T, a.n * T))

    dev = CTX.DevCtx(arena_mb=a.arena, name="attnprobe")
    dev.activate()
    rng = np.random.default_rng(0)
    ws = [(rng.standard_normal((a.c, a.c)) * 0.02).astype(np.float32) for _ in range(4)]
    att = FullAttn(dev, a.n, a.c, a.h, a.w)
    att.set_weights(*ws)
    x = rng.standard_normal((a.n, a.c, a.h, a.w)).astype(np.float32)
    xd = dev.up(x.astype(np.float16))
    before = dev._off

    t0 = time.time()
    y = att.forward(xd)
    dev.q.finish()
    dt = time.time() - t0
    yv = dev.down(y, (a.n, a.c, a.h, a.w)).astype(np.float32)
    used = dev._off - before
    print("  forward %.3fs · arena delta %.0f MB · total %.0f MB" %
          (dt, used / 1048576, dev._off / 1048576))

    with torch.no_grad():
        xt = torch.from_numpy(x).flatten(2).transpose(1, 2)
        q = xt @ torch.from_numpy(ws[0]).T
        k = xt @ torch.from_numpy(ws[1]).T
        v = xt @ torch.from_numpy(ws[2]).T
        o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / np.sqrt(a.c))
        ref = (o @ torch.from_numpy(ws[3]).T).transpose(1, 2).reshape(
            a.n, a.c, a.h, a.w).numpy()
    rel = np.linalg.norm(yv - ref) / np.linalg.norm(ref)
    print("  forward rel %.2e" % rel)
    print("RESULT " + ("PASS" if rel < 2e-2 else "FAIL"))
    return 0 if rel < 2e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
