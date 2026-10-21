"""Debug the 10-layer full-GPU assembly: does it execute all layers, or is the
35-56ms just an unsynchronised submission time?

Builds N layers, runs forward only, downloads the final x2 and compares it with a
torch fp32 reference chain using the same fp16 weights/inputs.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_layer as L  # noqa: E402
import gt1030.full_gpu_model as FM  # noqa: E402

M, MD, FF, WROW = L.M, L.MD, L.FF, L.WROW
B, T, NH, NKV, HD = L.B, L.T, L.NH, L.NKV, L.HD
scale = L.scale


def cpu_layer(xt, wqkv, wo, w1, w3, w2):
    h1 = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6)
    qkvr = h1 @ wqkv
    orow = torch.zeros(M, MD)
    for bi in range(B):
        for hh in range(NH):
            rows = slice(bi * T, (bi + 1) * T)
            q = qkvr[rows, hh * HD:(hh + 1) * HD]
            kw = hh // 2
            k = qkvr[rows, MD + kw * HD:MD + (kw + 1) * HD]
            v = qkvr[rows, MD + NKV * HD + kw * HD:MD + NKV * HD + (kw + 1) * HD]
            s = (q @ k.T) * scale
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            s = s.masked_fill(~mask, float('-inf'))
            orow[rows, hh * HD:(hh + 1) * HD] = torch.softmax(s, -1) @ v
    x1 = xt + orow @ wo
    h2 = x1 * torch.rsqrt(x1.pow(2).mean(-1, keepdim=True) + 1e-6)
    return x1 + (torch.nn.functional.silu(h2 @ w1) * (h2 @ w3)) @ w2


def main():
    rng = np.random.default_rng(0)
    n = 3
    x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
    xdev = FM.upload(x)
    layers = [FM.Layer(rng, i) for i in range(n)]
    with torch.no_grad():
        cur = xdev
        for li, lay in enumerate(layers):
            cur = lay.forward(cur)
            L.A.queue.finish()
            tmp = np.empty((M, MD), dtype=np.float16)
            L.cl.enqueue_copy(L.A.queue, tmp, cur)
            L.A.queue.finish()
            print('layer %d out max %.3e min %.3e' % (li, np.abs(tmp).max(), tmp.min()), flush=True)
        out = tmp
    xt = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        ref = xt
        for lay in layers:
            ref = cpu_layer(ref,
                            torch.from_numpy(lay.WQKV['host'].astype(np.float32)),
                            torch.from_numpy(lay.WO['host'].astype(np.float32)),
                            torch.from_numpy(lay.W1['host'].astype(np.float32)),
                            torch.from_numpy(lay.W3['host'].astype(np.float32)),
                            torch.from_numpy(lay.W2['host'].astype(np.float32)))
    err = float(np.abs(out.astype(np.float32) - ref.numpy()).max())
    rel = err / (float(np.abs(ref.numpy()).max()) + 1e-9)
    print('N=%d forward max abs err %.3e rel %.3e' % (n, err, rel), flush=True)


if __name__ == '__main__':
    main()
