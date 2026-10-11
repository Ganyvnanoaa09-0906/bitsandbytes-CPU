"""quant2bit.py -- 2-bit blockwise weight quantisation, packed four indices per byte.

Why this shape: the format's loss cost is already measured. quantize_model_eval.py
took the trained 32x32 AR checkpoint and quantised every 2-D weight blockwise at 8,
4, 3 and 2 bits:

    fp32                5.5038
    4 bit uniform/nf    5.5080 / 5.5084   (+0.004)
    3 bit uniform/nf    5.5234 / 5.5231   (+0.019)
    2 bit uniform/nf    5.6354 / 5.6206   (+0.132 / +0.117)

so 2 bits costs about 2.1% of validation loss, and NF-style levels beat uniform by
11% at that width. What is not yet built is a representation that actually stores
2 bits per weight -- the existing general path quantises against an arbitrary code
table but indexes with 8 bits, so it saves nothing.

This module does the packing. Quantisation itself reuses the same scheme the kernels
use (absmax per block of 64, nearest level), so the numbers here describe the format
rather than an idealisation of it.

Bit order, stated because getting it wrong is silent: within a byte, index k of the
group occupies bits [2k, 2k+1], so the first weight of each group is in the two
lowest bits. A group is four consecutive weights inside a block; blocks are
independent, each with its own fp32 absmax.

Layout per tensor: packed (n/4 bytes) + absmax (n/block fp32), i.e. 2 + 32/64 = 2.5
bits per weight, against 32 for fp32 and 4.5 for the existing 4-bit path.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')

BLOCK = 64
BITS = 2
LEVELS = 1 << BITS


def nf_table(levels=BITS):
    """Normal mid-quantiles, normalised to [-1, 1]. Wins over uniform at 2 bits."""
    from statistics import NormalDist
    nd = NormalDist()
    n = 1 << levels
    q = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])
    q = q / np.abs(q).max()
    q[0], q[-1] = -1.0, 1.0
    return np.sort(q).astype(np.float32)


NF2 = nf_table()
UNIFORM2 = np.linspace(-1.0, 1.0, LEVELS).astype(np.float32)


def quantize(w: torch.Tensor, table=None, block: int = BLOCK):
    """Return (packed uint8, absmax float32). Pads the tail up to a multiple of 4."""
    table = NF2 if table is None else table
    shape = w.shape
    flat = w.detach().reshape(-1).numpy().astype(np.float32)
    n = flat.size
    pad = (-n) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    # block scales, padding the block dimension too
    bpad = (-len(flat)) % block
    if bpad:
        flat = np.concatenate([flat, np.zeros(bpad, dtype=np.float32)])
    wb = flat.reshape(-1, block)
    absmax = np.abs(wb).max(axis=1).astype(np.float32)
    absmax[absmax == 0] = 1.0
    x = wb / absmax[:, None]
    idx = np.abs(x[:, :, None] - table[None, None, :]).argmin(axis=2).astype(np.uint8)
    idx = idx.reshape(-1)[:n + pad]
    # pack four per byte, first index in the lowest bits
    g = idx.reshape(-1, 4).astype(np.uint16)
    packed = (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).astype(np.uint8)
    return (torch.from_numpy(packed.copy()), torch.from_numpy(absmax.copy()),
            n, pad, tuple(shape))


def dequantize(packed: torch.Tensor, absmax: torch.Tensor, n: int, pad: int,
               shape, table=None, block: int = BLOCK) -> torch.Tensor:
    table = NF2 if table is None else table
    p = packed.numpy().astype(np.uint16)
    idx = np.empty((p.size, 4), dtype=np.uint8)
    idx[:, 0] = (p & 0x03)
    idx[:, 1] = ((p >> 2) & 0x03)
    idx[:, 2] = ((p >> 4) & 0x03)
    idx[:, 3] = ((p >> 6) & 0x03)
    idx = idx.reshape(-1)[:n + pad]
    vals = table[idx]
    # apply the block scale
    reps = int(np.ceil(len(vals) / block))
    s = np.repeat(absmax.numpy(), block)[:len(vals)]
    out = (vals * s).astype(np.float32)[:n]
    return torch.from_numpy(out.copy()).reshape(shape)


def bytes_per_weight(block: int = BLOCK) -> float:
    return (BITS + 32.0 / block) / 8.0


if __name__ == '__main__':
    import time
    torch.set_num_threads(6)
    print('NF2 码表: %s' % NF2)
    print('每权重字节: %.4f（%.2f bit）  对比 fp32 4.0、4-bit 路径 %.4f'
          % (bytes_per_weight(), bytes_per_weight() * 8, (4 + 32.0 / BLOCK) / 8))
    print()

    # 1) 位序自检：手工构造一个已知字节，验证解包顺序
    p = torch.tensor([0b11100100], dtype=torch.uint8)      # idx = 0,1,2,3
    am = torch.ones(1, dtype=torch.float32)
    d = dequantize(p, am, 4, 0, (4,))
    expect = torch.tensor([NF2[0], NF2[1], NF2[2], NF2[3]])
    print('位序自检: %s ⇒ %s  期望 %s  %s'
          % (bin(0b11100100), np.round(d.numpy(), 4), np.round(expect.numpy(), 4),
             '✓' if torch.allclose(d, expect) else '✗'))
    print()

    # 2) 往返精度与相对误差，和之前的代理研究对齐
    print('%-22s %12s %12s %10s' % ('张量', '往返最大差', '相对误差', '压缩比'))
    print('-' * 62)
    import glob
    paths = (glob.glob(r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt') +
             glob.glob(r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt'))
    n_checked = 0
    for path in paths:
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        sd = ck.get('model', ck)
        for k, v in sd.items():
            if not (torch.is_tensor(v) and v.dtype == torch.float32 and v.ndim >= 2):
                continue
            if v.numel() < 4096 or n_checked >= 6:
                continue
            packed, absmax, n, pad, shape = quantize(v)
            back = dequantize(packed, absmax, n, pad, shape)
            rel = float((back - v).norm()) / (float(v.norm()) or 1.0)
            orig = v.numel() * 4
            comp = orig / (packed.numel() + absmax.numel() * 4)
            print('%-22s %12.4g %12.5f %9.2fx'
                  % ((k[:20] + '..') if len(k) > 22 else k,
                     float((back - v).abs().max()), rel, comp))
            n_checked += 1
    print()

    # 3) 打包/解包的速度
    w = torch.randn(2048, 2048)
    t0 = time.perf_counter()
    for _ in range(5):
        packed, absmax, n, pad, shape = quantize(w)
    t_q = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        back = dequantize(packed, absmax, n, pad, shape)
    t_d = (time.perf_counter() - t0) / 5
    mb = w.numel() * 4 / 2**20
    print('2048x2048 (%.0f MB fp32): 量化 %.1f ms (%.1f GB/s)  解包 %.1f ms (%.1f GB/s)'
          % (mb, t_q * 1e3, mb / 2**10 / t_q, t_d * 1e3, mb / 2**10 / t_d))
