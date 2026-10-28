"""quantnbit.py -- a general N-bit blockwise quantiser, packed.

Why general rather than a 2-bit module with a 3-bit sibling: the two differ only in
the bit count and the table, and a single code path means the 2-bit case can be
regression-tested against the number already measured (5.5538 for Linears only) every
time the 3-bit path is exercised. Two implementations would need two sets of
verification; one needs one.

Measured costs on the trained 32x32 AR checkpoint, 512-sample validation slice:

    fp32                 5.5038
    3 bit uniform        5.5234   (+0.0195)   <- the target this must reproduce
    3 bit nf             5.5231   (+0.0193)   <- 0.0003 better, i.e. not worth NF
    2 bit uniform        5.6354   (+0.1316)
    2 bit nf             5.6206   (+0.1168)   <- 11% better, so NF matters only at 2

So 3 bits uses uniform levels: the difference against NF is 0.0003 of loss, well
inside noise, and uniform keeps the decode a multiply-add.

Packing: n values of `bits` bits are packed consecutively, LSB first, into
ceil(n*bits/8) bytes. Value k starts at bit k*bits, which may span a byte boundary --
3 bits does, 2 and 4 do not. Getting this wrong is silent, so the module ships a
self-test that packs a known index sequence and reads it back.

Storage per weight: bits + 32/blocksize. At 64 that is 3.5 bits for 3-bit, against
32 for fp32.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


DEFAULT_BLOCK = 64


def uniform_table(bits):
    """2**bits evenly spaced levels over [-1, 1]. What the 3-bit path uses."""
    return np.linspace(-1.0, 1.0, 1 << bits).astype(np.float32)


def nf_table(bits):
    """Normal mid-quantiles normalised to [-1, 1]. Wins at 2 bits, loses at 4."""
    from statistics import NormalDist
    nd = NormalDist()
    n = 1 << bits
    q = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])
    q = q / np.abs(q).max()
    q[0], q[-1] = -1.0, 1.0
    return np.sort(q).astype(np.float32)


def pack_indices(idx: np.ndarray, bits: int) -> np.ndarray:
    """Pack values of `bits` bits LSB-first, consecutive, into bytes.

    Vectorised: value k lands at bit k*bits and spans at most two bytes. np.add.at is
    safe rather than np.bitwise_or.at because the per-value bit ranges are disjoint,
    so the contributions never collide.
    """
    idx = idx.astype(np.uint32).reshape(-1)
    n = idx.size
    total_bits = n * bits
    out = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    if n == 0:
        return out
    starts = np.arange(n, dtype=np.int64) * bits
    byte = starts >> 3
    off = (starts & 7).astype(np.uint32)
    lo = ((idx << off) & np.uint32(0xFF)).astype(np.uint8)
    np.add.at(out, byte, lo)
    spill = off.astype(np.int64) + bits - 8
    sel = spill > 0
    if sel.any():
        hi = (idx[sel] >> (bits - spill[sel])).astype(np.uint8)
        np.add.at(out, byte[sel] + 1, hi)
    return out


def unpack_indices(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    """Inverse of pack_indices, vectorised."""
    p = packed.astype(np.uint32)
    mask = np.uint32((1 << bits) - 1)
    if count == 0:
        return np.zeros(0, dtype=np.uint32)
    starts = np.arange(count, dtype=np.int64) * bits
    byte = starts >> 3
    off = (starts & 7).astype(np.uint32)
    v = (p[byte] >> off).astype(np.uint32)
    spill = off.astype(np.int64) + bits - 8
    sel = spill > 0
    if sel.any():
        nxt = np.zeros(count, dtype=np.uint32)
        nxt[sel] = p[byte[sel] + 1]
        # shift the next byte up so its low bits land above what the first byte gave
        v[sel] |= (nxt[sel] << (np.uint32(8) - off[sel]))
    return v & mask


def quantize(w: torch.Tensor, bits: int = 3, block: int = DEFAULT_BLOCK,
             table=None):
    """Return (packed, absmax, meta). meta carries what dequantize needs."""
    if table is None:
        table = uniform_table(bits)
    flat = w.detach().reshape(-1).numpy().astype(np.float32)
    n = flat.size
    if n % block:
        flat = np.concatenate([flat, np.zeros(block - n % block, dtype=np.float32)])
    wb = flat.reshape(-1, block)
    absmax = np.abs(wb).max(axis=1).astype(np.float32)
    absmax[absmax == 0] = 1.0
    x = wb / absmax[:, None]
    idx = np.abs(x[:, :, None] - table[None, None, :]).argmin(axis=2)
    packed = pack_indices(idx.reshape(-1), bits)
    return (torch.from_numpy(packed), torch.from_numpy(absmax.copy()),
            dict(n=n, bits=bits, block=block, table=table, shape=tuple(w.shape)))


def dequantize(packed, absmax, meta) -> torch.Tensor:
    bits, block, n = meta['bits'], meta['block'], meta['n']
    table = meta['table']
    padded = ((n + block - 1) // block) * block
    idx = unpack_indices(packed.numpy(), padded, bits)
    vals = table[idx]
    s = np.repeat(absmax.numpy(), block)[:padded]
    return torch.from_numpy((vals * s).astype(np.float32)[:n].copy()).reshape(meta['shape'])


def bytes_per_weight(bits: int, block: int = DEFAULT_BLOCK) -> float:
    return (bits + 32.0 / block) / 8.0


if __name__ == '__main__':
    import time
    torch.set_num_threads(6)
    print('通用 N-bit 打包器自检')
    print()
    ok = True
    for bits in (2, 3, 4, 5):
        levels = 1 << bits
        # a sequence that exercises every level and byte boundaries
        seq = np.arange(0, 2000) % levels
        p = pack_indices(seq, bits)
        back = unpack_indices(p, len(seq), bits)
        same = np.array_equal(seq.astype(np.uint32), back)
        ok &= same
        print('  %d bit: %d 值 → %d 字节（%.3f 字节/值）  往返 %s'
              % (bits, len(seq), p.size, p.size / len(seq), '✓' if same else '✗'))
    print()
    print('码表:')
    for bits in (2, 3, 4):
        print('  %d bit uniform %s' % (bits, np.round(uniform_table(bits), 4)))
    print()
    print('每权重字节: 2 bit %.4f   3 bit %.4f   4 bit %.4f   fp32 4.0'
          % (bytes_per_weight(2), bytes_per_weight(3), bytes_per_weight(4)))
    print()
    # side-by-side with the shipped NF4 table at 4 bits
    NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0], dtype=np.float32)
    w = torch.randn(512, 512)
    for bits, tab, name in ((4, NF4, 'nf4（本仓库真表）'), (4, uniform_table(4), 'uniform 4'),
                            (3, uniform_table(3), 'uniform 3'), (2, nf_table(2), 'nf 2')):
        pk, am, meta = quantize(w, bits, table=tab)
        back = dequantize(pk, am, meta)
        rel = float((back - w).norm()) / float(w.norm())
        print('  %-16s 相对误差 %.5f  打包 %d 字节 vs fp32 %d ⇒ %.1fx'
              % (name, rel, pk.numel(), w.numel() * 4, w.numel() * 4 / pk.numel()))
    t0 = time.perf_counter()
    for _ in range(3):
        pk, am, meta = quantize(w, 3)
    tq = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        dequantize(pk, am, meta)
    td = (time.perf_counter() - t0) / 3
    mb = w.numel() * 4 / 2**20
    print()
    print('速度（512x512, %.0f MB）: 量化 %.1f ms  解包 %.1f ms' % (mb, tq * 1e3, td * 1e3))
    print('全部自检通过 ✓' if ok else '有失败 ✗')
