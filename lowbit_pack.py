# -*- coding: utf-8 -*-
"""lowbit_pack.py — 真正的低位宽打包/解包（2/3/4-bit），带往返自检

为什么需要:
    上一轮我的测试实现把量化索引存成 uint8（1 字节/权重），所以"7.11× 压缩"只是
    存储口径，RAM 根本没降；而且 forward 里 `codebook[idx] * amax` 会物化完整 fp32
    权重，导致端到端前向内存尖峰直接崩。
    这一步把它做对：**真正按位打包**，并且解包路径不物化整块 fp32。

打包格式（自定，简单且可验证）:
    2-bit: 4 个索引 → 1 字节（低位起）
    4-bit: 2 个索引 → 1 字节（低位起；与 bnb 的 nibble 顺序一致：偶 k = 低 nibble）
    3-bit: 8 个索引 → 3 字节（紧凑位流，低位起）

判据（三档）:
    1. **往返正确**：pack → unpack 必须逐元素等于原索引（这是硬判据）
    2. **存储真的降**：packed 字节数 ≈ n*bits/8（含 amax 开销）
    3. **解包速度**：解包 n 个权重要多久 —— 与 fp32 GEMM 比，判断是否值得

⚠️ 解包速度是这条路的关键。4-bit 的 CPU 融合核实测 M=1 时 6-11× 快于 fp32；
   2-bit 在我早先的原型里反而慢（3.2 GB/s vs 19.7），所以**必须实测而不是假设**。
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(int(__import__("os").environ.get("THREADS", "6")))


# ------------------------------------------------------------------ 2-bit
def pack2(idx: torch.Tensor) -> torch.Tensor:
    """idx: uint8/int，取值 0..3，任意形状。返回 packed uint8（最后一维按 4 个/字节）。"""
    x = idx.to(torch.uint8).reshape(-1)
    pad = (-x.numel()) % 4
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=torch.uint8)])
    x = x.reshape(-1, 4)
    b = (x[:, 0] & 3) | ((x[:, 1] & 3) << 2) | ((x[:, 2] & 3) << 4) | ((x[:, 3] & 3) << 6)
    return b.to(torch.uint8)


def unpack2(packed: torch.Tensor, n: int) -> torch.Tensor:
    p = packed.to(torch.int16)
    out = torch.empty(p.numel() * 4, dtype=torch.uint8)
    out[0::4] = (p & 3).to(torch.uint8)
    out[1::4] = ((p >> 2) & 3).to(torch.uint8)
    out[2::4] = ((p >> 4) & 3).to(torch.uint8)
    out[3::4] = ((p >> 6) & 3).to(torch.uint8)
    return out[:n]


# ------------------------------------------------------------------ 4-bit
def pack4(idx: torch.Tensor) -> torch.Tensor:
    x = idx.to(torch.uint8).reshape(-1)
    pad = (-x.numel()) % 2
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=torch.uint8)])
    x = x.reshape(-1, 2)
    # 与 bnb 一致：偶 k = 低 nibble
    return (x[:, 0] | (x[:, 1] << 4)).to(torch.uint8)


def unpack4(packed: torch.Tensor, n: int) -> torch.Tensor:
    p = packed.to(torch.int16)
    out = torch.empty(p.numel() * 2, dtype=torch.uint8)
    out[0::2] = (p & 0x0F).to(torch.uint8)
    out[1::2] = ((p >> 4) & 0x0F).to(torch.uint8)
    return out[:n]


# ------------------------------------------------------------------ 3-bit（紧凑位流）
def pack3(idx: torch.Tensor) -> torch.Tensor:
    """8 个 3-bit 索引 → 3 字节（低位起，位流拼接）。"""
    x = idx.to(torch.int64).reshape(-1)
    pad = (-x.numel()) % 8
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=torch.int64)])
    x = x.reshape(-1, 8)
    # 拼成 24-bit：v = i0 | i1<<3 | ... | i7<<21
    v = torch.zeros(x.shape[0], dtype=torch.int64)
    for j in range(8):
        v |= (x[:, j] & 7) << (3 * j)
    b0 = (v & 0xFF).to(torch.uint8)
    b1 = ((v >> 8) & 0xFF).to(torch.uint8)
    b2 = ((v >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=1).reshape(-1)


def unpack3(packed: torch.Tensor, n: int) -> torch.Tensor:
    p = packed.reshape(-1, 3).to(torch.int64)
    v = p[:, 0] | (p[:, 1] << 8) | (p[:, 2] << 16)
    out = torch.empty(v.numel() * 8, dtype=torch.uint8)
    for j in range(8):
        out[j::8] = ((v >> (3 * j)) & 7).to(torch.uint8)
    return out[:n]


PACKERS = {2: (pack2, unpack2), 3: (pack3, unpack3), 4: (pack4, unpack4)}


def selftest():
    print("=" * 78)
    print("低位宽打包/解包 往返自检")
    print("=" * 78)
    torch.manual_seed(0)
    allok = True
    for bits, (pk, up) in PACKERS.items():
        levels = 1 << bits
        for n in (16, 64, 4096, 4097, 12345):     # 含非 8/4/2 倍数，测 padding
            idx = torch.randint(0, levels, (n,), dtype=torch.uint8)
            packed = pk(idx)
            back = up(packed, n)
            ok = bool(torch.equal(back.to(torch.int64), idx.to(torch.int64)))
            ratio = packed.numel() / n
            exp = bits / 8.0
            allok &= ok and abs(ratio - exp) < 0.02
            if n == 4097:
                print("  %d-bit n=%-6d 往返=%s  packed/n=%.4f (期望 %.4f) 字节"
                      % (bits, n, "OK" if ok else "FAIL", ratio, exp))
        # 边界：全 0 / 全 max
        for val in (0, levels - 1):
            idx = torch.full((1000,), val, dtype=torch.uint8)
            back = up(pk(idx), 1000)
            ok = bool((back == val).all())
            allok &= ok
        print("  %d-bit 边界值(全0/全max) 往返 %s" % (bits, "OK" if allok else "FAIL"))
    print("\n判据: %s" % ("PASS —— 打包/解包逐元素正确" if allok else "FAIL"))
    return allok


def bench_unpack():
    print("\n" + "=" * 78)
    print("解包速度（决定这条路值不值得）")
    print("=" * 78)
    n = 4_000_000
    print("  %-8s %12s %14s %16s" % ("位宽", "打包字节", "解包 ms", "解包 GB/s(读)"))
    print("  " + "-" * 56)
    for bits, (pk, up) in PACKERS.items():
        levels = 1 << bits
        idx = torch.randint(0, levels, (n,), dtype=torch.uint8)
        packed = pk(idx)
        up(packed, n)
        t0 = time.perf_counter()
        for _ in range(3):
            out = up(packed, n)
        dt = (time.perf_counter() - t0) / 3
        gbs = packed.numel() / dt / 1e9
        print("  %-8d %12d %14.2f %16.2f" % (bits, packed.numel(), dt * 1e3, gbs))
    print("\n  参照: 本机纯内存拷贝 ~15-22 GB/s（16MB 档）")
    print("        fp32 GEMM 2048^3 实测 247 GFLOPS")
    return True


if __name__ == "__main__":
    ok = selftest()
    bench_unpack()
    raise SystemExit(0 if ok else 1)
