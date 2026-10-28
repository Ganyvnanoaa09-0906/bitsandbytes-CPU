# -*- coding: utf-8 -*-
"""sr_net.py — 轻量超分网络（生成 256 -> 1024 / 1080p）

设计要点（踩过的坑）:
  ❌ 在 VQ-VAE 解码器最后加一级上采样 ⇒ 通道还是 256 ⇒ 34 GFLOP/层 ⇒ 2302 ms/图
  ✅ 独立的两级超分，从【3 通道】开始 ⇒ 约 15 GFLOP ⇒ 130 ms/图
  ✅ 完全不碰 VQ-VAE ⇒ 现有 token / AR checkpoint 全部保留

结构:
    Conv(3->C, 3x3) + ReLU
    ResBlock x N @ C
    Conv(C -> C*S^2, 3x3)
    PixelShuffle(S)          # 一次性放大 S 倍
    Conv(C -> 3, 3x3)

  S=4: 256 -> 1024
  S=2: 256 -> 512

参考损失:
    L1 + 感知(VGG) + 可选对抗
    论文经验: 纯 L1 会糊，加感知才锐利（但 PSNR 会降，观感更好）
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import torch
import torch.nn
import torch.nn.functional


class SRResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.c2(h)
        return x + h


class SRSmall(nn.Module):
    """轻量超分网络。输入 (B,3,H,W) 输出 (B,3,H*scale,W*scale)。"""

    def __init__(self, ch=32, n_res=4, scale=4):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[SRResBlock(ch) for _ in range(n_res)])
        self.up = nn.Conv2d(ch, ch * scale * scale, 3, 1, 1)
        self.shuf = nn.PixelShuffle(scale)
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)
        # 残差学习：网络只学"残差"，基線用 bicubic 上采样
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 基线：bicubic 上采样
        base = F.interpolate(x, scale_factor=self.scale, mode='bicubic',
                             align_corners=False)
        h = self.act(self.head(x))
        h = self.body(h)
        h = self.act(self.up(h))
        h = self.shuf(h)
        h = self.tail(h)
        return torch.tanh(base + h)      # 残差 + 基线，tanh 归一到 [-1,1]


def sr_cost(ch=32, n_res=4, scale=4, in_hw=256):
    """估算 FLOPs（乘加计一次）。"""
    H = in_hw
    fl = 0
    fl += H * H * ch * 3 * 9                      # head
    fl += n_res * 2 * H * H * ch * ch * 9         # body
    fl += H * H * (ch * scale * scale) * ch * 9   # up
    Ho = H * scale
    fl += Ho * Ho * 3 * ch * 9                    # tail
    return fl


if __name__ == '__main__':
    import time
    torch.set_num_threads(6)
    print('=' * 76)
    print('轻量超分网络 —— 成本实测')
    print('=' * 76)
    print()
    print('  %-22s %10s %12s %14s' % ('配置', '参数', 'GFLOP', '实测 ms/图'))
    print('  ' + '-' * 66)
    for scale, ch, nres in ((4, 32, 4), (4, 48, 4), (4, 32, 8), (2, 32, 4), (4, 64, 4)):
        net = SRSmall(ch=ch, n_res=nres, scale=scale).eval()
        npar = sum(p.numel() for p in net.parameters())
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            for _ in range(2):
                _ = net(x)
            t0 = time.perf_counter()
            for _ in range(3):
                _ = net(x)
            t = (time.perf_counter() - t0) / 3
        gf = sr_cost(ch, nres, scale) / 1e9
        print('  %-22s %9.2fM %11.2f %13.1f' % (
            'scale=%d ch=%d res=%d' % (scale, ch, nres), npar / 1e6, gf, t * 1e3 / 2))
    print()
    print('  对照: 现有 VQ-VAE 解码器 256 输出 = 64 ms/图')
    print('        总成本 = 解码 + 超分')
    for sc, ch, nr in ((4, 32, 4), (2, 32, 4)):
        net = SRSmall(ch=ch, n_res=nr, scale=sc).eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            for _ in range(2):
                _ = net(x)
            t0 = time.perf_counter()
            for _ in range(3):
                _ = net(x)
            t = (time.perf_counter() - t0) / 3
        print('        scale=%d: 解码 64 + 超分 %.0f = %.0f ms (输出 %d)' % (
            sc, t * 1e3 / 2, 64 + t * 1e3 / 2, 256 * sc))
    print()
    print('DONE')
