# -*- coding: utf-8 -*-
"""统一解码器加载：同时支持自训 VQVAE 和 diffusers 的 AutoencoderKL（SDXL VAE）。

为什么需要：潜空间换成 SDXL VAE 之后，解码路径完全不同：
  自训 VQVAE:      dec(z)                        直接吃 z
  SDXL VAE:        decode(z / scaling_factor).sample   必须先除回去
                  （编码时乘了 scaling_factor=0.13025，解码必须除）

这个差异是"静默出错"的高发点 —— 忘了除 scaling_factor 出来的是垃圾，
但不会报错。所以抽成一处，并且在里面断言形状。
"""
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
import os

import torch


class Decoder:
    """统一接口：dec(z) -> 图像 [-1,1]。"""

    def __init__(self, path, verbose=True):
        self.path = path
        self.kind = None
        if os.path.isdir(path):
            from diffusers import AutoencoderKL
            # SDXL VAE 权重是 fp16 变体名；fp16 解码会出 NaN，必须 fp32
            self.m = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float32,
                                                   variant='fp16').float().eval()
            self.sf = float(self.m.config.scaling_factor)
            self.kind = 'sdxl'
            if verbose:
                print('[decode] SDXL VAE  scaling_factor=%.5f  params=%.1fM'
                      % (self.sf, sum(p.numel() for p in self.m.parameters()) / 1e6), flush=True)
        else:
            import sys
            sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
            from small_vqvae import vqvae_from_ckpt
            vk = torch.load(path, map_location='cpu', weights_only=False)
            self.m, _ = vqvae_from_ckpt(vk)
            self.m.eval()
            self.sf = 1.0
            self.kind = 'vqvae'
            if verbose:
                print('[decode] 自训 VQVAE  %s' % os.path.basename(path), flush=True)

    @property
    def grid(self):
        """该 VAE 的下采样倍数推断用的网格边长（给 256 输入时）。"""
        return 32 if self.kind == 'sdxl' else None

    @torch.no_grad()
    def dec(self, z):
        if self.kind == 'sdxl':
            return self.m.decode(z / self.sf).sample.clamp(-1, 1)
        return self.m.dec(z).clamp(-1, 1)
