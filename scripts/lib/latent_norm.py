# -*- coding: utf-8 -*-
"""latent_norm.py — 潜变量归一化/反归一化的【唯一实现】，训练与采样共用。

为什么单独抽出来：训练把潜变量变换到单位尺度，采样必须做【完全一致的逆变换】。
两边各写一份是最容易悄悄错位的地方（一旦错位，出图就是糊的，而且很难查）。

支持两种模式，由 ckpt 里的字段决定：
  旧: 只有 'scale'（单标量）           -> z = x * scale
  新: 'chan_mean' + 'chan_std'（逐通道）-> z = x * chan_std + chan_mean

为什么默认要用逐通道：实测这份 VQ 潜变量逐通道 std 差 12.5×（0.167~2.085）。
单标量归一化后仍差 12.5×，而 MSE 的权重按方差走 => 小方差通道权重小 156 倍。
但消融实验（diag_channel_ablation.py）显示：通道 49-64 方差只占 3.7%，
删掉却让重建 PSNR 掉 10.23 dB，与删掉方差占 38% 的前 8 个通道一样严重。
=> 方差排序 ≠ 重要性排序，必须逐通道拉平，否则模型学不到那些通道。

可移植性：所有参数都【从数据统计出来】并记进 ckpt，没有任何写死的机器/数据集常数。
"""
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


def normalize(L, train_idx, chan_norm=True):
    """把潜变量变换到单位尺度。返回 (L_norm, meta)。

    meta 会被写进 ckpt，采样端据此反变换。
    """
    N, C = L.shape[0], L.shape[1]
    flat = L[train_idx].reshape(len(train_idx), C, -1)
    chan_mean = flat.mean(dim=(0, 2))
    chan_std = flat.std(dim=(0, 2)).clamp_min(1e-6)
    scale = float(L[train_idx].std())
    if chan_norm:
        Ln = (L - chan_mean.view(1, C, 1, 1)) / chan_std.view(1, C, 1, 1)
        meta = dict(scale=scale, chan_norm=True,
                    chan_mean=chan_mean.clone(), chan_std=chan_std.clone())
    else:
        Ln = L / scale
        meta = dict(scale=scale, chan_norm=False)
    return Ln, meta


def denormalize(z, meta):
    """把模型输出的潜变量变回解码器认的尺度。"""
    if meta.get('chan_norm') and meta.get('chan_mean') is not None:
        C = z.shape[1]
        return z * meta['chan_std'].view(1, C, 1, 1).to(z.dtype) + \
               meta['chan_mean'].view(1, C, 1, 1).to(z.dtype)
    return z * float(meta.get('scale', 1.0))


def meta_from_ckpt(ck):
    """从 ckpt 里取归一化参数。旧 ckpt 只有 scale，新 ckpt 有 chan_*。"""
    return dict(scale=float(ck.get('scale', 1.0)),
                chan_norm=bool(ck.get('chan_norm', False)),
                chan_mean=ck.get('chan_mean'),
                chan_std=ck.get('chan_std'))


def describe(meta):
    if meta.get('chan_norm') and meta.get('chan_mean') is not None:
        cs = meta['chan_std']
        return "逐通道标准化 (std %.3f~%.3f)" % (float(cs.min()), float(cs.max()))
    return "单标量 scale=%.6f" % float(meta.get('scale', 1.0))
