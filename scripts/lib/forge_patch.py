# -*- coding: utf-8 -*-
"""forge_patch.py —— 把 nn.Linear 换成 AVX2 内核的 autograd 实现（可开可关）。

为什么这么做而不是改模型源码：
  模型文件 small_image_model_v2.py 被训练/采样/评估多处共用，改源码风险大。
  这里用【运行时替换 forward 绑定】的方式，随时 enable/disable，模型权重和
  结构一动不动 —— 出问题一行关掉就回到原状。

三处矩阵乘对应关系（W 的形状跟 nn.Linear 一致，是 [out, in]）：
  前向   y  = x @ W^T          -> NT: A=x[M,K],      B=W[N,K]
  grad_x gx = gy @ W           -> NN: A=gy[M,N],     B=W[N,K]
  grad_W gW = gy^T @ x         -> TN: A=gy[M,N],     B=x[M,K]

用法：
    import forge_patch
    forge_patch.enable(model)                    # 全部 nn.Linear
    forge_patch.enable(model, only=('w12','w3')) # 只换 FFN
    forge_patch.disable(model)
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import torch
import torch.nn as nn

import forge_gemm as FG

_MARK = '_forge_original_forward'

# ★ 统计：模型里 GEMM 的操作数常常是上游算子的【视图】（非连续），
#   而隔离基准用的是 torch.randn（永远连续）。如果这里大量拷贝，
#   就解释了"单形状很快、真实步里却慢 1.6 倍"。
STATS = {'fwd_calls': 0, 'fwd_copy': 0, 'fwd_copy_bytes': 0,
         'bwd_calls': 0, 'bwd_copy': 0, 'bwd_copy_bytes': 0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0


def _as_contig(t, which):
    """reshape 成 2D 并在必要时拷贝；同时记账。"""
    shp = t.shape
    x2 = t.reshape(-1, shp[-1])
    STATS[which + '_calls'] += 1
    if not x2.is_contiguous():
        STATS[which + '_copy'] += 1
        STATS[which + '_copy_bytes'] += x2.numel() * x2.element_size()
        x2 = x2.contiguous()
    return x2, shp


class ForgeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W):
        x2, shp = _as_contig(x, 'fwd')
        y = FG.mm(x2, W, 'NT')
        ctx.save_for_backward(x2, W)
        ctx.in_shape = shp
        return y.reshape(*shp[:-1], W.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, W = ctx.saved_tensors
        gy2, _ = _as_contig(gy, 'bwd')
        gx = FG.mm(gy2, W, 'NN')      # [M,N] @ [N,K] -> [M,K]
        gW = FG.mm(gy2, x2, 'TN')     # [M,N]^T @ [M,K] -> [N,K]
        return gx.reshape(ctx.in_shape), gW


def _patched_forward(self, x):
    return ForgeLinearFn.apply(x, self.weight)


def enable(model, only=None, threads=None):
    """把 model 里的 nn.Linear 换成 forge 版本。返回被替换的模块名列表。"""
    if threads:
        FG.set_threads(threads)
    done = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if only and not any(k in name for k in only):
            continue
        if hasattr(mod, _MARK):
            continue
        setattr(mod, _MARK, mod.forward)
        import types
        mod.forward = types.MethodType(_patched_forward, mod)
        done.append(name)
    return done


def disable(model):
    """还原成原来的 forward。"""
    done = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and hasattr(mod, _MARK):
            mod.forward = getattr(mod, _MARK)
            delattr(mod, _MARK)
            done.append(name)
    return done


def enabled_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and hasattr(m, _MARK)]
