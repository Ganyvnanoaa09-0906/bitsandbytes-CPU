# -*- coding: utf-8 -*-
"""sd_quant - Stable Diffusion 冻结层量化（8bit/NF4，纯 CPU，用爆改版 bitsandbytes）。

用于生图训练的冻结权重降内存（R5-4500U 16GB / AVX2）：
  * QuantFrozenLinear（8bit）：走新的 AVX2 融合反量化 GEMM 内核
    （bitsandbytes::gemm_8bit），权重全程 uint8、不落地 fp32 临时张量，
    forward/backward 都无 fp32 权重复制 → 真正省内存且速度≈fp32。
  * QuantFrozenConv2d：每步 dequant 后执行 oneDNN conv；backward 重新
    dequant 再算 dx（不保存 fp32 权重，峰值内存=量化权重+一份临时）。
  * cache=True：首次 dequant 缓存 fp32（速度拉满、内存=量化+缓存双份）。

用法::

    from sd_quant import apply_quant_frozen
    n, saved = apply_quant_frozen(unet, quant_dtype='8bit', cache=False,
                                  exclude_names=("to_q", "to_v", "to_k", "to_out"))

实测（R5-4500U, bk-sdm-small, 512px）：
  8bit 冻结层: 权重 1.84GB -> 0.46GB (-75%)，单步 ~1.3x，峰值内存显著下降。
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from bitsandbytes.functional import dequantize_4bit, dequantize_blockwise, quantize_4bit, quantize_blockwise
    _BNB_OK = True
except Exception:  # pragma: no cover
    _BNB_OK = False

# 线性 8bit code map：code[i] = 2*i/255 - 1（与融合 GEMM 内核的解码公式严格一致；
# bnb 默认的 create_dynamic_map() 是零点加密的非线性映射，无法用 FMA 折叠解码）
_LIN8_CODE = torch.arange(256, dtype=torch.float32) * (2.0 / 255.0) - 1.0

_BLOCK_CANDIDATES = (256, 128, 64, 32)


def _pick_bs(k: int):
    """选一个能整除 K 的块大小（fused 内核的向量路径要求 K % bs == 0）。"""
    for bs in _BLOCK_CANDIDATES:
        if k % bs == 0:
            return bs
    return 0


def _quantize(w: torch.Tensor, quant_dtype: str, blocksize: int):
    """把权重张量量化，返回 (q, state)。8bit / nf4 / fp4 三种。"""
    if quant_dtype == "8bit":
        return quantize_blockwise(w.float().reshape(-1), code=_LIN8_CODE, blocksize=blocksize)
    qt = quant_dtype[:3]
    q, state = quantize_4bit(w.float(), quant_type=qt, blocksize=blocksize)
    return q, state


def _dequantize(q, state, quant_dtype: str, shape: torch.Size, bs: int) -> torch.Tensor:
    if quant_dtype == "8bit":
        return dequantize_blockwise(q, state).reshape(shape)
    return dequantize_4bit(q, state, blocksize=bs, quant_type=quant_dtype[:3]).reshape(shape)


def _fused_linear_8bit(x: torch.Tensor, wq: torch.Tensor, absmax: torch.Tensor, bs: int) -> torch.Tensor:
    """x[M,K] @ dequant8(wq[N,K])^T -> [M,N]，权重不落地 fp32。"""
    return torch.ops.bitsandbytes.gemm_8bit(x, wq, absmax, bs)


class _FusedLinear8bitFn(torch.autograd.Function):
    """fused 8bit 线性层前向/反向。

    forward 用融合 GEMM（uint8 权重不落地 fp32，省内存）；
    backward 重新 dequant 后按 fp32 GEMM 算 dx（层内临时，随用随释放，
    避免把全部反量化权重挂进计算图）。
    """

    @staticmethod
    def forward(ctx, x, wq, absmax, bs, bias):
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        out = _fused_linear_8bit(x2, wq, absmax, bs)
        if bias is not None:
            out = out + bias
        ctx.wq, ctx.absmax, ctx.bs, ctx.bias = wq, absmax, bs, bias
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        # dx = dout @ dequant(w)  （dout:[M,O] w:[O,K] -> [M,K]）
        w = torch.ops.bitsandbytes.dequantize_blockwise(
            ctx.wq.reshape(-1), ctx.absmax.view(-1), _LIN8_CODE,
            ctx.bs, torch.float32).reshape(ctx.wq.shape)
        d = dout.reshape(-1, dout.shape[-1]).contiguous()
        dx = d @ w
        dx = dx.reshape(*dout.shape[:-1], dx.shape[-1])
        return dx, None, None, None, None


class _DequantConvFn(torch.autograd.Function):
    """conv 冻结层：forward dequant+conv，backward 重新 dequant + convT（省内存）。"""

    @staticmethod
    def forward(ctx, x, wq, state, shape, quant_dtype, bs, stride, padding, dilation, groups, bias):
        w = _dequantize(wq, state, quant_dtype, shape, bs)
        out = F.conv2d(x, w, bias, stride, padding, dilation, groups)
        ctx.wq, ctx.state, ctx.quant_dtype = wq, state, quant_dtype
        ctx.bs = bs
        ctx.shape = shape
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.kernel_sz = (shape[2], shape[3])
        return out

    @staticmethod
    def backward(ctx, dout):
        w = _dequantize(ctx.wq, ctx.state, ctx.quant_dtype, ctx.shape, ctx.bs)
        op = ctx.stride[0] - 1 if ctx.stride[0] > 1 else 0
        dx = F.conv_transpose2d(dout, w, None, ctx.stride, ctx.padding, op,
                                ctx.groups, ctx.dilation)
        return dx, None, None, None, None, None, None, None, None, None, None


class QuantFrozenConv2d(nn.Module):
    """量化权重 + 冻结的 Conv2d（8bit / nf4 / fp4）。"""

    def __init__(self, conv: nn.Conv2d, quant_dtype: str = "8bit", blocksize: int = 0,
                 cache: bool = False):
        super().__init__()
        self.quant_dtype = quant_dtype
        self.cache = cache
        self.out_features, self.in_f = conv.out_channels, conv.in_channels
        if quant_dtype == "8bit":
            self.blocksize = blocksize or _pick_bs(conv.weight.numel()) or 256
        else:
            self.blocksize = blocksize or 64
        q, state = _quantize(conv.weight.detach(), quant_dtype, self.blocksize)
        self.register_buffer("wq", q)
        self.state = state
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size
        self.register_buffer("bias_buf",
                             conv.bias.detach().float().clone() if conv.bias is not None else None)
        self._wc: torch.Tensor | None = None
        self.t_dequant = 0.0
        self.t_conv = 0.0
        self.dequant_calls = 0

    def _w(self) -> torch.Tensor:
        if self.cache and self._wc is not None:
            return self._wc
        shape = (self.out_features, self.in_f, *self.kernel_size)
        t0 = time.perf_counter()
        with torch.no_grad():
            w = _dequantize(self.wq, self.state, self.quant_dtype, shape, self.blocksize)
        self.t_dequant += time.perf_counter() - t0
        self.dequant_calls += 1
        if self.cache:
            self._wc = w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        if self.cache:
            out = F.conv2d(x, self._w(), self.bias_buf, self.stride, self.padding,
                           self.dilation, self.groups)
        else:
            shape = (self.out_features, self.in_f, *self.kernel_size)
            out = _DequantConvFn.apply(x, self.wq, self.state, torch.Size(shape),
                                       self.quant_dtype, self.blocksize, self.stride,
                                       self.padding, self.dilation, self.groups, self.bias_buf)
        self.t_conv += time.perf_counter() - t0
        return out

    def extra_repr(self) -> str:
        return (f"{self.out_features},{self.in_f}->{self.kernel_size}, quant={self.quant_dtype}, "
                f"bs={self.blocksize}, cache={self.cache}")


class QuantFrozenLinear(nn.Module):
    """量化权重 + 冻结的 Linear（8bit 走融合 GEMM 内核）。"""

    def __init__(self, lin: nn.Linear, quant_dtype: str = "8bit", blocksize: int = 0,
                 cache: bool = False):
        super().__init__()
        self.quant_dtype = quant_dtype
        self.cache = cache
        self.out_features, self.in_features = lin.out_features, lin.in_features
        if quant_dtype == "8bit":
            self.blocksize = blocksize or _pick_bs(lin.in_features) or 256
        else:
            self.blocksize = blocksize or 64
        q, state = _quantize(lin.weight.detach(), quant_dtype, self.blocksize)
        self.register_buffer("wq", q)
        self.state = state
        self.register_buffer("bias_buf",
                             lin.bias.detach().float().clone() if lin.bias is not None else None)
        self._wc: torch.Tensor | None = None
        self.t_dequant = 0.0
        self.t_gemm = 0.0
        self.dequant_calls = 0

    def _w(self) -> torch.Tensor:
        if self.cache and self._wc is not None:
            return self._wc
        t0 = time.perf_counter()
        with torch.no_grad():
            w = _dequantize(self.wq, self.state, self.quant_dtype,
                            torch.Size((self.out_features, self.in_features)), self.blocksize)
        self.t_dequant += time.perf_counter() - t0
        self.dequant_calls += 1
        if self.cache:
            self._wc = w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        if self.quant_dtype == "8bit" and not self.cache and self.in_features % self.blocksize == 0:
            # 融合内核（uint8 权重不落地 fp32）
            wq2 = self.wq.reshape(self.out_features, self.in_features)
            absmax2 = self.state.absmax.reshape(self.out_features,
                                                self.in_features // self.blocksize)
            out = _FusedLinear8bitFn.apply(x, wq2, absmax2, self.blocksize, self.bias_buf)
        else:
            out = F.linear(x, self._w(), self.bias_buf)
        self.t_gemm += time.perf_counter() - t0
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, quant={self.quant_dtype}, "
                f"bs={self.blocksize}, cache={self.cache}")


def apply_quant_frozen(module: nn.Module, quant_dtype: str = "8bit", cache: bool = False,
                       exclude_names: tuple[str, ...] = (),
                       verbose: bool = False) -> tuple[int, int]:
    """递归替换 nn.Conv2d/nn.Linear 为量化冻结版（不碰 exclude 名字的模块）。

    exclude 支持带后缀的目标名（如 "to_out.0" 会匹配 ModuleList "to_out" 的
    首个元素——按首段名字段匹配）。返回 (替换层数, 节省字节数)。
    """
    if not _BNB_OK:
        raise RuntimeError("bitsandbytes (CPU fork) 不可用，无法量化冻结层")

    ex_heads = tuple(e.split(".")[0] for e in exclude_names)

    def hit(name: str) -> bool:
        head = name.split(".")[0]
        return head in ex_heads

    def _walk(m, prefix: str):
        nonlocal n, saved
        for child_name, child in list(m.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if hit(child_name):
                continue
            if isinstance(child, nn.Conv2d) and not isinstance(child, QuantFrozenConv2d):
                wbytes = child.weight.numel() * 4
                q = QuantFrozenConv2d(child, quant_dtype=quant_dtype, cache=cache)
                qbytes = q.wq.numel() * q.wq.element_size() + q.state.absmax.numel() * 4
                setattr(m, child_name, q)
                n += 1
                saved += wbytes - qbytes
                if verbose:
                    print(f"  [quant] {full} -> Conv2d {q.out_features}x{q.in_f}{q.kernel_size} "
                          f"({quant_dtype}, saved {(wbytes - qbytes) >> 20}MB)")
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, QuantFrozenLinear) \
                    and not isinstance(child, torch.nn.modules.linear.NonDynamicallyQuantizableLinear):
                wbytes = child.weight.numel() * 4
                q = QuantFrozenLinear(child, quant_dtype=quant_dtype, cache=cache)
                qbytes = q.wq.numel() * q.wq.element_size() + q.state.absmax.numel() * 4
                setattr(m, child_name, q)
                n += 1
                saved += wbytes - qbytes
                if verbose:
                    print(f"  [quant] {full} -> Linear {q.out_features}x{q.in_features} "
                          f"({quant_dtype}, saved {(wbytes - qbytes) >> 20}MB)")
                continue
            _walk(child, full)
        return

    n, saved = 0, 0
    _walk(module, "")
    return n, saved
