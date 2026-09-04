# -*- coding: utf-8 -*-
"""8bit/4bit 量化冻结层 + LoRA（R5 优化版 v3）。

支持 7B 级模型：8bit 权重 7GB、4bit(NF4/FP4) 权重 3.5GB，16GB 机器可跑。
特性：
  - dequant 结果跨步缓存（base 冻结）：速度回 fp32 水平，峰值内存不随层数膨胀
  - quant_dtype: '8bit' | 'nf4' | 'fp4'
  - blocksize 可调（内存/精度权衡）
用法：
    from quant_lora import quantize_model_8bit_lora, QuantLinearLora
    model = quantize_model_8bit_lora(model, quant_dtype='nf4', cache_dequant=True)
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.functional import (
    dequantize_4bit,
    dequantize_blockwise,
    quantize_4bit,
    quantize_blockwise,
)


class _QuantLinearFn(torch.autograd.Function):
    """无缓存版前向/反向：backward 重新 dequant（不保存 fp32 权重）。

    解决大模型（7B+）非缓存模式的内存爆炸：普通 F.linear 会把每层 dequant 的
    fp32 权重（253层 × 67MB ≈ 16GB）保存到 backward。这里 backward 重算，
    内存只留 nf4 权重 + 激活。
    """

    @staticmethod
    def forward(ctx, x, wq, stats, blocksize, quant_dtype, out_f, in_f, bias,
                lora_A, lora_B, scaling):
        w = _dequant_w(wq, stats, quant_dtype, blocksize, out_f, in_f)
        out = F.linear(x, w, bias)
        xa = x @ lora_A.t()
        out = out + (xa @ lora_B.t()) * scaling
        ctx.save_for_backward(x, lora_A, lora_B)
        ctx.wq = wq
        ctx.stats = stats          # QuantState 对象（非 tensor，存引用）
        ctx.blocksize = blocksize
        ctx.quant_dtype = quant_dtype
        ctx.out_f = out_f
        ctx.in_f = in_f
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, dout):
        x, lora_A, lora_B = ctx.saved_tensors
        w = _dequant_w(ctx.wq, ctx.stats, ctx.quant_dtype, ctx.blocksize, ctx.out_f, ctx.in_f)
        # x/dout 是 [B,T,K]/[B,T,O]；批量梯度累加转 2D matmul（oneDNN 最优路径，
        # einsum 的 bmm+归约慢 2-3 倍）
        dx = dout @ w                                    # [B,T,K] base 梯度
        dl = dout * ctx.scaling                          # [B,T,O]
        M = x.numel() // x.shape[-1]                     # B*T
        xa = x @ lora_A.t()                              # [B,T,r]
        dB = dl.reshape(M, -1).t() @ xa.reshape(M, -1)   # lora_B 梯度 [O,r]
        dlb = dl @ lora_B                                # [B,T,r]
        dA = dlb.reshape(M, -1).t() @ x.reshape(M, -1)   # lora_A 梯度 [r,K]
        dx = dx + dlb @ lora_A                           # LoRA 对 x 的梯度
        return dx, None, None, None, None, None, None, None, dA, dB, None


def _dequant_w(wq, stats, quant_dtype, blocksize, out_f, in_f) -> torch.Tensor:
    # nf4u/fp4u（unpacked uint8）走 8bit 标量查表快路径（pshufb 慢 ~4x）
    if quant_dtype in ("8bit", "nf4u", "fp4u"):
        return dequantize_blockwise(wq, stats, blocksize=blocksize).reshape(out_f, in_f)
    return dequantize_4bit(wq, stats, blocksize=blocksize, quant_type=quant_dtype)


class QuantLinearLora(nn.Module):
    """量化基座权重 + fp32 LoRA（8bit / 4bit-NF4 / 4bit-FP4）。

    cache_dequant=True：首次 forward 时 dequant 并缓存 fp32 权重（base 冻结，
    结果不变），之后每步直接 GEMM——速度同 fp32，内存=量化权重+fp32 缓存双份。
    cache_dequant=False：每步重新 dequant（省内存，适合长序列内存紧）。
    """

    def __init__(self, weight: torch.Tensor, bias, lora_r: int = 8, lora_alpha: int = 16,
                 cache_dequant: bool = True, quant_dtype: str = "8bit",
                 blocksize: int | None = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.quant_dtype = quant_dtype
        if quant_dtype == "8bit":
            self.blocksize = blocksize or 256
            w = weight.detach().float().reshape(-1)
            wq, stats = quantize_blockwise(w, blocksize=self.blocksize)
            self.register_buffer("wq", wq)
            self.stats = stats
        else:  # nf4 / fp4 / nf4u / fp4u
            self.blocksize = blocksize or 64
            w = weight.detach().float()
            qt = quant_dtype[:3]
            q, state = quantize_4bit(w, quant_type=qt, blocksize=self.blocksize)
            if quant_dtype.endswith("u"):
                # 解包 nibble -> uint8（顺序 hi,lo 与 avx2_dequant_4bit 一致），走 8bit 快路径
                from bitsandbytes.functional import QuantState
                packed = q.reshape(-1)
                lo = (packed & 0x0F).to(torch.uint8)
                hi = (packed >> 4).to(torch.uint8)
                wq = torch.stack([hi, lo], dim=1).reshape(-1)
                self.register_buffer("wq", wq)
                self.stats = QuantState(absmax=state.absmax, code=state.code.to(torch.float32),
                                        blocksize=self.blocksize, dtype=torch.uint8)
            else:
                self.register_buffer("wq", q)      # uint8 打包
                self.stats = state
        if bias is not None:
            self.register_buffer("bias", bias.detach().float().clone())
        else:
            self.register_buffer("bias", None)
        self.cache_dequant = cache_dequant
        self._w_cache: torch.Tensor | None = None
        self.scaling = lora_alpha / lora_r
        self.lora_A = nn.Parameter(torch.randn(lora_r, self.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_r))
        self.t_dequant = 0.0
        self.t_gemm = 0.0
        self.dequant_calls = 0

    def _dequant_w(self) -> torch.Tensor:
        if self.cache_dequant and self._w_cache is not None:
            return self._w_cache
        t0 = time.perf_counter()
        with torch.no_grad():
            if self.quant_dtype == "8bit":
                w = dequantize_blockwise(self.wq, self.stats, blocksize=self.blocksize)
                w = w.reshape(self.out_features, self.in_features)
            else:
                w = dequantize_4bit(self.wq, self.stats, blocksize=self.blocksize,
                                    quant_type=self.quant_dtype)
        self.t_dequant += time.perf_counter() - t0
        self.dequant_calls += 1
        if self.cache_dequant:
            self._w_cache = w
        return w

    @classmethod
    def from_quantized(cls, out_features: int, in_features: int,
                       wq: torch.Tensor, absmax: torch.Tensor, bias,
                       quant_dtype: str = "8bit", blocksize: int = 256,
                       lora_r: int = 8, lora_alpha: int = 16,
                       cache_dequant: bool = True,
                       code16: torch.Tensor | None = None) -> "QuantLinearLora":
        """从已量化权重直接构建（不经过 fp32），7B 级流式加载用。
        code16: nf4u/fp4u（unpacked）的 16 项 codebook（convert 时存进 meta）。"""
        from bitsandbytes.functional import QuantState
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.out_features, obj.in_features = out_features, in_features
        obj.quant_dtype = quant_dtype
        obj.blocksize = blocksize
        obj.register_buffer("wq", wq)
        if quant_dtype == "8bit":
            code = torch.arange(256, dtype=torch.float32) * (2 / 255) - 1
            obj.stats = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.uint8)
        elif quant_dtype in ("nf4u", "fp4u"):
            # unpacked 4bit：走 8bit 标量查表快路径（code 16 项）
            code16 = code16.to(torch.float32) if code16 is not None else torch.zeros(16)
            obj.stats = QuantState(absmax=absmax, code=code16, blocksize=blocksize, dtype=torch.uint8)
        else:
            # 打包 4bit：走 dequantize_4bit（pshufb 内核）
            obj.stats = QuantState(absmax=absmax, blocksize=blocksize, dtype=torch.float32,
                                   quant_type=quant_dtype, shape=(out_features, in_features))
        obj.register_buffer("bias", bias.detach().float().clone() if bias is not None else None)
        obj.cache_dequant = cache_dequant
        obj._w_cache = None
        obj.scaling = lora_alpha / lora_r
        obj.lora_A = nn.Parameter(torch.randn(lora_r, in_features) * 0.02)
        obj.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
        obj.t_dequant = 0.0
        obj.t_gemm = 0.0
        obj.dequant_calls = 0
        return obj

    def clear_cache(self):
        self._w_cache = None

    def forward(self, x):
        if self.cache_dequant:
            # 缓存版：首次 dequant 存 fp32 权重，之后每步直接 GEMM（快；8B+ 慎用，缓存占内存大）
            if self._w_cache is None:
                with torch.no_grad():
                    self._w_cache = _dequant_w(self.wq, self.stats, self.quant_dtype,
                                               self.blocksize, self.out_features, self.in_features)
            return (F.linear(x, self._w_cache, self.bias)
                    + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling)
        # 无缓存版：走自定义 Function，backward 重算 dequant（省内存，8B+ 用这个）
        return _QuantLinearFn.apply(
            x, self.wq, self.stats, self.blocksize, self.quant_dtype,
            self.out_features, self.in_features, self.bias,
            self.lora_A, self.lora_B, self.scaling)

    def extra_repr(self) -> str:
        return (f"out={self.out_features}, in={self.in_features}, "
                f"quant={self.quant_dtype}, bs={self.blocksize}, "
                f"cache={self.cache_dequant}")


class QuantLinearTrainable(nn.Module):
    """量化基座直接训练 —— 真量化存储 + LSQ 可学习标度（R7）。

    与旧伪量化（fp32 master 驻留、不减内存）不同，本版把基座权重真正存成
    8bit/NF4/FP4 码字，fp32 master 权重不再驻留：

      - 8bit：权重 1 B/元素，省 75%
      - 4bit(nf4/fp4)：权重 0.5 B/元素，省 87.5%

    训练时仅学习 per-block 标度 ``scale``（LSQ，Learned Step-size Quantization）：

      - forward：w = code[wq] * scale（纯 PyTorch 查表 + 广播，梯度直通 scale）
      - backward：梯度自然流回 scale，无需对离散码字做 STE
      - 优化器只维护 scale（每 block 一个标量，量级极小）

    收益：权重内存显著压缩（8bit 省 75%，4bit 省 87.5%）。基座码字本身不更新；
    若需同时更新基座与低秩增量，请改用 qlora（量化冻结基座 + fp32 LoRA）。
    """

    def __init__(self, weight: torch.Tensor, bias, quant_dtype: str = "nf4",
                 blocksize: int | None = None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.quant_dtype = quant_dtype

        w = weight.detach().float()
        if quant_dtype in ("8bit", "int8"):
            self.blocksize = blocksize or 256
            wq, state = quantize_blockwise(w.reshape(-1), blocksize=self.blocksize)
        elif quant_dtype.startswith(("nf4", "fp4")):
            self.blocksize = blocksize or 64
            wq, state = quantize_4bit(w, quant_type=quant_dtype[:3],
                                      blocksize=self.blocksize)
        else:
            raise ValueError(f"unsupported quant_dtype: {quant_dtype!r}")

        self.register_buffer("wq", wq)                                  # uint8 码字（冻结）
        self.register_buffer("code", state.code.float().clone())        # 码本（与量化内核一致）
        self.scale = nn.Parameter(state.absmax.detach().float().clone())  # LSQ 可学习标度

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().float().clone())
        else:
            self.register_buffer("bias", None)

    def _dequant(self) -> torch.Tensor:
        n = self.out_features * self.in_features
        if self.quant_dtype in ("8bit", "int8"):
            codes = self.code[self.wq.reshape(-1).long()]               # [n]
        else:
            packed = self.wq.reshape(-1)
            lo = (packed & 0x0F).long()
            hi = (packed >> 4).long()
            nibbles = torch.stack([hi, lo], dim=1).reshape(-1)[:n]      # [n]
            codes = self.code[nibbles]
        scale = self.scale.repeat_interleave(self.blocksize)[:n]        # [n]
        return (codes * scale).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._dequant(), self.bias)

    def quant_weight_bytes(self) -> int:
        """本层量化基座实际占用的内存字节（码字 + 标度 + 码本）。"""
        return (self.wq.numel() * self.wq.element_size()
                + self.scale.numel() * self.scale.element_size()
                + self.code.numel() * self.code.element_size())

    def extra_repr(self) -> str:
        return (f"out={self.out_features}, in={self.in_features}, "
                f"quant={self.quant_dtype}, bs={self.blocksize}, LSQ")


def quantize_model_8bit_lora(model, lora_r: int = 8, lora_alpha: int = 16,
                             cache_dequant: bool = True, quant_dtype: str = "8bit",
                             blocksize: int | None = None,
                             target: type = nn.Linear) -> tuple[int, int]:
    """把模型里所有 nn.Linear 换成量化 + LoRA。

    返回 (替换层数, 权重内存节省字节数)。只替换尚未量化的 nn.Linear。
    """
    n = 0
    saved = 0
    for name, m in list(model.named_children()):
        if isinstance(m, target) and not isinstance(m, QuantLinearLora):
            q = QuantLinearLora(m.weight, m.bias, lora_r=lora_r, lora_alpha=lora_alpha,
                                cache_dequant=cache_dequant, quant_dtype=quant_dtype,
                                blocksize=blocksize)
            wbytes = m.weight.numel() * 4
            qbytes = q.wq.numel() * q.wq.element_size() + q.stats.absmax.numel() * 4
            saved += wbytes - qbytes
            setattr(model, name, q)
            n += 1
        elif not isinstance(m, QuantLinearLora):
            sn, sb = quantize_model_8bit_lora(m, lora_r, lora_alpha, cache_dequant,
                                              quant_dtype, blocksize, target)
            n += sn
            saved += sb
    return n, saved
