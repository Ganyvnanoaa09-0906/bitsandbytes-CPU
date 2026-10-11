"""Block-quantized Conv2d for CPU inference.

bitsandbytes cannot quantize a convolution -- it implements only Linear and Embedding
variants -- so on SD1.5's UNet the 68.5% of parameters that live in Conv2d are stored in
fp32 no matter what. That is why the whole 4-bit -> 2-bit campaign could only shrink the
model by 2.7%: the lever was being applied to the 31.4% that was already quantized.

This module is the other 68.5%. It stores a convolution's weights as integer codes plus
one fp32 scale per block of `block` weights, and rebuilds an fp32 weight only for the
duration of the forward pass. Peak extra memory is one layer's weights, not the model's.

The measured quality, on real im2col patches from a real forward pass:
    8-bit, per output channel   SQNR 39.47 dB   (transparent)
    4-bit, per output channel   SQNR 13.86 dB   (poor)
    4-bit, per 64-weight block  SQNR 23.57 dB   (better than the 4-bit Linear path)

Note the ordering that surprised me: on convolutions per-channel is the BAD scheme and
per-block is the good one, the opposite of the usual intuition. Per-channel lets a single
large weight in a row set the scale for the whole row; a 64-weight block adapts locally.

Layout: a Conv2d weight is contiguous (C_out, C_in, kH, kW), so flattening to
(C_out, C_in*kH*kW) and cutting that into blocks of 64 gives blocks that never straddle an
output channel, which is what makes one scale per block meaningful.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantConv2d", "quantize_convs", "quant_bytes", "dequantize_all"]


def _pack4(codes):
    """codes: int8 tensor in [-7, 7], flattened. Two per byte."""
    n = codes.numel()
    pad = n % 2
    if pad:
        codes = torch.cat([codes, codes.new_zeros(1)])
    u = (codes.to(torch.int16) + 8).to(torch.uint8)     # 1..15
    hi = u[0::2]
    lo = u[1::2]
    return (hi << 4) | lo, n


def _unpack4(packed, n, like):
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = torch.empty(hi.numel() * 2, dtype=torch.int16, device=packed.device)
    out[0::2] = hi.to(torch.int16)
    out[1::2] = lo.to(torch.int16)
    return (out[:n] - 8).to(like)


class QuantConv2d(nn.Module):
    """Drop-in Conv2d whose weight is stored as integer codes + per-block scales."""

    def __init__(self, conv, bits=8, block=64):
        super().__init__()
        assert bits in (4, 8), "bits must be 4 or 8"
        self.bits = bits
        self.block = block
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.shape = conv.weight.shape
        self.qmax = (1 << (bits - 1)) - 1                 # 127 or 7
        self.has_bias = conv.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._quantize_from(conv.weight.detach())

    def _quantize_from(self, w):
        # Flattened to 1-D before blocking. The earlier version reshaped to
        # (out_channels, -1) and then cat'd a 1-D zero pad onto it, which raises as soon
        # as the tensor needs padding -- and the VAE's 1x1 convolutions are exactly
        # where that happens. Blocks are taken over the flat tensor, so one block may
        # span two output channels; that is the scheme the 23.57 dB figure was measured
        # on, and it is kept so the numbers stay comparable.
        flat = w.reshape(-1).contiguous()
        n = flat.numel()
        pad = (-n) % self.block
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blk = flat.reshape(-1, self.block)
        scale = (blk.abs().amax(dim=1).clamp_min(1e-12) / self.qmax).to(torch.float32)
        code = torch.clamp(torch.round(blk / scale.unsqueeze(1)),
                           -self.qmax, self.qmax).reshape(-1)[:n]
        self.n_codes = n
        self.register_buffer("scales", scale)
        if self.bits == 8:
            self.register_buffer("codes",
                                 (code.to(torch.int16) + 128).to(torch.uint8))
        else:
            packed, _ = _pack4(code.to(torch.int8))
            self.register_buffer("codes", packed)

    def weight_dequant(self):
        """Rebuild the fp32 weight. This is the only place memory is spent."""
        n = self.n_codes
        if self.bits == 8:
            c = self.codes.to(torch.int16) - 128
        else:
            c = _unpack4(self.codes, n, torch.int16)
        nb = self.scales.numel()
        need = nb * self.block
        # The codes were truncated back to n when they were stored, but a layer whose
        # weight count is not a multiple of `block` still has a full block's worth of
        # scales. Padded back here, which is the mirror of the pad applied on the way in;
        # without it a 16-weight layer like the VAE's post_quant_conv asks for a (1, 64)
        # reshape from 16 values.
        if c.numel() < need:
            c = torch.cat([c, c.new_zeros(need - c.numel())])
        c = c[:need].to(torch.float32).reshape(nb, self.block)
        w = (c * self.scales.unsqueeze(1)).reshape(-1)[:n]
        return w.reshape(self.shape)

    def forward(self, x):
        w = self.weight_dequant()
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel={self.kernel_size}, "
                f"bits={self.bits}, block={self.block}, "
                f"groups={self.groups}, bias={self.has_bias}")


def quant_bytes(mod):
    """Bytes actually held for the codes and their scales, plus any fp32 weights left."""
    codes = scales = fp32 = 0
    for m in mod.modules():
        if isinstance(m, QuantConv2d):
            codes += m.codes.numel() * m.codes.element_size()
            scales += m.scales.numel() * m.scales.element_size()
            if m.has_bias:
                fp32 += m.bias.numel() * 4
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            fp32 += m.weight.numel() * m.weight.element_size()
            if getattr(m, "bias", None) is not None:
                fp32 += m.bias.numel() * 4
    return codes + scales + fp32, codes, scales, fp32


def quantize_convs(module, bits=8, block=64, min_weights=0, skip=()):
    """Replace every nn.Conv2d under `module` with a QuantConv2d, in place."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            if any(s in name for s in skip):
                continue
            if child.weight.numel() < min_weights:
                continue
            setattr(module, name, QuantConv2d(child, bits=bits, block=block))
            replaced += 1
        else:
            replaced += quantize_convs(child, bits=bits, block=block,
                                       min_weights=min_weights, skip=skip)
    return replaced


def dequantize_all(module, bits=8, block=64, skip=()):
    """Replace every Conv2d weight with its quantize-then-dequantize value, in place.

    Keeps the module an ordinary nn.Conv2d, so this isolates the effect of the
    quantizer from the effect of the wrapper. If an image survives this and not the
    QuantConv2d version, the wrapper is at fault, not the quantization.
    """
    done = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            if any(s in name for s in skip):
                continue
            q = QuantConv2d(child, bits=bits, block=block)
            child.weight.data.copy_(q.weight_dequant())
            done += 1
        else:
            done += dequantize_all(child, bits=bits, block=block, skip=skip)
    return done


def conv_param_and_byte_report(unet):
    conv_params = sum(m.weight.numel() for m in unet.modules()
                      if isinstance(m, nn.Conv2d))
    return conv_params
