"""Torch reference implementation of the fused operators (the contract)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ['rms_norm', 'silu_mul', 'rope', 'fused_ce']


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dim: x * rsqrt(mean(x^2)+eps) * weight."""
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y * weight


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU gate: silu(gate) * up."""
    return F.silu(gate) * up


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding: x*cos + rotate_half(x)*sin.

    x: (..., D), D even; cos/sin broadcast to x's shape.
    rotate_half: [x0,x1,...,xd-1] -> [-x_{d/2},...,-x_{d-1}, x0,...,x_{d/2-1}]
    """
    if x.shape[-1] % 2:
        raise ValueError('rope last dim must be even')
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def fused_ce(logits: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100) -> torch.Tensor:
    """Mean cross entropy over N rows (logits [N,V], labels [N])."""
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
