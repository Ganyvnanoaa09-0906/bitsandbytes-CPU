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

def linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                         labels: torch.Tensor, chunk: int = 1024,
                         ignore_index: int = -100) -> torch.Tensor:
    """Cross entropy of hidden @ weight.T without materialising full logits."""
    M, D = hidden.shape
    V = weight.shape[0]
    mask = labels != ignore_index
    tgt = labels.clone()
    tgt[~mask] = 0
    target_logit = (hidden * weight[tgt]).sum(dim=-1)
    maxv = torch.full((M,), -float('inf'), dtype=torch.float64)
    sumexp = torch.zeros(M, dtype=torch.float64)
    target_logit = target_logit.double()
    for s in range(0, V, chunk):
        e = min(V, s + chunk)
        logits = (hidden @ weight[s:e].t()).double()
        cmax = logits.max(dim=-1).values
        newmax = torch.maximum(maxv, cmax)
        sumexp = sumexp * torch.exp(maxv - newmax) + torch.exp(logits - newmax[:, None]).sum(dim=-1)
        maxv = newmax
    loss = torch.log(sumexp) + maxv - target_logit
    if mask.any():
        return loss[mask].mean()
    return loss.mean()
