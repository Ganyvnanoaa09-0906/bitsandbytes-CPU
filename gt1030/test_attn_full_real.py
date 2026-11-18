# -*- coding: utf-8 -*-
"""test_attn_full_real.py — real SD1.5 decoder mid-attention weights vs torch SDPA.

    python gt1030/test_attn_full_real.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

import gt1030.igpu_ctx as CTX
import gt1030.igpu_fx as FX
from gt1030.igpu_attn_full import FullAttn

VAE = r"D:\work\textmodel\_sd15_combined\vae"



def block_check():
    """Whole diffusers Attention block: group_norm + qkv + out + residual."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE)
    vae.eval()
    a = vae.decoder.mid_block.attentions[0]
    C = a.to_q.in_features
    x = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        yref = a(x)
    fm = FX.FullAttnModule(C)
    with torch.no_grad():
        fm.q_proj.weight.copy_(a.to_q.weight)
        fm.q_proj.bias.copy_(a.to_q.bias)
        fm.k_proj.weight.copy_(a.to_k.weight)
        fm.k_proj.bias.copy_(a.to_k.bias)
        fm.v_proj.weight.copy_(a.to_v.weight)
        fm.v_proj.bias.copy_(a.to_v.bias)
        fm.o_proj.weight.copy_(a.to_out[0].weight)
        fm.o_proj.bias.copy_(a.to_out[0].bias)
        h = a.group_norm(x)
        y = x + fm(h) if getattr(a, "residual_connection", False) else fm(h)
    rel = float((y - yref).norm() / yref.norm())
    print("  whole block rel = %.2e" % rel)
    return rel < 2e-2


def main() -> int:
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(VAE)
    vae.eval()
    attn = vae.decoder.mid_block.attentions[0]
    C = attn.to_q.in_features
    print("  real decoder mid attention: C=%d heads=%s scale=%s" %
          (C, attn.heads, getattr(attn, "scale", None)))
    print("  q_bias=%s k_bias=%s v_bias=%s o_bias=%s group_norm=%s" %
          (attn.to_q.bias is not None, attn.to_k.bias is not None,
           attn.to_v.bias is not None, attn.to_out[0].bias is not None,
           getattr(attn, "group_norm", None) is not None))

    dev = CTX.DevCtx(arena_mb=900, name="attnreal")
    dev.activate()
    fa = FullAttn(dev, 2, C, 8, 8)
    fa.set_weights(
        attn.to_q.weight.detach().numpy(),
        attn.to_k.weight.detach().numpy(),
        attn.to_v.weight.detach().numpy(),
        attn.to_out[0].weight.detach().numpy(),
        attn.to_q.bias.detach().numpy(),
        attn.to_k.bias.detach().numpy(),
        attn.to_v.bias.detach().numpy(),
        attn.to_out[0].bias.detach().numpy(),
    )

    x = torch.randn(2, C, 8, 8)
    xd = dev.up(np.ascontiguousarray(x.numpy(), np.float16))
    y = fa.forward(xd)
    yv = dev.down(y, (2, C, 8, 8)).astype(np.float32)

    with torch.no_grad():
        xt = x.flatten(2).transpose(1, 2)  # (N,T,C)
        q = attn.to_q(xt)
        k = attn.to_k(xt)
        v = attn.to_v(xt)
        o = F.scaled_dot_product_attention(q, k, v, scale=1.0 / np.sqrt(C))
        o = attn.to_out[0](o).transpose(1, 2).reshape(2, C, 8, 8).numpy()

    rel = np.linalg.norm(yv - o) / np.linalg.norm(o)
    print("  forward rel = %.2e" % rel)
    ok = rel < 2e-2
    ok &= block_check()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
