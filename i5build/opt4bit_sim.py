"""Would a 4-bit optimizer state survive? Simulation first, kernel later.

The 8-bit optimizer stores each state as a uint8 code plus one fp32 absmax per 256
weights: 2.03 B/param for the two Adam states, measured at 2.28 with the tables. Packing
two 4-bit codes into a byte would take that to ~1.13 B/param, another 2x -- and since what
actually kills long runs on this machine is system RAM, not compute (a 1500-step LoRA run
died at step 1450 with 2.49 GB free, §11.2), that is a real lever rather than a vanity one.

The obstacle is not the packing, it is the dynamic range. `exp_avg_sq` is a running mean
of squares and spans several orders of magnitude across a tensor; 16 levels with one
absmax per block would quantise it to mush. The dynamic map the 8-bit path uses is a
7-bit exponent plus a sign precisely for that reason.

Two properties make a 4-bit state worth testing anyway:

  * `exp_avg_sq` is NON-NEGATIVE, so all 16 levels can be exponent -- a log-domain map
    covers 16 octaves inside a block instead of 15 linear steps;
  * `exp_avg` is roughly symmetric and near-Gaussian, which is what the nf4 codebook was
    designed for.

This script measures the accuracy of those choices by round-tripping the state through
bnb's own quantisers at every step -- the same arithmetic a fused kernel would do -- so
that a kernel only gets written if the simulation says the loss survives. Kernel speed is
a separate question and is NOT measured here.

Usage:
    python opt4bit_sim.py
    python opt4bit_sim.py --steps 600 --blocksize 64
"""
import argparse
import time

import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes import functional as F

torch.manual_seed(0)

# 16-level codebooks, built the way the real thing would build them.
NF4 = F.create_normal_map()                       # 16 levels, Gaussian-matched, signed


def log_map(levels=16, lo=1e-6, hi=1.0):
    """Non-negative log-spaced codebook: 16 octaves of range for a quantity that spans
    many. Returned ascending in [0, 1] so it composes with a block absmax."""
    import math
    vals = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (levels - 1))
            for i in range(levels)]
    return torch.tensor(vals, dtype=torch.float32)


def lin_map(levels=16, signed=True):
    if signed:
        return torch.linspace(-1, 1, levels)
    return torch.linspace(0, 1, levels)


def quant_roundtrip(x, code, blocksize):
    """What the kernel would do: blockwise absmax, nearest code, back to fp32.

    The all-zero case has to be short-circuited: the first step round-trips a state that
    is still zeros, absmax comes out 0, and the dequantiser divides by it. Every arm of
    the first run returned NaN for exactly this reason, which reads as "4-bit destroys
    training" when it is really "the simulator divided by zero".
    """
    if float(x.abs().max()) == 0.0:
        return x.clone()
    q, state = F.quantize_blockwise(x.contiguous().view(-1), code=code,
                                    blocksize=blocksize)
    return F.dequantize_blockwise(q, state).view_as(x)


class SimAdamW(torch.optim.Optimizer):
    """AdamW whose two states are stored at 4 bits, emulated by round-tripping them
    through a codebook after every step. The state stays fp32 in this process; what is
    being measured is the error that storage would introduce, not the memory saved."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, blocksize=64, code1=NF4, code2=None,
                 bits2=4):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay))
        self.blocksize = blocksize
        self.code1 = code1
        self.code2 = code2 if code2 is not None else log_map()
        self.bits2 = bits2

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                m, v = st["exp_avg"], st["exp_avg_sq"]
                # what the kernel reads back: the states as storage would hold them
                m_q = quant_roundtrip(m, self.code1, self.blocksize) if m.numel() else m
                v_q = quant_roundtrip(v, self.code2, self.blocksize) if v.numel() else v
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** st["step"]
                bc2 = 1 - beta2 ** st["step"]
                denom = (v_q.sqrt() / bc2 ** 0.5).add_(group["eps"])
                p.addcdiv_(m_q, denom, value=-group["lr"] / bc1)
                if group["weight_decay"]:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])


def train(factory, steps, seed=123, width=64):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(width, 256), nn.GELU(),
                          nn.Linear(256, 256), nn.GELU(),
                          nn.Linear(256, 1))
    opt = factory(model.parameters())
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(1024, width, generator=g)
    w = torch.randn(width, generator=g)
    y = x @ w + 0.3
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def bytes_per_param(opt, projected=None):
    """State bytes per parameter.

    For the simulated arms the states still live in fp32 in this process, so the measured
    number is meaningless -- what matters is what the packed format would cost, which is
    4 bits per code plus one fp32 absmax per block, per state.
    """
    if projected is not None:
        return projected
    total_b = total_n = 0
    for p, st in opt.state.items():
        n = p.numel()
        total_n += n
        if not st:
            continue
        for k, v in st.items():
            if k == "step" or not torch.is_tensor(v):
                continue
            itemsize = v.element_size()
            total_b += v.numel() * itemsize
    return total_b / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--blocksize", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    a = ap.parse_args()

    arms = [
        ("fp32 AdamW", lambda ps: torch.optim.AdamW(ps, lr=1e-3),
         torch.optim.AdamW),
        ("bnb 8-bit", lambda ps: bnb.optim.AdamW8bit(ps, lr=1e-3),
         bnb.optim.AdamW8bit),
        ("sim m4 nf4 / v4 log", lambda ps: SimAdamW(ps, lr=1e-3,
                                                    blocksize=a.blocksize),
         SimAdamW),
        ("sim m4 nf4 / v4 lin", lambda ps: SimAdamW(ps, lr=1e-3,
                                                    blocksize=a.blocksize,
                                                    code2=lin_map(16, signed=False)),
         SimAdamW),
        ("sim m4 nf4 / v4 nf4", lambda ps: SimAdamW(ps, lr=1e-3,
                                                    blocksize=a.blocksize,
                                                    code2=NF4),
         SimAdamW),
    ]

    print(f"steps {a.steps}  blocksize {a.blocksize}  width {a.width}")
    print(f"{'arm':<22} {'final loss':>11} {'gap vs fp32':>12} {'B/param':>9} "
          f"{'time s':>7}")
    ref_final = None
    # what a packed 4-bit state costs: two states, 4 bits each, one fp32 absmax per block
    proj4 = 2 * (0.5 + 4.0 / a.blocksize)
    for name, factory, cls in arms:
        t0 = time.perf_counter()
        losses = train(factory, a.steps, width=a.width)
        dt = time.perf_counter() - t0
        if ref_final is None:
            ref_final = losses[-1]
        # measure state bytes with a one-step run of the same factory
        opt = factory(nn.Linear(a.width, a.width).parameters())
        p = next(iter(opt.param_groups))["params"][0]
        p.grad = torch.randn_like(p)
        opt.step()
        bpp = bytes_per_param(opt, projected=proj4 if cls is SimAdamW else None)
        print(f"{name:<22} {losses[-1]:>11.4f} {losses[-1]-ref_final:>12.4f} "
              f"{bpp:>9.3f} {dt:>7.1f}")


if __name__ == "__main__":
    main()
