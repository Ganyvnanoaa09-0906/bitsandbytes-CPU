"""End-to-end validation of the CPU training path (fused 8-bit optimizer + GDN).

Run: python -m tests.test_cpu_e2e   (from the bitsandbytes repo root)
"""
import time

import torch

torch.manual_seed(0)


def train(optimizer_factory, steps=150, seed=123):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 1),
    )
    opt = optimizer_factory(model.parameters())
    x = torch.randn(1024, 64)
    w = torch.randn(64)
    y = x @ w + 0.3
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model, opt


def main() -> int:
    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib

    fused_available = hasattr(lib, "coptimizer_update_8bit_blockwise_cpu")
    print(f"native lib: {type(lib).__name__}  fused kernel exported: {fused_available}")
    assert fused_available, "fused 8-bit optimizer entry point missing from native lib"

    t0 = time.perf_counter()
    ref_losses, _, _ = train(lambda p: torch.optim.AdamW(p, lr=1e-3))
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    bnb_losses, model, opt = train(lambda p: bnb.optim.AdamW8bit(p, lr=1e-3), steps=150)
    t_bnb = time.perf_counter() - t0

    print(f"loss  step0   torch {ref_losses[0]:.4f}  bnb8bit {bnb_losses[0]:.4f}")
    print(f"loss  final   torch {ref_losses[-1]:.4f} bnb8bit {bnb_losses[-1]:.4f}")
    print(f"time  torch {t_ref*1000:.0f} ms   bnb8bit {t_bnb*1000:.0f} ms")
    assert bnb_losses[-1] < bnb_losses[0] * 0.2, "8-bit AdamW is not converging"
    gap = abs(bnb_losses[-1] - ref_losses[-1])
    print(f"final-loss gap vs fp32 AdamW: {gap:.4f}")
    assert gap < 0.05, f"8-bit path diverged from reference (gap {gap:.4f})"

    # state bytes: uint8 codes + fp32 absmax per 256-block, x2 states.
    # tensors below min_8bit_size keep fp32 states (no absmax) - count those too.
    bytes_8bit = sum(
        s["state1"].numel() + s["absmax1"].numel() * 4 +
        (s["state2"].numel() + s["absmax2"].numel() * 4 if "state2" in s else 0)
        if "absmax1" in s
        else s["state1"].numel() * 4 + (s["state2"].numel() * 4 if "state2" in s else 0)
        for s in opt.state.values() if "state1" in s
    )
    n_param = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"8-bit state: {bytes_8bit/1e6:.1f} MB  vs fp32 Adam states {n_param*8/1e6:.1f} MB "
          f"({n_param*8/max(bytes_8bit, 1):.1f}x smaller)")

    # non-contiguous grad must not corrupt states (falls back to composite path)
    torch.manual_seed(7)
    g_val = torch.randn(64, 64)
    buf = torch.zeros(64, 128)
    buf[:, :64] = g_val
    pa = torch.nn.Parameter(torch.zeros(64, 64))
    pb = torch.nn.Parameter(torch.zeros(64, 64))
    oa = bnb.optim.AdamW8bit([pa], lr=1e-2)
    ob = bnb.optim.AdamW8bit([pb], lr=1e-2)
    for _ in range(3):
        oa.zero_grad(); ob.zero_grad()
        pa.grad = g_val.clone()
        pb.grad = buf[:, :64]  # same values, strides (128,1) -> non-contiguous
        assert not pb.grad.is_contiguous()
        oa.step(); ob.step()
    diff = (pa - pb).abs().max().item()
    print(f"non-contiguous grad update diff: {diff:.2e}")
    assert diff < 1e-6, "non-contiguous grad corrupted the update"

    print("ALL E2E PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
