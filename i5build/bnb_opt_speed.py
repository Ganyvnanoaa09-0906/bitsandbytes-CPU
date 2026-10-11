"""How fast is the CPU 8-bit optimizer, and what does the state actually cost?

Measured, not assumed:
  * state bytes per parameter for fp32 AdamW and 8-bit AdamW at a realistic size
  * wall time per optimizer step for both, since the fused CPU kernel does
    dequant -> update -> write-back -> requant in one pass and should therefore be
    moving 4x fewer bytes
  * the same for a LoRA-sized parameter set, because that is where the trainable
    parameter count is small and the optimizer state stops being the dominant term
"""
import time

import torch
from bitsandbytes.optim import AdamW8bit


def step_time(opt, p, shape, reps=30):
    x = torch.randn(128, shape[1])
    y = torch.randn(128, shape[0])
    for _ in range(3):
        opt.zero_grad()
        ((x @ p.t() - y) ** 2).mean().backward()
        opt.step()
    t0 = time.perf_counter()
    for _ in range(reps):
        opt.zero_grad()
        ((x @ p.t() - y) ** 2).mean().backward()
        opt.step()
    return (time.perf_counter() - t0) / reps


def state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


def bench(n_params, label):
    rows = {}
    for name, make in (("fp32 AdamW", lambda p: torch.optim.AdamW([p], lr=1e-3)),
                       ("8-bit AdamW", lambda p: AdamW8bit([p], lr=1e-3))):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(n_params) * 0.05)
        opt = make(p)
        # warm the state
        opt.zero_grad()
        (p.sum() * 0.0 + (p ** 2).mean()).backward()
        opt.step()
        sb = state_bytes(opt)
        t = step_time(opt, p.reshape(n_params // 128, 128), (n_params // 128, 128))
        rows[name] = (sb, sb / n_params, t)
        del opt, p
    print(f"\n=== {label} ({n_params/1e6:.2f} M params) ===")
    print(f"  {'optimizer':<14} {'state':>12} {'B/param':>9} {'ms/step':>9}")
    for k, (sb, bpp, t) in rows.items():
        print(f"  {k:<14} {sb/2**20:9.2f} MiB {bpp:9.3f} {t*1e3:9.2f}")
    f, e = rows["fp32 AdamW"], rows["8-bit AdamW"]
    print(f"  -> state {f[0]/e[0]:.2f}x smaller, step {f[2]/e[2]:.2f}x faster")


if __name__ == "__main__":
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    bench(1 << 20, "1 MiB of parameters (typical layer group)")
    bench(8 << 20, "8 M parameters (LoRA-sized trainable set, rank 32 on SD1.5)")
    bench(64 << 20, "64 M parameters (full fine-tune of a small block)")
    print("\nprojection for SD1.5 UNet full fine-tune, 859.5 M params:")
    for name, bpp in (("fp32 AdamW", 8.0), ("8-bit AdamW", 2.04), ("4-bit (proposed)", 1.04)):
        print(f"  {name:<18} {859.5e6*bpp/2**30:6.2f} GiB of optimizer state")
