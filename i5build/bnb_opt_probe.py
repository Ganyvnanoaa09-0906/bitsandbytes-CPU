"""Does the CPU 8-bit optimizer actually work here, and what would 4-bit add?

bitsandbytes made its name on 8-bit optimizers, and this fork already ports
kOptimizerStatic8bit to CPU (BNB_CPU_TRAINING.md). Before building a 4-bit one, three
things need to be true and none of them is established by documentation:

  1. the 8-bit optimizer runs on CPU in this install
  2. it converges like fp32 AdamW on a real objective
  3. the memory it actually saves matches the arithmetic

The arithmetic, per parameter, for AdamW:
    fp32 weights + fp32 grads + fp32 (m, v)              = 16 B
    bf16 weights + bf16 grads + fp32 (m, v)              = 12 B   (mixed precision)
    bf16 weights + bf16 grads + int8 (m, v)              =  6 B   (8-bit optimizer)
    bf16 weights + bf16 grads + int4 (m, v)              =  5 B   (4-bit optimizer)

Against the MIXED-PRECISION baseline that people actually train with, 8-bit halves the
footprint and 4-bit saves only a further 17%. Against a full fp32 run the first step is
4x. So the question is not whether 4-bit is better -- it is -- but whether it is worth
building before the 8-bit path is measured.
"""
import gc
import os
import time

import torch


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return float("nan")


def main():
    print("torch", torch.__version__, "threads", torch.get_num_threads())
    try:
        import bitsandbytes as bnb
        print("bitsandbytes", bnb.__version__, "at", bnb.__file__)
    except Exception as exc:
        print("bitsandbytes import failed:", exc)
        return

    import inspect
    from bitsandbytes.optim import AdamW8bit, AdamW32bit
    print("AdamW8bit :", AdamW8bit)
    print("AdamW32bit:", AdamW32bit)
    src = inspect.getsource(AdamW8bit.__init__)
    print("--- AdamW8bit.__init__ ---")
    print(src)

    # ---- does it run at all -------------------------------------------------------
    # A parameter must exceed min_8bit_size (4096) or bitsandbytes deliberately keeps
    # its state in 32-bit. The first version of this probe used a 64x32 = 2048-element
    # parameter, so it measured the fallback and "found" that the 8-bit optimizer saves
    # nothing -- an artefact of the test, not a property of the optimizer.
    print("\n=== smoke test on CPU ===")
    torch.manual_seed(0)
    N = 512 * 512                                   # 262144 >> 4096
    w = torch.nn.Parameter(torch.randn(N) * 0.1)
    ref = torch.nn.Parameter(w.detach().clone())
    try:
        opt8 = AdamW8bit([w], lr=1e-3)
        print(f"  constructed AdamW8bit on CPU over {N} params: ok")
    except Exception as exc:
        print(f"  construct FAILED: {type(exc).__name__}: {exc}")
        return
    opt32 = torch.optim.AdamW([ref], lr=1e-3)

    x = torch.randn(256, 128)
    y = torch.randn(256, N // 128)
    for i in range(30):
        for opt, p in ((opt8, w), (opt32, ref)):
            opt.zero_grad()
            loss = ((x @ p.reshape(N // 128, 128).t() - y) ** 2).mean()
            loss.backward()
            opt.step()
        if i in (0, 14, 29):
            print(f"  step {i+1:3d}  loss8={loss.item():.6f}  "
                  f"|w8-w32|max={(w.detach()-ref.detach()).abs().max().item():.3e}")
    print(f"  final param delta vs fp32 AdamW: "
          f"{(w.detach()-ref.detach()).abs().max().item():.3e}")

    # ---- optimizer state size ------------------------------------------------------
    print("\n=== optimizer state size (the whole point) ===")
    for name, opt, p in (("fp32 AdamW", opt32, ref), ("8-bit AdamW", opt8, w)):
        tot = 0
        detail = []
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    tot += v.numel() * v.element_size()
                    detail.append(f"{k}:{tuple(v.shape)}/{v.dtype}")
        n = p.numel()
        print(f"  {name:<12} {tot:9d} B for {n} params -> {tot/n:6.3f} B/param")
        if detail:
            print(f"      {', '.join(detail[:6])}")

    # ---- what a real training footprint looks like ---------------------------------
    print("\n=== projected footprint per parameter (bf16 weights+grads) ===")
    rows = [("fp32 AdamW (all fp32)", 4 + 4 + 8),
            ("bf16 + fp32 Adam", 2 + 2 + 8),
            ("bf16 + 8-bit Adam  <- this fork", 2 + 2 + 2),
            ("bf16 + 4-bit Adam  <- the proposal", 2 + 2 + 1)]
    base = None
    for name, bpp in rows:
        if name.startswith("bf16 + fp32"):
            base = bpp
        print(f"  {name:<34} {bpp:2d} B/param   "
              f"{'' if base is None else f'{base/bpp:.2f}x vs bf16+fp32'}")

    print("\n  for SD1.5's UNet (859.5 M params):")
    for name, bpp in rows:
        print(f"    {name:<34} {859.5e6*bpp/2**30:6.2f} GiB")
    print(f"\n  this machine has {torch.get_num_threads()} threads; RAM check below")
    import subprocess
    try:
        out = subprocess.run(["wmic", "OS", "get", "TotalVisibleMemorySize,"
                              "FreePhysicalMemory", "/format:list"],
                             capture_output=True, text=True, timeout=20).stdout
        print("  " + "  ".join(l.strip() for l in out.splitlines() if l.strip()))
    except Exception:
        pass


if __name__ == "__main__":
    main()
