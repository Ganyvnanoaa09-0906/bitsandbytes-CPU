"""lowmem_guard.py -- what does the quantized model actually cost at runtime?

The target is not this 15.4 GB machine. It is an ordinary CPU-only desktop, which in
practice means 8 GB or less, so "does it fit" has to be answered for a budget far
below what this box has. This measures the real figures instead of estimating them.

Two costs that are easy to conflate:

  * the quantized weights themselves, which is what the checkpoint size shows
  * dequantize_4bit, which allocates `torch.empty(shape, dtype=...)` -- a FULL
    precision tensor -- every time a 4-bit weight is used. Per layer that is
    temporary, but a forward pass over 232 layers at a large batch can hold many of
    them at once, and that is the number that decides whether a small machine can
    run the model at all.

So this walks up: load the checkpoint, load one layer, run a forward at increasing
token counts, and report the process working set and peak at each step. A step that
exceeds the budget is reported rather than allowed to swap the machine to death.

Usage:
    python lowmem_guard.py --ckpt <4bit dir> --component transformer [--budget-gb 4]
"""
import argparse
import ctypes
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bitsandbytes"))


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def mem():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize, c.PeakWorkingSetSize


def avail():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys


def gb(n):
    return f"{n/1e9:.2f} GB"


def step(label, budget):
    ws, peak = mem()
    free = avail()
    flag = ""
    if peak > budget:
        flag = "  <-- OVER BUDGET"
    print(f"  {label:38s} ws {gb(ws):>9s}  peak {gb(peak):>9s}  "
          f"free {gb(free):>9s}{flag}")
    return ws, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\work\qwen_env\Qwen-Image-2.1-4bit")
    ap.add_argument("--component", default="transformer")
    ap.add_argument("--budget-gb", type=float, default=4.0)
    ap.add_argument("--tokens", default="256,1024,4096")
    a = ap.parse_args()
    budget = int(a.budget_gb * 1e9)

    import qwen21_load as L
    import bitsandbytes as bnb

    print(f"component {a.component}, budget {gb(budget)}")
    print(f"machine has {gb(avail())} free right now")
    print()
    print("baseline (interpreter + torch + bitsandbytes):")
    step("imported everything", budget)
    print()

    print("step 1: read the checkpoint into tensors")
    t0 = time.time()
    tensors, meta = L.load_checkpoint(a.ckpt, a.component)
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"  checkpoint tensors: {len(tensors)}, {gb(nbytes)} of tensor data "
          f"({time.time()-t0:.1f}s)")
    step("checkpoint resident", budget)
    print()

    print("step 2: dequantize the largest single weight, which is what a layer "
          "does when it runs")
    biggest = max(meta["layers"], key=lambda k: (
        meta["layers"][k]["shape"][0] * meta["layers"][k]["shape"][1]))
    info = meta["layers"][biggest]
    qs = bnb.functional.QuantState.from_dict(
        {k[len(biggest) + 1:]: v for k, v in tensors.items()
         if k.startswith(biggest + ".")},
        device=torch.device("cpu"))
    before = mem()[0]
    t0 = time.time()
    dq = bnb.functional.dequantize_4bit(tensors[biggest], qs)
    dt = time.time() - t0
    after = mem()[0]
    print(f"  {biggest}  shape {info['shape']}")
    print(f"  packed {gb(tensors[biggest].numel())} -> dequantized {gb(dq.numel()*dq.element_size())} "
          f"in {dt*1e3:.1f} ms")
    print(f"  working set grew by {gb(after-before)}")
    step("one layer dequantized", budget)
    del dq
    print()

    print("step 3: build the model on the meta device and load the weights")
    with L.meta_device_context():
        model = (L.build_transformer(a.ckpt) if a.component == "transformer"
                 else L.build_text_encoder(a.ckpt))
    step("architecture built (meta)", budget)
    L.replace_quantized(model, meta, tensors, verbose=False)
    step("quantized modules replaced", budget)
    L.materialize_rest(model, meta, tensors)
    step("model fully materialized", budget)
    model.eval()
    print()

    if a.component != "transformer":
        print("(forward pass skipped for this component)")
        return 0

    print("step 4: forward pass at increasing sizes. This is where dequantize_4bit")
    print("        allocates a full fp32 tensor per layer, so it is the real test.")
    cfg = json.load(open(os.path.join(a.ckpt, "transformer", "config.json"),
                         encoding="utf-8"))
    for tok in [int(x) for x in a.tokens.split(",")]:
        hidden = torch.randn(1, tok, cfg["in_channels"], dtype=torch.float32)
        enc = torch.randn(1, 64, cfg["context_in_dim"], dtype=torch.float32)
        img_shapes = [[(1, max(1, tok // 4), 4)]]
        img_mask = torch.cat([torch.zeros(1, 64, dtype=torch.bool),
                              torch.ones(1, tok // 4, dtype=torch.bool)], dim=1)
        ts = torch.tensor([0.5], dtype=torch.float32)
        # keep the mask consistent with the token count the model derives
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model(hidden_states=hidden, encoder_hidden_states=enc,
                            timestep=ts, img_shapes=img_shapes, img_mask=img_mask,
                            return_dict=False)[0]
            dt = time.time() - t0
            print(f"  tokens={tok:6d}  ran in {dt:6.1f}s  out {tuple(out.shape)}  "
                  f"finite={bool(torch.isfinite(out).all())}")
            step(f"after forward at {tok} tokens", budget)
            del out
        except Exception as e:  # noqa: BLE001
            print(f"  tokens={tok}: FAILED {type(e).__name__}: {str(e)[:130]}")
            step(f"after failed forward at {tok}", budget)
        del hidden, enc, img_mask
    print()
    print("reading:")
    print("  the number that matters for a small machine is the PEAK column minus")
    print("  the baseline. If the peak is dominated by dequantize_4bit allocating a")
    print("  full fp32 tensor per layer, then shrinking the checkpoint does not")
    print("  shrink the runtime, and that is the thing to fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
