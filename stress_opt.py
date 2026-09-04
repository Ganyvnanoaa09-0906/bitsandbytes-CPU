"""Stress the fused 8-bit optimizer ctypes path: all optimizers, dtypes, sizes."""
import torch
import bitsandbytes as bnb

torch.manual_seed(0)


def make_opt(name, params, lr=1e-3):
    ctor = {
        "adamw8bit": bnb.optim.AdamW8bit,
        "adam8bit": bnb.optim.Adam8bit,
        "sgd8bit": bnb.optim.SGD8bit,
        "lion8bit": bnb.optim.Lion8bit,
        "adagrad8bit": bnb.optim.Adagrad8bit,
        "rmsprop8bit": bnb.optim.RMSprop8bit,
        "ademamix8bit": bnb.optim.AdEMAMix8bit,
    }[name]
    if name == "sgd8bit":
        return ctor(params, lr=lr, momentum=0.9)
    return ctor(params, lr=lr)


def run(name, n, dtype, steps=3):
    p = torch.nn.Parameter(torch.randn(n).to(dtype))
    opt = make_opt(name, [p])
    for _ in range(steps):
        opt.zero_grad()
        (p.float().square().sum() * 0.01).backward()
        opt.step()
    return p.detach().float()


fails = 0
sizes = [1, 7, 255, 256, 257, 1000, 4096, 16384, 65536, 262145]
names = ["adamw8bit", "adam8bit", "sgd8bit", "lion8bit", "adagrad8bit", "rmsprop8bit", "ademamix8bit"]
for name in names:
    for n in sizes:
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            try:
                out = run(name, n, dtype)
                assert torch.isfinite(out).all(), f"{name} n={n} {dtype}: non-finite output"
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name} n={n} {dtype}: {type(e).__name__}: {e}")
print(f"done, {fails} failures")

# determinism: identical params/grads -> identical results (catches races)
a = torch.nn.Parameter(torch.randn(4096))
b = torch.nn.Parameter(a.data.clone())
oa, ob = bnb.optim.AdamW8bit([a], lr=1e-3), bnb.optim.AdamW8bit([b], lr=1e-3)
for _ in range(5):
    a.grad, b.grad = torch.randn(4096), a.grad if a.grad is not None else None
    a.grad = torch.randn(4096)
    b.grad = a.grad.clone()
    oa.step(); ob.step()
d = (a - b).abs().max().item()
print(f"determinism diff: {d:.2e}")
assert d == 0.0, "non-deterministic update"

# non-contiguous grad falls back to composite and must match contiguous update
g_val = torch.randn(64, 64)
buf = torch.zeros(64, 128)
buf[:, :64] = g_val
nc_grad = buf[:, :64]  # same values, strides (128,1) -> non-contiguous
assert not nc_grad.is_contiguous()
pa = torch.nn.Parameter(torch.zeros(64, 64))
pb = torch.nn.Parameter(torch.zeros(64, 64))
oa = bnb.optim.AdamW8bit([pa], lr=1e-2)
ob = bnb.optim.AdamW8bit([pb], lr=1e-2)
for _ in range(3):
    oa.zero_grad(); ob.zero_grad()
    pa.grad = g_val.clone()
    pb.grad = buf[:, :64]  # genuinely strided view, same values
    assert not pb.grad.is_contiguous()
    oa.step(); ob.step()
d = (pa - pb).abs().max().item()
print(f"non-contig diff: {d:.2e}")
assert d < 1e-6

print("ALL STRESS PASSED")
