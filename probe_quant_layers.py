"""probe_quant_layers.py -- what does bnb's quantized-layer API support WITHOUT transformers?

The Termux 50-step run stopped at `No module named 'transformers'`, and
transformers cannot be pip-installed on Android: its dependencies tokenizers and
safetensors are Rust extensions and regex/pyyaml are C extensions, and every
aarch64 wheel on PyPI is manylinux (glibc) while Android is bionic. So before
deciding whether to chase that dependency, this asks the cheaper question: how
much of the training path can be exercised with torch alone?

Reported per class:
  * does it exist in this build
  * does a forward pass work on CPU
  * does BACKWARD work (training needs it; inference-only layers are useless here)
  * how many parameters, and the dtype/shape of the output

Run on x86 first, then on the phone, and compare. Anything that works on x86 but
not on ARM64 is a portability finding; anything that fails on both is a design
boundary, not a Termux problem.
"""
import sys
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, ".")
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes from {bnb.__file__}")
except Exception as e:  # noqa: BLE001
    print(f"FATAL: cannot import bitsandbytes: {type(e).__name__}: {e}")
    raise SystemExit(1)

print(f"torch {torch.__version__} on {torch.get_num_threads()} thread(s)")
print()

CANDIDATES = [
    ("nn.Linear8bitLt", lambda: bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)),
    ("nn.Linear4bit", lambda: bnb.nn.Linear4bit(64, 64, quant_type="nf4")),
]

for name, ctor in CANDIDATES:
    print("=" * 68)
    print(name)
    print("=" * 68)
    obj = getattr(bnb.nn, name.split(".")[-1], None)
    if obj is None:
        print("  not present in this build")
        print()
        continue
    try:
        layer = ctor()
    except Exception as e:  # noqa: BLE001
        print(f"  construct FAILED: {type(e).__name__}: {e}")
        print()
        continue

    n = sum(p.numel() for p in layer.parameters())
    print(f"  constructed, {n:,} params, class {type(layer).__name__}")

    x = torch.randn(4, 64, requires_grad=True)
    try:
        y = layer(x)
        print(f"  forward  OK: out {tuple(y.shape)} {y.dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"  forward  FAILED: {type(e).__name__}: {e}")
        print()
        continue

    try:
        y.sum().backward()
        gx = x.grad
        gw = [p.grad for p in layer.parameters() if p.grad is not None]
        finite = bool(torch.isfinite(gx).all()) and all(bool(torch.isfinite(g).all()) for g in gw)
        print(f"  backward OK: x.grad finite={finite}, "
              f"{len(gw)}/{len(list(layer.parameters()))} param grads present")
        print(f"             |x.grad|max={float(gx.abs().max()):.4e}")
        if gw:
            print(f"             |w.grad|max={max(float(g.abs().max()) for g in gw):.4e}")
        else:
            print("             NO parameter gradients -- cannot train this layer")
    except Exception as e:  # noqa: BLE001
        print(f"  backward FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
    print()

print("=" * 68)
print("plain torch control (must work everywhere)")
print("=" * 68)
m = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
y = m(torch.randn(4, 64)).sum()
y.backward()
print("  nn.Linear forward+backward OK")

print()
print("=" * 68)
print("optimizer availability")
print("=" * 68)
for oname in ("AdamW8bit", "Adam8bit", "SGD8bit", "Lion8bit", "RMSprop8bit",
              "Adagrad8bit", "AdEMAMix8bit"):
    print(f"  {oname:14s} {'yes' if hasattr(bnb.optim, oname) else 'MISSING'}")

print()
print("transformers availability (informational):")
try:
    import transformers  # noqa: F401
    print(f"  present {transformers.__version__}")
except Exception as e:  # noqa: BLE001
    print(f"  {type(e).__name__}: {e}")
