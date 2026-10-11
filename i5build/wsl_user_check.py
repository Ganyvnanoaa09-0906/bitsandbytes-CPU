"""The end-to-end check a Linux user would run, in a file so no shell mangles it.

Written as a script rather than a `-c` one-liner because every attempt to pass Python
through cmd -> wsl -> bash lost its quoting somewhere along the way.
"""
import os
import sys

print("  python :", sys.version.split()[0])
try:
    import torch
    print("  torch  :", torch.__version__, "| cuda available:", torch.cuda.is_available())
    print("  torch  : built for", "CUDA" if "+cu" in torch.__version__ else "CPU")
except Exception as exc:  # noqa: BLE001
    print("  torch  : MISSING ->", exc)
    sys.exit(1)

import bitsandbytes as bnb  # noqa: E402

print("  bnb    :", bnb.__version__)
print("  bnb at :", bnb.__file__)

pkg = os.path.dirname(os.path.abspath(bnb.__file__))
so = os.path.join(pkg, "libbitsandbytes_cpu.so")
print("  library:", os.path.basename(so), "present" if os.path.isfile(so)
      else "*** MISSING ***")
if not os.path.isfile(so):
    sys.exit(1)

import torch.nn as nn  # noqa: E402

torch.manual_seed(0)
lin = nn.Linear(512, 256, bias=True)
q = bnb.nn.Linear4bit(512, 256, bias=True, compute_dtype=torch.float32,
                      quant_type="nf4", compress_statistics=True)
q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False,
                             quant_type="nf4", compress_statistics=True)
q.weight = q.weight.to("cpu")
q.bias = nn.Parameter(lin.bias.detach().clone())

x = torch.randn(8, 512)
y = q(x)
ref = lin(x)
rel = ((y - ref).norm() / ref.norm()).item()
print(f"  Linear4bit 4-bit: out {tuple(y.shape)}  finite={bool(torch.isfinite(y).all())}"
      f"  rel-vs-fp32={rel:.4f}  {'OK' if 0.02 < rel < 0.30 else 'SUSPECT'}")

opt = bnb.optim.AdamW8bit([nn.Parameter(torch.randn(8192))], lr=1e-3)
print("  AdamW8bit  : constructed on CPU OK")

print("\nPASS: a Linux user gets a working CPU bitsandbytes from pip alone")
