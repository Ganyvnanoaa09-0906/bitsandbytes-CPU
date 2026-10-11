"""What a Linux user actually sees today, step by step, with the published package.

The point of running all three steps rather than just the import: `import bitsandbytes`
SUCCEEDS on the published sdist even though there is no Linux library in the wheel,
because the native library is loaded lazily. So every signal a user checks first says
the install worked, and the failure only appears when the package is used.
"""
import torch
import torch.nn as nn

print("STEP 1: pip install bitsandbytes-cpu-fork          -> succeeded")
print("STEP 2: import bitsandbytes")
import bitsandbytes as bnb  # noqa: E402

print(f"          -> succeeded, version {bnb.__version__}")

print("STEP 3: actually use it")
try:
    q = bnb.nn.Linear4bit(512, 256, bias=True, compute_dtype=torch.float32,
                          quant_type="nf4", compress_statistics=True)
    lin = nn.Linear(512, 256, bias=True)
    q.weight = bnb.nn.Params4bit(lin.weight.data, requires_grad=False,
                                 quant_type="nf4", compress_statistics=True)
    q.weight = q.weight.to("cpu")
    y = q(torch.randn(8, 512))
    print("          -> succeeded?!", tuple(y.shape))
except Exception as exc:  # noqa: BLE001
    print(f"          -> FAILED: {type(exc).__name__}")
    msg = str(exc).strip().splitlines()
    for line in msg[:6]:
        print("             " + line)
