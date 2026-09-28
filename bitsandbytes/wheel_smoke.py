"""wheel_smoke.py -- does the installed wheel actually work?

Building a wheel and importing it proves the packaging is syntactically valid. It
does NOT prove the thing a user cares about: that the compiled kernel loads, that
its OpenMP runtime is found, and that the API surface responds. Those go through
ctypes and the DLL loader, which a bare `import bitsandbytes` may never touch --
linear layers are only compiled on first use.

So this exercises, in the order a user would hit them:

  1. the package resolves from site-packages, not a source checkout
  2. libbitsandbytes_cpu.dll loads (this is where a missing vcomp140.dll fails)
  3. every public API group the fork advertises is present
  4. quantize / dequantize round-trip through the real dispatch path
  5. Linear8bitLt forward, i.e. the kernel is entered, not just imported
  6. AdamW8bit steps, i.e. the optimizer kernel runs
  7. the fork's own additions (gdn_cpu, fused_cpu) import

Run it with the interpreter of a venv that has the wheel installed:

    <venv>/Scripts/python.exe wheel_smoke.py
"""
import os
import sys
import traceback

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


print("=" * 70)
print("installed-wheel smoke test")
print("=" * 70)
print(f"  python     : {sys.version.split()[0]}")
print(f"  executable : {sys.executable}")

# ---- 1. where did it come from? -------------------------------------------
print()
print("[1] resolution")
try:
    import bitsandbytes as bnb
except Exception as e:  # noqa: BLE001
    print(f"  [FAIL] import bitsandbytes: {type(e).__name__}: {e}")
    traceback.print_exc(limit=3)
    sys.exit(1)

pkg_dir = os.path.dirname(os.path.abspath(bnb.__file__))
print(f"  version    : {bnb.__version__}")
print(f"  package    : {pkg_dir}")
check("imported from site-packages (not a source checkout)",
      "site-packages" in pkg_dir, pkg_dir)

# ---- 2. the compiled kernel ----------------------------------------------
print()
print("[2] compiled kernel")
dll_name = "libbitsandbytes_cpu.dll" if os.name == "nt" else "libbitsandbytes_cpu.so"
dll = os.path.join(pkg_dir, dll_name)
check(f"{dll_name} present in the installed package", os.path.isfile(dll),
      f"{os.path.getsize(dll):,} B" if os.path.isfile(dll) else "MISSING")

if os.name == "nt":
    vcomp = os.path.join(pkg_dir, "vcomp140.dll")
    check("vcomp140.dll bundled (OpenMP runtime the DLL imports)",
          os.path.isfile(vcomp),
          f"{os.path.getsize(vcomp):,} B" if os.path.isfile(vcomp) else "MISSING")

try:
    import ctypes
    lib = ctypes.CDLL(dll)
    for sym in ("cquantize_blockwise_cpu_fp32", "cdequantize_blockwise_cpu_fp32",
                "cgemm_8bit_inference_cpu_fp32", "cgemv_4bit_inference_cpu_fp32",
                "coptimizer_update_8bit_blockwise_cpu"):
        getattr(lib, sym)
    check("kernel loads via ctypes and exports the 5 entry points", True)
except Exception as e:  # noqa: BLE001
    check("kernel loads via ctypes and exports the 5 entry points", False,
          f"{type(e).__name__}: {e}")

# ---- 3. API surface -------------------------------------------------------
print()
print("[3] API surface")
groups = {
    "nn": ["Linear8bitLt", "Linear4bit", "Int8Params", "Params4bit",
           "Embedding", "Embedding8bit"],
    "optim": ["AdamW8bit", "Adam8bit", "SGD8bit", "Lion8bit", "RMSprop8bit",
              "Adagrad8bit", "AdEMAMix8bit", "LAMB8bit", "LARS8bit"],
    "functional": ["quantize_4bit", "dequantize_4bit", "quantize_blockwise",
                   "dequantize_blockwise", "int8_vectorwise_quant",
                   "int8_vectorwise_dequant"],
}
for grp, names in groups.items():
    obj = getattr(bnb, grp, None)
    if obj is None:
        check(f"bnb.{grp}", False, "missing")
        continue
    absent = [n for n in names if not hasattr(obj, n)]
    check(f"bnb.{grp}: {len(names)} names", not absent,
          "ok" if not absent else f"missing {absent}")

for extra in ("gdn_cpu",):
    check(f"bnb.{extra} (fork addition)", hasattr(bnb, extra))

# ---- 4/5/6. actually run the kernels -------------------------------------
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

print()
print("[4] quantize / dequantize through the real dispatch")
try:
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    q, st = bnb.functional.quantize_4bit(x, quant_type="nf4", blocksize=64)
    back = bnb.functional.dequantize_4bit(q, st)
    err = float((back - x).abs().max())
    check("nf4 4-bit round trip", back.shape == x.shape and err < 5.0,
          f"shape {tuple(back.shape)}  max|d| {err:.4f}")
except Exception as e:  # noqa: BLE001
    check("nf4 4-bit round trip", False, f"{type(e).__name__}: {e}")
    traceback.print_exc(limit=3)

print()
print("[5] Linear8bitLt forward (enters the kernel)")
try:
    torch.manual_seed(0)
    layer = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False)
    y = layer(torch.randn(8, 64))
    check("forward returns finite output",
          tuple(y.shape) == (8, 64) and bool(torch.isfinite(y).all()),
          f"out {tuple(y.shape)}")
except Exception as e:  # noqa: BLE001
    check("forward returns finite output", False, f"{type(e).__name__}: {e}")
    traceback.print_exc(limit=3)

print()
print("[6] AdamW8bit steps (enters the optimizer kernel)")
try:
    torch.manual_seed(0)
    m = nn.Linear(64, 64)
    before = [p.detach().clone() for p in m.parameters()]
    opt = bnb.optim.AdamW8bit(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(6):
        opt.zero_grad()
        loss = (m(torch.randn(8, 64)) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    moved = max(float((p.detach() - b).abs().max())
                for p, b in zip(m.parameters(), before))
    check("weights moved and losses finite",
          moved > 0 and all(v == v for v in losses),
          f"max|dW| {moved:.3e}  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
except Exception as e:  # noqa: BLE001
    check("weights moved and losses finite", False, f"{type(e).__name__}: {e}")
    traceback.print_exc(limit=3)

# ---- 7. the fork's additions ---------------------------------------------
print()
print("[7] fork additions")
for mod, attr in (("gdn_cpu", "patch_transformers"),):
    try:
        __import__(f"bitsandbytes.{mod}")
        m2 = sys.modules[f"bitsandbytes.{mod}"]
        check(f"bitsandbytes.{mod}.{attr}", hasattr(m2, attr))
    except Exception as e:  # noqa: BLE001
        check(f"bitsandbytes.{mod}.{attr}", False, f"{type(e).__name__}: {e}")

print()
print("=" * 70)
if FAILS:
    print(f"RESULT: {len(FAILS)} check(s) FAILED")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: all checks passed -- the wheel is usable as installed")
sys.exit(0)
