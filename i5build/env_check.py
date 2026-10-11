import sys
print("python", sys.version.split()[0])
print("exe   ", sys.executable)
for m in ("torch", "diffusers", "transformers", "safetensors", "numpy", "scipy"):
    try:
        mod = __import__(m)
        print("  %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("  %-14s MISSING (%s)" % (m, type(e).__name__))
try:
    import torch
    print("threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
except Exception:
    pass
try:
    import bitsandbytes as bnb
    print("bitsandbytes", bnb.__version__, bnb.__file__)
except Exception as e:
    print("bitsandbytes MISSING", type(e).__name__)
