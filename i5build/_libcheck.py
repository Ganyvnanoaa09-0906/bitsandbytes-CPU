import importlib
import sys

print(sys.executable)
print(sys.version.split()[0])
print()
for m in ("huggingface_hub", "modelscope", "hf_transfer", "requests", "tqdm", "safetensors", "torch"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:18s} {getattr(mod, '__version__', 'installed')}")
    except Exception:
        print(f"  {m:18s} MISSING")
