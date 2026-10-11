"""What is on this machine, and can it run a 4-bit Qwen-Image-2.1 render?

Run on the i5 through ssh. Printed as plain text so the answer survives the trip back
through a shell that mangles quotes.

The question this exists to answer: the R5-4500U laptop renders 1024x1024 at 328 s/step
with 6 cores / 6 threads, and the i5-10400 has 6 cores / 12 threads with the same
instruction set (AVX2+FMA, no AVX-512). If the second machine does not land near 1.5-1.8x
faster, the bottleneck is not arithmetic, and that is worth knowing before optimising
anything else.
"""
import ctypes
import os
import platform
import shutil
import struct
import sys


def mem_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def main():
    print("=" * 68)
    print("machine")
    print("=" * 68)
    print(f"  host      : {platform.node()}")
    print(f"  os        : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  python    : {sys.version.split()[0]}  {sys.executable}")
    print(f"  cpu       : {cpu_name()}")
    print(f"  cores     : {os.cpu_count()} logical")
    tot, avail = mem_gb()
    print(f"  RAM       : {tot:.1f} GB total, {avail:.1f} GB available")
    for drive in ("C:", "D:", "E:"):
        try:
            u = shutil.disk_usage(drive + "\\")
            print(f"  {drive} free  : {u.free/2**30:6.1f} GB of {u.total/2**30:6.1f} GB")
        except Exception:
            pass

    print()
    print("=" * 68)
    print("packages")
    print("=" * 68)
    for m in ("torch", "diffusers", "transformers", "safetensors", "numpy", "psutil"):
        try:
            mod = __import__(m)
            print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            print(f"  {m:<14} MISSING ({type(exc).__name__})")

    print()
    print("=" * 68)
    print("bitsandbytes")
    print("=" * 68)
    try:
        import bitsandbytes as bnb
        print(f"  version   : {bnb.__version__}")
        print(f"  path      : {bnb.__file__}")
        here = os.path.dirname(os.path.abspath(bnb.__file__))
        for fn in sorted(os.listdir(here)):
            if fn.startswith("libbitsandbytes"):
                p = os.path.join(here, fn)
                print(f"  library   : {fn}  {os.path.getsize(p)/2**20:.2f} MB")
                with open(p, "rb") as fh:
                    blob = fh.read()
                for sym in (b"omp_get_max_threads", b"VCOMP140",
                            b"cgemv_4bit_inference_cpu_fp32",
                            b"cgemm_4bit_inference_cpu_fp32",
                            b"optimizer_update_8bit_blockwise_cpu"):
                    print(f"      {'yes' if sym in blob else 'NO ':>3}  "
                          f"{sym.decode('latin1')}")
        try:
            from bitsandbytes.nn import Linear4bit  # noqa: F401
            print("  Linear4bit: importable")
        except Exception as exc:
            print(f"  Linear4bit: FAILED {type(exc).__name__}: {str(exc)[:60]}")
    except Exception as exc:
        print(f"  IMPORT FAILED: {type(exc).__name__}: {str(exc)[:120]}")

    print()
    print("=" * 68)
    print("what a 4-bit Qwen-Image-2.1 render needs")
    print("=" * 68)
    need = {"4-bit checkpoint": r"D:\work\qwen_env\Qwen-Image-2.1-4bit",
            "vae + configs": r"D:\work\qwen_env\Qwen-Image-2.1"}
    for label, path in need.items():
        if os.path.isdir(path):
            n = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(path) for f in fs)
            print(f"  {label:<18} present, {n/2**30:.2f} GB at {path}")
        else:
            print(f"  {label:<18} ABSENT  ({path})")
    print(f"  needs about 9.0 GB resident; this box has {avail:.1f} GB available")


if __name__ == "__main__":
    main()
