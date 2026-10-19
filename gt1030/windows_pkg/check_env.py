# -*- coding: utf-8 -*-
"""check_env.py — iGPU training environment check (Windows first, also works on Linux).

Verifies, in order:
  1. Python version / 64-bit;
  2. torch / transformers / pyopencl installed, versions;
  3. CLBlast dll found and loadable (CLBLAST_PATH or next to this file / cwd);
  4. OpenCL platforms and devices (iGPU), with memory sizes;
  5. optional --bench: measure iGPU memory bandwidth via pyopencl (copy 256MB there and back).

Exit code 0 = READY for `train.bat`, 1 = something missing (prints the fix).
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def line(msg=""):
    print(msg, flush=True)


def check_python():
    impl = platform.python_implementation()
    bits = platform.architecture()[0]
    line("Python        : %s %s (%s, %s)" % (sys.version.split()[0], impl, bits, sys.platform))
    if sys.version_info < (3, 10):
        line("  !! Python 3.10+ is required. Install 64-bit Python 3.10-3.12 from python.org")
        return False
    if bits != "64bit" or sys.maxsize <= 2**32:
        line("  !! 64-bit Python is required (this is 32-bit)")
        return False
    return True


def check_modules():
    ok = True
    for name in ("torch", "transformers", "numpy", "pyopencl"):
        try:
            mod = __import__(name)
            line("%-13s: %s" % (name, getattr(mod, "__version__", "installed")))
        except Exception as exc:
            line("%-13s: MISSING (%s)" % (name, exc))
            ok = False
    try:
        import torch
        line("torch device  : %s (threads=%d)" % (torch.device("cpu"), torch.get_num_threads()))
    except Exception:
        pass
    return ok


def find_clblast():
    candidates = []
    env = os.environ.get("CLBLAST_PATH")
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(HERE, "clblast.dll"),
        os.path.join(os.getcwd(), "clblast.dll"),
        os.path.join(HERE, "clblast.so"),
        "libclblast.so",
    ]
    for path in candidates:
        if not path:
            continue
        if os.path.sep in path or path.endswith((".dll", ".so")):
            if not os.path.isfile(path):
                continue
        try:
            lib = ctypes.CDLL(path)
            lib.CLBlastSgemm  # must exist
            line("CLBlast dll   : %s" % os.path.abspath(path))
            return path
        except Exception as exc:
            line("CLBlast dll   : found %r but failed to load: %s" % (path, exc))
            return None
    line("CLBlast dll   : NOT FOUND")
    line("  fix: keep clblast.dll next to this file, or set CLBLAST_PATH=full\\path\\to\\clblast.dll")
    return None


def check_opencl():
    try:
        import pyopencl as cl
    except Exception as exc:
        line("OpenCL        : pyopencl missing (%s)" % exc)
        return None
    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        line("OpenCL        : no platforms (%s). Install/update the GPU driver." % exc)
        return None
    if not platforms:
        line("OpenCL        : no OpenCL platform. Install/update the GPU driver "
             "(AMD Adrenalin / Intel Arc or Iris driver).")
        return None
    chosen = None
    for p in platforms:
        line("OpenCL platform: %s (%s)" % (p.name.strip(), p.vendor.strip()))
        for d in p.get_devices():
            gmem = d.global_mem_size // (1 << 20)
            line("  device       : %s | type=%s | %d MB global" % (d.name.strip(), d.type, gmem))
            if chosen is None:
                chosen = d
    return chosen


def bench_igpu(device):
    """256 MB device-to-device copy bandwidth as a minimal 'the iGPU works' test."""
    try:
        import pyopencl as cl
        import numpy as np
    except Exception:
        return
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    nbytes = 256 << 20
    a = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nbytes)
    b = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, nbytes)
    host = np.zeros(nbytes // 4, dtype=np.float32)
    cl.enqueue_copy(queue, a, host).wait()
    cl.enqueue_copy(queue, b, host).wait()
    cl.enqueue_copy(queue, a, host).wait()
    t0 = time.perf_counter()
    for _ in range(3):
        cl.enqueue_copy(queue, b, a).wait()
    dt = (time.perf_counter() - t0) / 3
    line("iGPU copy     : %.0f MB in %.1f ms -> %.1f GB/s (d2d)" %
         (nbytes / (1 << 20), dt * 1e3, (2 * nbytes) / dt / 1e9))


def main() -> int:
    line("=" * 64)
    line("CPU-forge iGPU training environment check")
    line("=" * 64)
    ok = check_python()
    ok = check_modules() and ok
    clblast = find_clblast()
    device = check_opencl()
    if device is not None and "--bench" in sys.argv:
        bench_igpu(device)
    line("")
    if ok and clblast and device is not None:
        line("READY: run  train.bat --model <local dir or HF id> --steps 20")
        return 0
    if ok and device is None:
        line("NO iGPU OpenCL device. You can still train on CPU: "
             "train.bat --backend cpu --model <model> --steps 20")
    line("NOT READY: fix the items marked above, then re-run doctor.bat")
    return 1


if __name__ == "__main__":
    sys.exit(main())
