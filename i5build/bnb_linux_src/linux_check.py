"""Does the Linux .so load, and does it export what the Python side asks for?

Run inside WSL after build_linux.sh. Deliberately does not import torch: the point is to
test the C library in isolation, and a missing torch in the WSL distribution would other-
wise be reported as a failure of the build.

The symbol list is taken from what bitsandbytes/backends/cpu/ops.py and
bitsandbytes/functional.py look up through ctypes, so a missing entry here is a real
breakage rather than a cosmetic one.
"""
import ctypes
import os
import sys

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "bitsandbytes", "libbitsandbytes_cpu.so")

REQUIRED = [
    "cdequantize_blockwise_cpu_fp32",
    "cdequantize_blockwise_cpu_bf16",
    "cquantize_blockwise_cpu_fp32",
    "cgemv_4bit_inference_cpu_fp32",
    "cgemv_4bit_inference_cpu_bf16",
    "cgemv_4bit_inference_cpu_fp16",
    "cgemm_8bit_inference_cpu_fp32",
    "coptimizer_update_8bit_blockwise_cpu",
]

# Compiled in only when the build targets AVX512-BF16, so its absence on an AVX2-only
# machine is correct rather than a breakage: functional.has_avx512bf16() catches
# AttributeError and returns False. Requiring it made this check report a failure on the
# first library that built cleanly.
OPTIONAL = ["has_avx512bf16_cpu"]


def main():
    if not os.path.isfile(SO):
        print(f"FAILED: {SO} does not exist")
        return 1
    print(f"  {SO}  ({os.path.getsize(SO)/1e6:.2f} MB)")
    try:
        lib = ctypes.CDLL(SO)
    except OSError as exc:
        print(f"FAILED to load: {exc}")
        return 1
    print("  CDLL loaded")
    missing = []
    for name in REQUIRED:
        ok = hasattr(lib, name)
        print(f"    {'yes' if ok else 'NO ':>3}  {name}")
        if not ok:
            missing.append(name)
    for name in OPTIONAL:
        ok = hasattr(lib, name)
        print(f"    {'yes' if ok else 'no ':>3}  {name}   (optional: AVX512-BF16 builds only)")
    if missing:
        print(f"FAILED: {len(missing)} symbol(s) missing: {missing}")
        return 1
    print("PASS: the Linux library loads and exports every symbol the Python side needs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
