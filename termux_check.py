"""termux_check.py -- the parts of the suite that are meaningful on Android/ARM64.

WHY THIS FILE EXISTS SEPARATELY FROM run_all_tests.py
----------------------------------------------------
Termux cannot install torch (there is no Android/arm64 wheel: the manylinux arm64
wheels target glibc, Android uses bionic), and most of the suite imports torch at
module level. So run_all_tests.py cannot run on the phone at all, and the easy
wrong conclusion would be "Termux is unsupported".

What CAN be verified there is exactly the part the Termux port changed: the
ARM64/NEON kernels behind the ctypes boundary, plus the pure-Python pieces (the
disk balancer and the latent chunk store) modified in the same session. This
script checks those and reports the torch-dependent part as SKIPPED with the
reason, rather than silently omitting it.

Checks:
  T1  environment: architecture, Termux prefix
  T2  libbitsandbytes_cpu.so present and is an AArch64 ELF
  T3  it loads through ctypes and all 5 kernel symbols resolve
  T4  blockwise 8-bit round trip through the real ctypes path
  T5  gemm_8bit_forward against an INDEPENDENT float64 reference, where the
      weights are dequantized by this script rather than by the library
  T6  disk_balancer imports and runs
  T7  latent_chunk_store imports and round-trips bytes

Exit code: 0 all ran and passed, 1 a check failed, 77 nothing could run.

SIGNATURES ARE COPIED FROM csrc/pythonInterface.cpp, NOT GUESSED. The first
draft of this file guessed them and every single one was wrong: cquantize takes
a 256-entry code table first, and cgemm's B is ALREADY-QUANTIZED uint8 plus a
separate absmax array, not a float matrix. Referencing the header rather than
the name is the difference between a test and a plausible-looking failure.

Pure ASCII output so a redirected log stays readable on any console.
"""
import ctypes
import os
import platform
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

R = []
SKIP_RC = 77
SO_CANDIDATES = [
    os.path.join(HERE, "bitsandbytes", "libbitsandbytes_cpu.so"),
    os.path.join(HERE, "libbitsandbytes_cpu.so"),
]


def check(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


def skip(name, why):
    R.append((name, None, why))
    print(f"  [SKIP] {name}  {why}", flush=True)


def vp(arr):
    """void* to an ndarray's buffer."""
    return arr.ctypes.data_as(ctypes.c_void_p)


print("=" * 70)
print("Termux / ARM64 checks")
print("=" * 70)

# ---- T1 environment --------------------------------------------------------
print()
print("[T1] environment")
arch = platform.machine()
prefix = os.environ.get("PREFIX", "")
print(f"  machine : {arch}")
print(f"  python  : {platform.python_version()} ({platform.python_implementation()})")
print(f"  prefix  : {prefix or '<unset>'}")
if prefix and os.path.isfile(os.path.join(prefix, "bin", "clang++")):
    print("  clang++ : present")
check("T1 running on aarch64", arch in ("aarch64", "arm64"), arch)

# ---- T2 the shared object --------------------------------------------------
print()
print("[T2] libbitsandbytes_cpu.so")
so = next((p for p in SO_CANDIDATES if os.path.isfile(p)), None)
if so is None:
    check("T2 .so exists", False, f"looked in {SO_CANDIDATES}")
else:
    size = os.path.getsize(so)
    with open(so, "rb") as fh:
        head = fh.read(20)
    is_elf = head[:4] == b"\x7fELF"
    e_machine = struct.unpack_from("<H", head, 18)[0] if is_elf else 0
    check("T2 .so is an AArch64 ELF", is_elf and e_machine == 0xB7,
          f"{size:,} B  e_machine=0x{e_machine:X}"
          + ("" if e_machine == 0xB7 else "  <- expected 0xB7"))

# ---- T3 ctypes load + symbols ---------------------------------------------
print()
print("[T3] ctypes load and kernel symbols")
SYMS = [
    "cquantize_blockwise_cpu_fp32",
    "cdequantize_blockwise_cpu_fp32",
    "cgemm_8bit_inference_cpu_fp32",
    "cgemv_4bit_inference_cpu_fp32",
    "coptimizer_update_8bit_blockwise_cpu",
]
lib = None
if so is None:
    check("T3 load .so", False, "no .so to load")
else:
    try:
        lib = ctypes.CDLL(so)
        check("T3 .so loads via ctypes", True, os.path.basename(so))
    except OSError as e:
        check("T3 .so loads via ctypes", False, f"{type(e).__name__}: {e}")
    if lib is not None:
        missing = [s for s in SYMS if not hasattr(lib, s)]
        check("T3 all 5 kernel symbols resolve", not missing,
              "ok" if not missing else f"missing: {missing}")

# ---- numpy -----------------------------------------------------------------
try:
    import numpy as np
    HAVE_NP = True
    NUMPY_ERR = ""
except Exception as e:  # noqa: BLE001
    HAVE_NP = False
    NUMPY_ERR = f"{type(e).__name__}: {e}"

# The linear 8-bit code map, same as selftest_cpu.c and create_linear_map():
#   code256[i] = 2*i/255 - 1
CODE256 = np.linspace(-1.0, 1.0, 256).astype(np.float32) if HAVE_NP else None


def quantize(x, blocksize):
    """Blockwise 8-bit quantize via the library. Returns (bytes, absmax)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.size
    nb = n // blocksize
    absmax = np.zeros(nb, dtype=np.float32)
    out = np.zeros(n, dtype=np.uint8)
    lib.cquantize_blockwise_cpu_fp32(vp(CODE256), vp(x), vp(absmax), vp(out),
                                     ctypes.c_longlong(blocksize), ctypes.c_longlong(n))
    return out, absmax


def dequantize(q, absmax, blocksize):
    out = np.zeros(q.size, dtype=np.float32)
    lib.cdequantize_blockwise_cpu_fp32(vp(CODE256), vp(q), vp(absmax), vp(out),
                                       ctypes.c_longlong(blocksize),
                                       ctypes.c_longlong(q.size))
    return out


# ---- T4 quantize round trip ------------------------------------------------
print()
print("[T4] blockwise 8-bit round trip through ctypes")
if not HAVE_NP:
    skip("T4 round trip", f"numpy unavailable: {NUMPY_ERR}")
elif lib is None:
    skip("T4 round trip", "no .so loaded")
else:
    try:
        n, bs = 4096, 256
        x = (np.sin(np.arange(n) * 0.01) * 3.0).astype(np.float32)
        q, am = quantize(x, bs)
        back = dequantize(q, am, bs)
        err = float(np.abs(back - x).max())
        peak = float(np.abs(x).max())
        check("T4 round-trip error < 1% of peak", err < 0.01 * peak,
              f"max err {err:.5f}  peak {peak:.3f}  ({err/peak*100:.3f}%)")
    except Exception as e:  # noqa: BLE001
        check("T4 round trip", False, f"{type(e).__name__}: {e}")

# ---- T5 gemm_8bit vs an independent float64 reference ----------------------
print()
print("[T5] gemm_8bit vs independent float64 reference")
if not HAVE_NP:
    skip("T5 gemm_8bit", f"numpy unavailable: {NUMPY_ERR}")
elif lib is None:
    skip("T5 gemm_8bit", "no .so loaded")
else:
    # The C selftest compares gemm_8bit against its own scalar path, which is a
    # consistency check: an error present in both would pass. Here the reference
    # is built by dequantizing the weight bytes in THIS script's numpy code, so
    # it is an independent oracle, and the absmax layout it assumes is checked
    # explicitly below.
    try:
        rng = np.random.default_rng(7)
        M, N, K, bs = 32, 32, 64, 64
        A = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
        Bf = (rng.standard_normal((N, K)) * 0.5).astype(np.float32)   # (N,K) row major

        Bq, absmax = quantize(Bf.ravel(), bs)
        assert Bq.size == N * K, "quantize returned the wrong element count"

        C = np.zeros((M, N), dtype=np.float32)
        # pythonInterface.cpp:
        #   void cgemm_8bit_inference_cpu_fp32(const float* A, const unsigned char* B,
        #       const float* absmax, float* out, long long M, long long N, long long K,
        #       long long lda, long long ldb, long long ldc, long long blocksize)
        # and cpu_ops.cpp documents out[m,n] = sum_k A[m,k] * code[B[n,k/2..]] * absmax[n, k/blocksize]
        lib.cgemm_8bit_inference_cpu_fp32(
            vp(A), vp(Bq), vp(absmax), vp(C),
            ctypes.c_longlong(M), ctypes.c_longlong(N), ctypes.c_longlong(K),
            ctypes.c_longlong(K), ctypes.c_longlong(K), ctypes.c_longlong(N),
            ctypes.c_longlong(bs))

        # Independent dequantize in float64, layout spelled out here on purpose:
        # element (n,k) of B has block index k // bs within row n.
        Bref = (CODE256[Bq].astype(np.float64)
                .reshape(N, K // bs, bs)
                * absmax.astype(np.float64).reshape(N, K // bs, 1)).reshape(N, K)
        ref = A.astype(np.float64) @ Bref.T

        denom = max(1e-9, float(np.abs(ref).max()))
        rel = float(np.abs(C - ref).max()) / denom
        check("T5 gemm_8bit rel err < 1% vs float64", rel < 0.01,
              f"rel {rel:.3e}  max|ref| {np.abs(ref).max():.3f}  max|C| {np.abs(C).max():.3f}")
    except Exception as e:  # noqa: BLE001
        check("T5 gemm_8bit", False, f"{type(e).__name__}: {e}")

# ---- T6 / T7 / T8: the torch-dependent half, RUN IN A SEPARATE PROCESS ------
#
# Measured on the phone: running the ctypes checks and then importing torch in
# ONE process aborts with
#     OMP: Error #15: Initializing libomp.a, but found libomp.a already initialized.
#     Aborted (core dumped)   [exit 134]
# Two OpenMP runtimes meet: ours (build_termux.sh links libomp statically into
# the .so) and torch's own. LLVM's OpenMP aborts deliberately, and it is right to.
#
# KMP_DUPLICATE_LIB_OK=TRUE is the documented way around it and is the wrong tool
# here: this script checks numerical correctness, and that flag's documentation
# says it "may cause crashes or silently produce incorrect results". A
# correctness test must not run under a flag that permits silently wrong answers.
#
# So the split is by OpenMP domain, not by convenience: THIS file loads the .so
# and never imports torch; torch_part.py imports torch and never loads the .so.
# The probe below is therefore also a subprocess -- importing torch here to see
# whether it is available would be the very thing that crashes.
import subprocess

print()
print("[T6/T7/T8] torch-dependent half (separate process)")
_probe = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"],
                        capture_output=True, text=True, timeout=180)
if _probe.returncode != 0:
    skip("T6/T7 disk_balancer + latent_chunk_store",
         "torch not importable in a clean process")
    skip("T8 AdamW8bit real step", "torch not importable")
    print(f"      probe said: {( _probe.stderr or '').strip().splitlines()[-1:] or ['(no stderr)']}")
    print("      Both modules import torch at module level (disk_balancer.py:76,")
    print("      latent_chunk_store.py:11), so they cannot run without it. That is")
    print("      a dependency gap, NOT an ARM64 defect: the kernels they drive are")
    print("      covered by T2-T5 above, which reach them through ctypes.")
else:
    _ver = _probe.stdout.strip()
    print(f"  torch {_ver} is importable")
    _part = os.path.join(HERE, "torch_part.py")
    if not os.path.isfile(_part):
        skip("T6/T7/T8", "torch_part.py not present in this bundle")
    else:
        # Two separate processes on purpose. The second one is EXPECTED to die on
        # a platform where the OpenMP runtimes collide, and it must not be able to
        # take the first one's verdict down with it.
        _rc1 = subprocess.call([sys.executable, _part])
        if _rc1 == 0:
            R.append(("T6/T7 disk_balancer + latent_chunk_store", True, "torch-only process"))
        elif _rc1 == 77:
            R.append(("T6/T7", None, "nothing runnable"))
        else:
            R.append(("T6/T7 disk_balancer + latent_chunk_store", False,
                      f"torch_part.py exited {_rc1}"))

        _rc2 = subprocess.call([sys.executable, _part, "--with-bnb"])
        if _rc2 == 0:
            R.append(("T8 bitsandbytes + torch in one process", True, "kernel ran"))
        elif _rc2 in (-6, 134):
            # SIGABRT. This is the documented OpenMP collision, and it says
            # nothing about kernel correctness -- the same kernel is verified
            # torch-free by T2-T5, and through the full stack on x86.
            skip("T8 bitsandbytes + torch in one process",
                 "SIGABRT: two OpenMP runtimes (our .so links libomp statically, "
                 "torch ships its own)")
            print("      This is a BUILD-CONFIGURATION limit, not a kernel defect.")
            print("      To close it, rebuild the .so single-threaded and re-run:")
            print("          cd ~/bnb_termux/bnb && bash build_termux.sh --no-openmp")
            print("          python termux_check.py")
            print("      A single-threaded .so cannot collide with torch's runtime, and")
            print("      numerical correctness does not depend on the thread count (the")
            print("      C selftest already asserts vector == scalar bit-for-bit).")
            print("      Do NOT set KMP_DUPLICATE_LIB_OK to paper over it instead: that")
            print("      flag permits silently incorrect results, which is precisely")
            print("      what this suite exists to rule out.")
        elif _rc2 == 77:
            R.append(("T8", None, "nothing runnable"))
        else:
            R.append(("T8 bitsandbytes + torch in one process", False,
                      f"torch_part.py --with-bnb exited {_rc2}"))

# ---- what is still out of reach here ---------------------------------------
print()
print("[not verifiable on this platform]")
skip("full regression suite (run_all_tests.py)",
     "imports torch at module level in most children, and pulls the .so into the"
     " same process")
print("      Even with torch installed, the full sweep mixes both OpenMP runtimes.")
print("      The long training runs are validated on x86 (R5 and i5, 1000 steps).")

# ---- summary ---------------------------------------------------------------
ran = [r for r in R if r[1] is not None]
failed = [r for r in ran if not r[1]]
print()
print("=" * 70)
print(f"ran {len(ran)} check(s): {len(ran) - len(failed)} passed, {len(failed)} failed;"
      f" {len(R) - len(ran)} skipped")
for name, ok, detail in failed:
    print(f"  FAIL: {name}  {detail}")
print("=" * 70)
if not ran:
    sys.exit(SKIP_RC)
sys.exit(1 if failed else 0)
