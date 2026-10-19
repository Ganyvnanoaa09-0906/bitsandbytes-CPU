"""CPU feature queries used to choose between the AVX2 and AVX-512 CPU libraries.

Why a separate module: two places need the same answer -- `cextension` (the main native
library loader) and `gdn_cpu` (the fused Gated DeltaNet wrapper). Importing one from the
other would drag the whole native library in at import time, so the query lives here and
imports nothing but the standard library; torch is imported lazily, only when it is already
present in the process.

The AVX-512 backend is a separate artifact because it is NOT safe on AVX2-only CPUs
(and it is compiled with AVX512-BF16, so it is also unsafe on AVX-512 CPUs without BF16):
building it with `-mavx512f` is a translation-unit-wide switch, and GCC auto-vectorises
ordinary kernels (dequantise, optimizer, rmsnorm) with zmm instructions. The library must
therefore only be loaded after the CPU and OS are known to support AVX-512. On Linux the
`/proc/cpuinfo` flags are authoritative; on Windows the query goes through torch, which
already performs the full CPUID + XGETBV check (and exists in any process that can use this
package). If neither source can answer, the answer is "no" -- refusing to load the AVX-512
library is always safe, guessing is not.
"""
from __future__ import annotations

import os

# avx512_bf16 is required too: the AVX-512 artifact is compiled with
# -mavx512bf16, so it contains bf16 instructions (auto-vectorised) even outside
# the explicitly guarded bf16 kernels. CPUs with AVX-512 but no BF16
# (Skylake-X, Cascade Lake) would SIGILL on them.
_AVX512_FLAGS = frozenset({"avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_bf16"})


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _torch_capability() -> str | None:
    """What torch already knows about this CPU ('AVX2', 'AVX512', 'DEFAULT', ...)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return None
    backend = getattr(torch, "backends", None)
    cpu = getattr(backend, "cpu", None)
    query = getattr(cpu, "get_cpu_capability", None)
    if query is None:
        return None
    try:
        return str(query()).upper()
    except Exception:  # pragma: no cover - defensive: a broken torch must not crash import
        return None


def _read_cpuinfo_flags(path: str = "/proc/cpuinfo") -> set[str] | None:
    """First ``flags`` line of /proc-style cpuinfo, or None when unreadable (non-Linux)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return None
    return None


def _proc_cpuinfo_supports_avx2(path: str = "/proc/cpuinfo") -> bool:
    flags = _read_cpuinfo_flags(path)
    return flags is not None and {"avx2", "fma", "f16c"} <= flags


def _proc_cpuinfo_supports_avx512(path: str = "/proc/cpuinfo") -> bool:
    flags = _read_cpuinfo_flags(path)
    return flags is not None and _AVX512_FLAGS <= flags


def avx512_supported() -> bool:
    """True only when the CPU + OS can execute the AVX-512 libraries' instructions.

    BNB_CPU_DISABLE_AVX512=1 forces the AVX2 library even on capable hardware (useful when
    comparing the two backends); BNB_CPU_FORCE_AVX512=1 forces the AVX-512 path (only safe on
    a machine that really supports it -- it exists so the selection logic can be tested).
    """
    if _env_flag("BNB_CPU_DISABLE_AVX512"):
        return False
    if _env_flag("BNB_CPU_FORCE_AVX512"):
        return True
    flags = _read_cpuinfo_flags()
    if flags is not None:
        # /proc/cpuinfo knows about BF16, so it is the authoritative answer on Linux.
        return _AVX512_FLAGS <= flags
    # torch reports only the ISA level ("AVX512"), not BF16. Since the shipped AVX-512
    # library contains bf16 instructions, an ISA-level "yes" alone is not safe there.
    capability = _torch_capability()
    if capability is not None:
        return False
    return False


def avx2_supported() -> bool:
    """True when the CPU can execute the regular (AVX2+FMA) artifact.

    BNB_CPU_FORCE_GENERIC=1 forces the portable artifact (it runs on any x86-64, so this is
    safe and makes the three-way selection testable on an AVX2 machine).
    """
    if _env_flag("BNB_CPU_FORCE_GENERIC"):
        return False
    capability = _torch_capability()
    if capability is not None:
        return capability in ("AVX2", "AVX512")
    return _proc_cpuinfo_supports_avx2()
