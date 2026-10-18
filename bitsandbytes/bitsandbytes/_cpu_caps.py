"""CPU feature queries used to choose between the AVX2 and AVX-512 CPU libraries.

Why a separate module: two places need the same answer -- `cextension` (the main native
library loader) and `gdn_cpu` (the fused Gated DeltaNet wrapper). Importing one from the
other would drag the whole native library in at import time, so the query lives here and
imports nothing but the standard library; torch is imported lazily, only when it is already
present in the process.

The AVX-512 backend is a separate artifact because it is NOT safe on AVX2-only CPUs:
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

_AVX512_FLAGS = frozenset({"avx512f", "avx512bw", "avx512dq", "avx512vl"})


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


def _proc_cpuinfo_supports_avx512() -> bool:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return _AVX512_FLAGS <= flags
    except OSError:
        return False
    return False


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
    capability = _torch_capability()
    if capability is not None:
        return capability == "AVX512"
    return _proc_cpuinfo_supports_avx512()
