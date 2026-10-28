# -*- coding: utf-8 -*-
"""test_cpu_caps_selection.py — regression test for the AVX-512/BF16 safety fix.

The shipped AVX-512 library contains AVX512-BF16 instructions (auto-vectorised), so the
loader must require the avx512_bf16 CPU flag, not just avx512f/bw/dq/vl.
Run: python test_cpu_caps_selection.py
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

from bitsandbytes import _cpu_caps  # noqa: E402


def _write_cpuinfo(flags: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix="_cpuinfo", delete=False, encoding="utf-8")
    f.write("processor\t: 0\n")
    f.write("vendor_id\t: GenuineIntel\n")
    f.write("flags\t\t: %s\n" % flags)
    f.write("processor\t: 1\n")
    f.write("flags\t\t: %s\n" % flags)
    f.close()
    return f.name


def test_flags_without_bf16():
    path = _write_cpuinfo("avx2 fma f16c avx512f avx512bw avx512dq avx512vl")
    try:
        assert _cpu_caps._read_cpuinfo_flags(path) is not None
        assert not _cpu_caps._proc_cpuinfo_supports_avx512(path), \
            "AVX-512 without avx512_bf16 must NOT select the AVX-512 artifact"
        assert _cpu_caps._proc_cpuinfo_supports_avx2(path)
    finally:
        os.unlink(path)


def test_flags_with_bf16():
    path = _write_cpuinfo(
        "avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512_bf16 avx512vbmi")
    try:
        assert _cpu_caps._proc_cpuinfo_supports_avx512(path), \
            "AVX-512 + avx512_bf16 should select the AVX-512 artifact"
    finally:
        os.unlink(path)


def test_missing_cpuinfo_is_conservative():
    assert _cpu_caps._read_cpuinfo_flags("/nonexistent/cpuinfo") is None
    assert not _cpu_caps._proc_cpuinfo_supports_avx512("/nonexistent/cpuinfo")
    assert not _cpu_caps._proc_cpuinfo_supports_avx2("/nonexistent/cpuinfo")


def test_env_overrides():
    saved = {k: os.environ.get(k) for k in
             ("BNB_CPU_DISABLE_AVX512", "BNB_CPU_FORCE_AVX512", "BNB_CPU_FORCE_GENERIC")}
    try:
        os.environ["BNB_CPU_DISABLE_AVX512"] = "1"
        assert not _cpu_caps.avx512_supported()
        os.environ.pop("BNB_CPU_DISABLE_AVX512")
        os.environ["BNB_CPU_FORCE_AVX512"] = "1"
        assert _cpu_caps.avx512_supported()
        os.environ.pop("BNB_CPU_FORCE_AVX512")
        os.environ["BNB_CPU_FORCE_GENERIC"] = "1"
        assert not _cpu_caps.avx2_supported()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    test_flags_without_bf16()
    test_flags_with_bf16()
    test_missing_cpuinfo_is_conservative()
    test_env_overrides()
    print("CPU CAPS SELECTION TESTS PASSED (avx512_bf16 required; env overrides work)")


if __name__ == "__main__":
    main()
