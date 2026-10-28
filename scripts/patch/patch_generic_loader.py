"""Three-way library selection: AVX-512 -> AVX2 -> generic x86-64.

The generic artifact (libbitsandbytes_cpu_generic.so, built with --generic) is the portable
fallback for CPUs without AVX2. Selection order in the loader:

  CPU supports AVX-512 and the avx512 artifact exists -> avx512 artifact
  CPU supports AVX2                                   -> regular artifact
  otherwise, generic artifact exists                  -> generic artifact
  otherwise                                           -> regular artifact (fails loudly, as before)

BNB_CPU_FORCE_GENERIC=1 forces the generic path (so the selection logic is testable on any CPU);
BNB_CPU_DISABLE_AVX512=1 keeps AVX2 as before. `doctor` prints the CPU's AVX2 status, whether the
generic library is present, and which file this process will load.

Touches _cpu_caps.py, cextension.py, gdn_cpu.py, bitsandbytes_cpu_cli.py, build_release_linux.sh.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
PKG = REPO + r'\bitsandbytes'
CAPS = os.path.join(PKG, 'bitsandbytes', '_cpu_caps.py')
CEXT = os.path.join(PKG, 'bitsandbytes', 'cextension.py')
GDN = os.path.join(PKG, 'bitsandbytes', 'gdn_cpu.py')
CLI = os.path.join(REPO, 'bitsandbytes_cpu_cli.py')
REL = os.path.join(PKG, 'build_release_linux.sh')

CAPS_ANCHOR = (
    'def _proc_cpuinfo_supports_avx512() -> bool:\n'
)
CAPS_NEW = (
    'def _proc_cpuinfo_supports_avx2() -> bool:\n'
    '    try:\n'
    '        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:\n'
    '            for line in handle:\n'
    '                if line.startswith("flags"):\n'
    '                    flags = set(line.split(":", 1)[1].split())\n'
    '                    return {"avx2", "fma", "f16c"} <= flags\n'
    '    except OSError:\n'
    '        return False\n'
    '    return False\n'
    '\n'
    '\n'
    'def _proc_cpuinfo_supports_avx512() -> bool:\n'
)

CAPS_TAIL_ANCHOR = (
    '    capability = _torch_capability()\n'
    '    if capability is not None:\n'
    '        return capability == "AVX512"\n'
    '    return _proc_cpuinfo_supports_avx512()\n'
)
CAPS_TAIL_NEW = CAPS_TAIL_ANCHOR + (
    '\n'
    '\n'
    'def avx2_supported() -> bool:\n'
    '    """True when the CPU can execute the regular (AVX2+FMA) artifact.\n'
    '\n'
    '    BNB_CPU_FORCE_GENERIC=1 forces the portable artifact (it runs on any x86-64, so this is\n'
    '    safe and makes the three-way selection testable on an AVX2 machine).\n'
    '    """\n'
    '    if _env_flag("BNB_CPU_FORCE_GENERIC"):\n'
    '        return False\n'
    '    capability = _torch_capability()\n'
    '    if capability is not None:\n'
    '        return capability in ("AVX2", "AVX512")\n'
    '    return _proc_cpuinfo_supports_avx2()\n'
)

CEXT_IMPORT_OLD = 'from bitsandbytes._cpu_caps import avx512_supported\n'
CEXT_IMPORT_NEW = 'from bitsandbytes._cpu_caps import avx2_supported, avx512_supported\n'

CEXT_ANCHOR = (
    '    if avx512_supported():\n'
    '        avx512_binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu_avx512{DYNAMIC_LIBRARY_SUFFIX}"\n'
    '        if avx512_binary_path.exists():\n'
    '            binary_path = avx512_binary_path\n'
)
CEXT_NEW = CEXT_ANCHOR + (
    '\n'
    '    # Generic x86-64 artifact: only when the CPU cannot run the AVX2 library AND the generic\n'
    '    # one is present. It contains no AVX2/FMA code (build_linux.sh --generic asserts that), so\n'
    '    # loading it is safe on any x86-64 CPU.\n'
    '    if not avx2_supported():\n'
    '        generic_binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu_generic{DYNAMIC_LIBRARY_SUFFIX}"\n'
    '        if generic_binary_path.exists():\n'
    '            binary_path = generic_binary_path\n'
)

GDN_OLD = (
    '    try:\n'
    '        from ._cpu_caps import avx512_supported\n'
    '\n'
    '        if avx512_supported():\n'
    '            dll_names.insert(0, "libbitsandbytes_cpu_avx512.dll")\n'
    '            so_names.insert(0, "libbitsandbytes_cpu_avx512.so")\n'
    '    except Exception:  # pragma: no cover - the choice is an optimisation, not a requirement\n'
    '        pass\n'
)
GDN_NEW = (
    '    try:\n'
    '        from ._cpu_caps import avx2_supported, avx512_supported\n'
    '\n'
    '        if avx512_supported():\n'
    '            dll_names.insert(0, "libbitsandbytes_cpu_avx512.dll")\n'
    '            so_names.insert(0, "libbitsandbytes_cpu_avx512.so")\n'
    '        elif not avx2_supported():\n'
    '            dll_names.insert(0, "libbitsandbytes_cpu_generic.dll")\n'
    '            so_names.insert(0, "libbitsandbytes_cpu_generic.so")\n'
    '    except Exception:  # pragma: no cover - the choice is an optimisation, not a requirement\n'
    '        pass\n'
)

CLI_OLD = (
    '    try:\n'
    '        caps = _load_from_package("_cpu_caps")\n'
    '        avx512_ok = bool(caps.avx512_supported())\n'
    '    except Exception:  # pragma: no cover - doctor must still work if the helper is absent\n'
    '        avx512_ok = False\n'
    '    avx512 = os.path.join(pkg, f"libbitsandbytes_cpu_avx512{suffix}")\n'
    '    has_avx512 = os.path.isfile(avx512)\n'
    '    print(f"  cpu supports AVX-512  {\'yes\' if avx512_ok else \'no\'}")\n'
    '    print(f"  AVX-512 library       {os.path.basename(avx512)} "\n'
    '          f"{\'(present)\' if has_avx512 else \'(absent)\'}")\n'
    '    if avx512_ok and has_avx512:\n'
    '        target = avx512\n'
    '    print(f"  this process will load  {os.path.basename(target)}")\n'
)
CLI_NEW = (
    '    try:\n'
    '        caps = _load_from_package("_cpu_caps")\n'
    '        avx512_ok = bool(caps.avx512_supported())\n'
    '        avx2_ok = bool(caps.avx2_supported())\n'
    '    except Exception:  # pragma: no cover - doctor must still work if the helper is absent\n'
    '        avx512_ok = avx2_ok = False\n'
    '    avx512 = os.path.join(pkg, f"libbitsandbytes_cpu_avx512{suffix}")\n'
    '    generic = os.path.join(pkg, f"libbitsandbytes_cpu_generic{suffix}")\n'
    '    has_avx512 = os.path.isfile(avx512)\n'
    '    has_generic = os.path.isfile(generic)\n'
    '    print(f"  cpu supports AVX-512  {\'yes\' if avx512_ok else \'no\'}")\n'
    '    print(f"  cpu supports AVX2     {\'yes\' if avx2_ok else \'no\'}")\n'
    '    print(f"  AVX-512 library       {os.path.basename(avx512)} "\n'
    '          f"{\'(present)\' if has_avx512 else \'(absent)\'}")\n'
    '    print(f"  generic library       {os.path.basename(generic)} "\n'
    '          f"{\'(present)\' if has_generic else \'(absent)\'}")\n'
    '    if avx512_ok and has_avx512:\n'
    '        target = avx512\n'
    '    elif not avx2_ok and has_generic:\n'
    '        target = generic\n'
    '    print(f"  this process will load  {os.path.basename(target)}")\n'
)

REL_OLD = (
    '# 可选：AVX-512 变体（build_linux.sh --avx512 产出）；只在支持 AVX-512 的机器上被加载\n'
    'if [ -f bitsandbytes/libbitsandbytes_cpu_avx512.so ]; then\n'
    '  cp bitsandbytes/libbitsandbytes_cpu_avx512.so "$STAGE/bitsandbytes/"\n'
    'fi\n'
)
REL_NEW = REL_OLD + (
    '# 可选：通用 x86-64 变体（build_linux.sh --generic 产出）；只在没有 AVX2 的机器上被加载\n'
    'if [ -f bitsandbytes/libbitsandbytes_cpu_generic.so ]; then\n'
    '  cp bitsandbytes/libbitsandbytes_cpu_generic.so "$STAGE/bitsandbytes/"\n'
    'fi\n'
)


EDITS = [
    (CAPS, CAPS_ANCHOR, CAPS_NEW, 'caps: avx2 query helper'),
    (CAPS, CAPS_TAIL_ANCHOR, CAPS_TAIL_NEW, 'caps: avx2_supported'),
    (CEXT, CEXT_IMPORT_OLD, CEXT_IMPORT_NEW, 'cextension: import'),
    (CEXT, CEXT_ANCHOR, CEXT_NEW, 'cextension: generic fallback'),
    (GDN, GDN_OLD, GDN_NEW, 'gdn_cpu: generic preference'),
    (CLI, CLI_OLD, CLI_NEW, 'cli doctor: three-way'),
    (REL, REL_OLD, REL_NEW, 'release: ship generic .so'),
]


def main() -> int:
    for path, old, new, label in EDITS:
        text = io.open(path, encoding='utf-8').read()
        n = text.count(old)
        if n != 1:
            print('ABORT %-40s %s anchor=%d' % (os.path.basename(path), label, n))
            return 1
        io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %-40s %s' % (os.path.basename(path), label))
    return 0


if __name__ == '__main__':
    sys.exit(main())
