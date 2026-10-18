"""Wire the AVX-512 CPU library in as an automatically selected second artifact.

Design: the AVX-512 library is unsafe on AVX2-only CPUs (a whole-TU -mavx512f build
auto-vectorises ordinary kernels with zmm instructions and SIGILLs there), so it is never
chosen by "does the file exist". The loader asks `_cpu_caps.avx512_supported()` -- torch's
get_cpu_capability() when available, /proc/cpuinfo otherwise -- and only then prefers
libbitsandbytes_cpu_avx512<SUFFIX>; any missing file, unknown CPU, or forced-off env var
falls back to the AVX2 library.

Changes here:
  cextension.py   import the capability query, prefer the AVX-512 artifact, record the
                  chosen path in LOADED_NATIVE_LIB (also makes the choice testable
                  without executing AVX-512 code)
  gdn_cpu.py      same preference in its own candidate list (it loads the library itself)
  build_linux.sh  --avx512 builds libbitsandbytes_cpu_avx512.so next to the AVX2 one
  build_release_linux.sh  ship the AVX-512 .so when it exists
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
PKG = REPO + r'\bitsandbytes'

EDITS = []

# ------------------------------------------------------------------ cextension.py
EDITS.append((
    PKG + r'\bitsandbytes\cextension.py',
    'from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR\n',
    'from bitsandbytes._cpu_caps import avx512_supported\n'
    'from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX, PACKAGE_DIR\n',
))

EDITS.append((
    PKG + r'\bitsandbytes\cextension.py',
    'logger = logging.getLogger(__name__)\n',
    'logger = logging.getLogger(__name__)\n'
    '\n'
    '# Which native library get_native_library() actually chose. Recorded so tests (and\n'
    '# `doctor`) can assert the AVX2/AVX-512 decision without executing kernel code.\n'
    'LOADED_NATIVE_LIB: Optional[Path] = None\n',
))

EDITS.append((
    PKG + r'\bitsandbytes\cextension.py',
    '    if torch._C._has_xpu:\n'
    '        binary_path = get_xpu_bnb_library_path()\n'
    '\n'
    '    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")\n',
    '    if torch._C._has_xpu:\n'
    '        binary_path = get_xpu_bnb_library_path()\n'
    '\n'
    '    # The AVX-512 backend ships as a second library. It must only be loaded when the CPU\n'
    '    # and OS really support AVX-512: compiling the whole translation unit with -mavx512f\n'
    '    # lets the compiler auto-vectorise ordinary kernels with zmm instructions, so the\n'
    '    # library SIGILLs on AVX2-only machines. A missing file (older wheel, platform that\n'
    '    # never built it) or an unknown CPU keeps the AVX2 library.\n'
    '    if avx512_supported():\n'
    '        avx512_binary_path = PACKAGE_DIR / f"libbitsandbytes_cpu_avx512{DYNAMIC_LIBRARY_SUFFIX}"\n'
    '        if avx512_binary_path.exists():\n'
    '            binary_path = avx512_binary_path\n'
    '\n'
    '    global LOADED_NATIVE_LIB\n'
    '    LOADED_NATIVE_LIB = binary_path\n'
    '\n'
    '    logger.debug(f"Loading bitsandbytes native library from: {binary_path}")\n',
))

# ------------------------------------------------------------------ gdn_cpu.py
EDITS.append((
    PKG + r'\bitsandbytes\gdn_cpu.py',
    '    if os.name == "nt":\n'
    '        order = [here / n for n in dll_names] + [pkg / "libbitsandbytes_cpu.dll"]\n',
    '    # Same AVX-512 preference as cextension: only when this CPU can execute it, and only\n'
    '    # if the artifact exists. Failure here can never be fatal -- the AVX2 library is a\n'
    '    # complete fallback.\n'
    '    try:\n'
    '        from ._cpu_caps import avx512_supported\n'
    '\n'
    '        if avx512_supported():\n'
    '            dll_names.insert(0, "libbitsandbytes_cpu_avx512.dll")\n'
    '            so_names.insert(0, "libbitsandbytes_cpu_avx512.so")\n'
    '    except Exception:  # pragma: no cover - the choice is an optimisation, not a requirement\n'
    '        pass\n'
    '    if os.name == "nt":\n'
    '        order = [here / n for n in dll_names] + [pkg / "libbitsandbytes_cpu.dll"]\n',
))

# ------------------------------------------------------------------ build_linux.sh
EDITS.append((
    PKG + r'\build_linux.sh',
    'RUN_SELFTEST=0\n'
    'for arg in "$@"; do\n'
    '  case "$arg" in\n'
    '    --clang) COMPILER="clang++" ;;\n'
    '    --no-avx2) NO_AVX2=1 ;;\n'
    '    --selftest) RUN_SELFTEST=1 ;;\n'
    '  esac\n'
    'done\n',
    'RUN_SELFTEST=0\n'
    'BUILD_AVX512=0\n'
    'for arg in "$@"; do\n'
    '  case "$arg" in\n'
    '    --clang) COMPILER="clang++" ;;\n'
    '    --no-avx2) NO_AVX2=1 ;;\n'
    '    --selftest) RUN_SELFTEST=1 ;;\n'
    '    --avx512) BUILD_AVX512=1 ;;\n'
    '  esac\n'
    'done\n',
))

EDITS.append((
    PKG + r'\build_linux.sh',
    'echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"\n',
    'echo "  -> $OUT ($(du -h "$OUT" | cut -f1))"\n'
    '\n'
    '# 可选的 AVX-512 变体：同一份源码、另加 AVX-512 flags。它【不能】发给 AVX2 机器\n'
    '# （整 TU 的 -mavx512f 会把普通内核自动向量化成 zmm，AVX2 上一调用就 SIGILL），\n'
    '# 载入端按 CPUID 选择，见 bitsandbytes/_cpu_caps.py。\n'
    'if [ "$BUILD_AVX512" = "1" ]; then\n'
    '  if [ "$ARCH" != "x86_64" ]; then\n'
    '    echo "  [AVX512] 跳过：架构 $ARCH 不是 x86_64"\n'
    '  else\n'
    '    echo "[2b/4] 编译 AVX-512 版（libbitsandbytes_cpu_avx512.so）..."\n'
    '    OUT512="bitsandbytes/libbitsandbytes_cpu_avx512.so"\n'
    '    rm -f "$OUT512"\n'
    '    AVX512_FLAGS="-mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl"\n'
    '    if echo \'int main(){return 0;}\' | "$COMPILER" -mavx512bf16 -x c++ - -o /dev/null 2>/dev/null; then\n'
    '      AVX512_FLAGS="$AVX512_FLAGS -mavx512bf16"\n'
    '    fi\n'
    '    "$COMPILER" -O2 -std=c++17 -fno-math-errno -fopenmp $AVX512_FLAGS \\\n'
    '      -fPIC -shared -DNOMINMAX -DNDEBUG \\\n'
    '      -DBUILD_CUDA=0 -DBUILD_HIP=0 -DBUILD_XPU=0 \\\n'
    '      -I csrc \\\n'
    '      csrc/cpu_ops.cpp csrc/cpu_gdn.cpp csrc/pythonInterface.cpp \\\n'
    '      -o "$OUT512" \\\n'
    '      -Wl,--no-undefined -Wl,--export-dynamic -Wl,-soname,libbitsandbytes_cpu_avx512.so\n'
    '    echo "  -> $OUT512 ($(du -h "$OUT512" | cut -f1))"\n'
    '    echo "  注意：这个 .so 只能在支持 AVX-512 的机器上加载；选择逻辑在 bitsandbytes/_cpu_caps.py"\n'
    '  fi\n'
    'fi\n',
))

# ------------------------------------------------------------------ build_release_linux.sh
EDITS.append((
    PKG + r'\build_release_linux.sh',
    'cp bitsandbytes/libbitsandbytes_cpu.so "$STAGE/bitsandbytes/"   # 编译出的 .so\n',
    'cp bitsandbytes/libbitsandbytes_cpu.so "$STAGE/bitsandbytes/"   # 编译出的 .so\n'
    '# 可选：AVX-512 变体（build_linux.sh --avx512 产出）；只在支持 AVX-512 的机器上被加载\n'
    'if [ -f bitsandbytes/libbitsandbytes_cpu_avx512.so ]; then\n'
    '  cp bitsandbytes/libbitsandbytes_cpu_avx512.so "$STAGE/bitsandbytes/"\n'
    'fi\n',
))


def main() -> int:
    for path, old, new in EDITS:
        try:
            text = open(path, encoding='utf-8').read()
        except OSError as exc:
            print('ABORT open %s: %s' % (path, exc))
            return 1
        count = text.count(old)
        if count != 1:
            print('ABORT %-60s anchor=%d' % (path.replace(REPO + '\\', ''), count))
            return 1
        open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %s' % path.replace(REPO + '\\', ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
