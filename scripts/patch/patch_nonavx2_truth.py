"""Correct the "non-AVX2 works" claim after testing it.

The docs claimed that on a CPU without AVX2 the kernels fall back to scalar and the library keeps
running, and that `build_linux.sh --no-avx2` produces a generic x86-64 artifact. Testing that path
shows two different things:

  * the runtime half exists and works: has_avx2_cpu()/gdn_has_avx2() (CPUID + XGETBV) guard the
    SIMD paths, every kernel has a scalar twin, BNB_CPU_NO_AVX2=1 forces the scalar twin and the
    selftest passes 5/5 (measured);
  * the binary half does not: the distributed artifacts are compiled with -mavx2/-mfma (Windows
    /arch:AVX2), so the compiler may emit AVX2/FMA anywhere including inside the scalar twins; and
    `--no-avx2` (generic x86-64) currently does not even compile, because cpu_ops.cpp and
    cpu_gdn.cpp guard their intrinsics with `__GNUC__ && x86` / GDN_X86 instead of `__AVX2__`.

So the honest statement is "AVX2 is the supported x86 baseline; a generic artifact does not exist
yet". This patch corrects the three places that said otherwise and points at the design that would
fix it (a third artifact, exactly like the AVX-512 dual library).
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
PKG = REPO + r'\bitsandbytes'
DOCS = os.path.join(PKG, 'docs_cpu')
ROOT = os.path.join(REPO, 'docs_cpu')

GUIDE_ZH_OLD = (
    '6. **非 AVX2 CPU（实验性）**：所有 AVX2 内核都有运行时 `has_avx2_cpu()`（CPUID）保护 + '
    '`BNB_CPU_NO_AVX2=1` 环境变量强制关。非 AVX2 机器运行时走 scalar fallback——**能跑（慢），'
    '但不保证性能**。`build_linux.sh` 会自动检测（无 AVX2 则编译 `-march=x86-64` 不定义 '
    '`__AVX2__`）。**实验性：无 AVX2 设备实测，理论可跑，遇崩设 `BNB_CPU_NO_AVX2=1` 或 '
    '`build_linux.sh --no-avx2`。**\n'
)
GUIDE_ZH_NEW = (
    '6. **非 AVX2 x86（当前不支持，需要单独的 generic 产物）**：**运行时**那一半是有的——'
    '所有 AVX2 内核都有 `has_avx2_cpu()`（CPUID+XGETBV）守卫、`cpu_gdn.cpp` 用 '
    '`VO<true>/VO<false>` 模板孪生、`BNB_CPU_NO_AVX2=1` 可强制标量路径（A/B 测试用，'
    '实测 selftest 5/5 PASS）。**但"二进制层面的保证"没有**：分发的产物整体按 `-mavx2 -mfma`'
    '（Windows `/arch:AVX2`）编译，编译器可以在**任何函数**（包括标量孪生和胶水代码）里发射 '
    'AVX2/FMA 指令 ⇒ 在真正的非 AVX2 x86 上不保证能跑。'
    '而且 `build_linux.sh --no-avx2` 的通用 x86-64 构建**目前编译不过**（`cpu_ops.cpp` / '
    '`cpu_gdn.cpp` 的 intrinsics 用 `__GNUC__ && x86` / `GDN_X86` 守卫而不是 `__AVX2__`，'
    '`-march=x86-64` 下报 `target specific option mismatch`）。'
    '**⇒ 现状：x86 的硬下限是 AVX2（本文档与报告 §7.1 的约定）；要支持非 AVX2，需按 AVX-512 '
    '双库的同一模式增加第三个 generic 产物（AVX2 内核隔离到独立 TU / target 属性函数 + '
    '载入端按 CPUID 三级选择 + 构建断言"generic 产物 0 条 ymm/vfmadd"）。**\n'
)

GUIDE_EN_OLD = (
    '6. **non-AVX2 CPUs (experimental)**: every AVX2 kernel has a runtime `has_avx2_cpu()` (CPUID) '
    'guard and `BNB_CPU_NO_AVX2=1` forces scalar. On a non-AVX2 machine the scalar fallback runs '
    '(slowly). `build_linux.sh` detects it (generic `-march=x86-64` without `__AVX2__`). '
    '**Experimental: never tested on a non-AVX2 device; if it crashes set `BNB_CPU_NO_AVX2=1` or '
    '`build_linux.sh --no-avx2`.**\n'
)
GUIDE_EN_NEW = (
    '6. **non-AVX2 x86 (not supported today; needs a separate generic artifact)**: the **runtime** '
    'half exists -- every AVX2 kernel has a `has_avx2_cpu()` (CPUID+XGETBV) guard, `cpu_gdn.cpp` has '
    '`VO<true>/VO<false>` template twins, and `BNB_CPU_NO_AVX2=1` forces the scalar path (A/B testing; '
    'selftest passes 5/5 with it). **The binary-level guarantee does not exist**: the shipped artifacts '
    'are compiled with `-mavx2 -mfma` (Windows `/arch:AVX2`), so the compiler may emit AVX2/FMA in '
    '**any** function, including the scalar twins and the glue code. On a real non-AVX2 x86 CPU they '
    'are therefore not guaranteed to run. `build_linux.sh --no-avx2` (generic x86-64) **does not even '
    'compile today**: the intrinsics in `cpu_ops.cpp` / `cpu_gdn.cpp` are guarded by `__GNUC__ && x86` '
    '/ `GDN_X86` rather than `__AVX2__`, so `-march=x86-64` fails with "target specific option '
    'mismatch". **=> x86 baseline is AVX2 (the documented contract, report 7.1). Supporting non-AVX2 '
    'means adding a third artifact the same way the AVX-512 dual library works** (AVX2 kernels isolated '
    'in their own TU or target-attributed functions, three-way CPUID selection in the loader, and a '
    'build assertion that the generic artifact has zero ymm/vfmadd instructions).**\n'
)

QS_ZH_OLD = (
    '脚本会自动探测 AVX2（x86_64）或 NEON（aarch64），并探测 OpenMP 是否可用；探测失败时自动降级，'
    '而不是让链接阶段报错。产物为 `bitsandbytes/libbitsandbytes_cpu.so`。\n'
)
QS_ZH_NEW = (
    '脚本会自动探测 AVX2（x86_64）或 NEON（aarch64），并探测 OpenMP 是否可用（OpenMP 缺失时关闭）。'
    '产物为 `bitsandbytes/libbitsandbytes_cpu.so`。**x86 的硬下限是 AVX2**：无 AVX2 的机器没有'
    '可用的通用产物（`--no-avx2` 当前编译不过，原因见 `TECHNICAL_GUIDE.md` §6.6）。\n'
)
QS_EN_OLD = (
    'The script detects AVX2 on x86_64 or NEON on aarch64, and whether OpenMP is available; when a '
    'probe fails it degrades instead of failing at link time. The artifact is '
    '`bitsandbytes/libbitsandbytes_cpu.so`.\n'
)
QS_EN_NEW = (
    'The script detects AVX2 on x86_64 or NEON on aarch64, and whether OpenMP is available (OpenMP is '
    'disabled when missing). The artifact is `bitsandbytes/libbitsandbytes_cpu.so`. **The hard x86 '
    'baseline is AVX2**: there is no usable generic artifact for non-AVX2 machines yet (`--no-avx2` '
    'currently does not compile; see `TECHNICAL_GUIDE_EN.md` 6.6).\n'
)

BUILD_OLD = (
    '    # 非 AVX2 CPU：不定义 __AVX2__、用通用 x86-64，运行时走 scalar（能跑，慢）\n'
    '    ARCH_FLAGS="-march=x86-64"\n'
    '    echo "  [scalar] 未检测到 AVX2，编译为通用 x86-64（实验性：无 AVX2 设备测试，"\n'
    '    echo "           理论能跑但不保证性能；若崩请设 BNB_CPU_NO_AVX2=1 或 --no-avx2）"\n'
)
BUILD_NEW = (
    '    # 非 AVX2 CPU：本该走通用 x86-64，但当前 TU 里的 intrinsics 用 __GNUC__&&x86 /\n'
    '    # GDN_X86 守卫而不是 __AVX2__，-march=x86-64 编译会报 target specific option\n'
    '    # mismatch ⇒ 这条路径【当前不可用】（要修见 TECHNICAL_GUIDE §6.6）。\n'
    '    ARCH_FLAGS="-march=x86-64"\n'
    '    echo "  [scalar] 未检测到 AVX2；注意：通用 x86-64 构建目前不受支持（intrinsics 未隔离，"\n'
    '    echo "           编译会失败）。x86 硬下限是 AVX2；非 AVX2 需要单独的 generic 产物。"\n'
)


def patch(name, old, new, label):
    path = os.path.join(DOCS, name)
    text = io.open(path, encoding='utf-8').read()
    if text.count(old) != 1:
        print('ABORT %-24s %s anchor=%d' % (name, label, text.count(old)))
        return False
    io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
    print('ok    %-24s %s' % (name, label))
    return True


def main() -> int:
    ok = True
    ok &= patch('TECHNICAL_GUIDE.md', GUIDE_ZH_OLD, GUIDE_ZH_NEW, 'guide zh item 6')
    ok &= patch('TECHNICAL_GUIDE_EN.md', GUIDE_EN_OLD, GUIDE_EN_NEW, 'guide en item 6')
    ok &= patch('QUICKSTART.md', QS_ZH_OLD, QS_ZH_NEW, 'quickstart zh')
    ok &= patch('QUICKSTART_EN.md', QS_EN_OLD, QS_EN_NEW, 'quickstart en')

    build = io.open(PKG + r'\build_linux.sh', encoding='utf-8').read()
    if build.count(BUILD_OLD) != 1:
        print('ABORT build_linux.sh anchor=%d' % build.count(BUILD_OLD))
        ok = False
    else:
        io.open(PKG + r'\build_linux.sh', 'w', encoding='utf-8', newline='').write(
            build.replace(BUILD_OLD, BUILD_NEW, 1))
        print('ok    build_linux.sh scalar message')

    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
