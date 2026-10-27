"""Document the AVX-512 dual-library selection that is now implemented.

The docs previously said "the dual library + selection is a todo". It is no longer a todo:
  * Linux `build_linux.sh --avx512` produces libbitsandbytes_cpu.so AND
    libbitsandbytes_cpu_avx512.so from the same sources;
  * Windows `build_manual.bat` with `BNB_BUILD_AVX512=1` produces the AVX-512 DLL;
  * bitsandbytes/_cpu_caps.py decides before loading (torch capability, /proc/cpuinfo fallback),
    cextension and gdn_cpu both honour it, and `bitsandbytes-cpu doctor` prints the decision;
  * BNB_CPU_DISABLE_AVX512 / BNB_CPU_FORCE_AVX512 are the escape hatches.

Updates: QUICKSTART 4.14 (CN+EN), TECHNICAL_GUIDE 3.12 (CN+EN), TECH_REPORT 3.11 (CN+EN), the
CLI's avx512 section (so the generated API_REFERENCE keeps the new text), both API_REFERENCE files
are regenerated from the CLI, and the root docs_cpu/ is mirrored again.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
PKG = REPO + r'\bitsandbytes'
DOCS = os.path.join(PKG, 'docs_cpu')
ROOT = os.path.join(REPO, 'docs_cpu')

QS_INTRO_OLD_ZH = (
    'CPU 后端在 AVX2 之外还有一套手写 AVX-512 内核（4-bit 反量化的 16-lane 路径、bf16 GEMV 的\n'
    'tinygemm 路径），默认关闭。构建方式（要求构建机自己能执行 AVX-512 探针）：\n'
    '\n'
    '```bat\n'
    ':: CMake 构建（Linux/GCC），BNB_ENABLE_AVX512 默认 OFF\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)
QS_INTRO_NEW_ZH = (
    'CPU 后端在 AVX2 之外还有一套手写 AVX-512 内核（4-bit 反量化的 16-lane 路径、bf16 GEMV 的\n'
    'tinygemm 路径）。两条构建路线：\n'
    '\n'
    '```bat\n'
    ':: 双库路线（推荐；在任意 x86_64 机器上都能编，产出 AVX2 + AVX-512 两个产物）\n'
    'bash build_linux.sh --avx512                 :: Linux: libbitsandbytes_cpu{,_avx512}.so\n'
    'set BNB_BUILD_AVX512=1 && build_manual\\build_manual.bat   :: Windows: 两个 .dll（需 VS2017+）\n'
    '\n'
    ':: CMake 路线：要求构建机自己就是 AVX-512 机器（探针真要跑），BNB_ENABLE_AVX512 默认 OFF\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)

QS_TAIL_OLD_ZH = (
    '⚠️ **注意**：`-mavx512f` 是**整编译单元**级开关，会让普通内核（dequantize、优化器、rmsnorm）\n'
    '被编译器自动向量化成 zmm → 在 AVX2 机器上**一调用就 SIGILL**。所以 AVX-512 二进制不能发给\n'
    'AVX2 机器，必须按 CPUID 选择加载（本 fork 尚未把"双库+选路"接入安装包，属于待办）。\n'
)
QS_TAIL_NEW_ZH = (
    '⚠️ **注意**：`-mavx512f` 是**整编译单元**级开关，会让普通内核（dequantize、优化器、rmsnorm）\n'
    '被编译器自动向量化成 zmm → 在 AVX2 机器上**一调用就 SIGILL**。所以 AVX-512 二进制不能发给\n'
    'AVX2 机器，载入端必须按 CPU 能力选择。\n'
    '\n'
    '**双库与自动选路（已实现）**：`bitsandbytes/_cpu_caps.py` 在加载前判断能力——优先问 torch\n'
    '的 `get_cpu_capability()`（Windows 上无 torch 时返回"不支持"），Linux 上回退 `/proc/cpuinfo`；\n'
    '只有支持时才加载 AVX-512 那个，文件缺失/能力未知/被环境变量关闭时一律回退 AVX2。\n'
    '`cextension.LOADED_NATIVE_LIB` 记录实际选择，`gdn_cpu` 用同一判定，`doctor` 直接打印判定结果。\n'
    '\n'
    '| 环境变量 | 作用 |\n'
    '|---|---|\n'
    '| `BNB_CPU_DISABLE_AVX512=1` | 强制用 AVX2 库（对比两个后端时用） |\n'
    '| `BNB_CPU_FORCE_AVX512=1` | 强制选 AVX-512 库（只应在支持 AVX-512 的机器上用） |\n'
    '\n'
    '```\n'
    'bitsandbytes-cpu doctor\n'
    '  cpu supports AVX-512  yes|no\n'
    '  AVX-512 library       libbitsandbytes_cpu_avx512.dll (present|absent)\n'
    '  this process will load  libbitsandbytes_cpu_avx512.dll|libbitsandbytes_cpu.dll\n'
    '```\n'
)

QS_INTRO_OLD_EN = (
    'Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit\n'
    'dequantise path and the bf16 GEMV tinygemm path), off by default. Build them with the opt-in\n'
    'switch (the build host must be able to *run* an AVX-512 probe):\n'
    '\n'
    '```bat\n'
    ':: CMake build (Linux/GCC); BNB_ENABLE_AVX512 defaults to OFF\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)
QS_INTRO_NEW_EN = (
    'Beyond AVX2 the CPU backend has a second set of hand-written AVX-512 kernels (the 16-lane 4-bit\n'
    'dequantise path and the bf16 GEMV tinygemm path). Two build routes:\n'
    '\n'
    '```bat\n'
    ':: dual-library route (recommended; builds on any x86_64 host: AVX2 + AVX-512 artifacts)\n'
    'bash build_linux.sh --avx512                 :: Linux: libbitsandbytes_cpu{,_avx512}.so\n'
    'set BNB_BUILD_AVX512=1 && build_manual\\build_manual.bat   :: Windows: two .dlls (VS2017+)\n'
    '\n'
    ':: CMake route: requires the build host itself to be AVX-512 capable (the probe really runs)\n'
    'cmake -DBNB_ENABLE_AVX512=ON ..\n'
    '```\n'
)

QS_TAIL_OLD_EN = (
    '⚠️ **Warning**: `-mavx512f` is a **translation-unit-wide** switch, so ordinary kernels\n'
    '(dequantise, optimizer, rmsnorm) get auto-vectorised to zmm and **SIGILL on an AVX2-only CPU** the\n'
    'moment they are called. An AVX-512 binary therefore cannot be shipped to AVX2 machines; it has to be\n'
    'loaded through a CPUID check (the dual-library selector is not wired into the installer yet -- it is\n'
    'on the todo list).\n'
)
QS_TAIL_NEW_EN = (
    '⚠️ **Warning**: `-mavx512f` is a **translation-unit-wide** switch, so ordinary kernels\n'
    '(dequantise, optimizer, rmsnorm) get auto-vectorised to zmm and **SIGILL on an AVX2-only CPU** the\n'
    'moment they are called. An AVX-512 binary therefore cannot be shipped to AVX2 machines; the loader\n'
    'decides by CPU capability.\n'
    '\n'
    '**Dual library and automatic selection (implemented)**: before loading, `bitsandbytes/_cpu_caps.py`\n'
    'asks torch (`get_cpu_capability()`; on Windows without torch the answer is "not supported") and\n'
    'falls back to `/proc/cpuinfo` on Linux. The AVX-512 artifact is used only when the CPU supports it;\n'
    'a missing file, an unknown CPU, or an environment override keeps the AVX2 library.\n'
    '`cextension.LOADED_NATIVE_LIB` records the choice, `gdn_cpu` uses the same query, and `doctor`\n'
    'prints the decision.\n'
    '\n'
    '| Environment variable | Effect |\n'
    '|---|---|\n'
    '| `BNB_CPU_DISABLE_AVX512=1` | force the AVX2 library (useful when comparing both backends) |\n'
    '| `BNB_CPU_FORCE_AVX512=1` | force the AVX-512 library (only on hardware that supports it) |\n'
    '\n'
    '```\n'
    'bitsandbytes-cpu doctor\n'
    '  cpu supports AVX-512  yes|no\n'
    '  AVX-512 library       libbitsandbytes_cpu_avx512.dll (present|absent)\n'
    '  this process will load  libbitsandbytes_cpu_avx512.dll|libbitsandbytes_cpu.dll\n'
    '```\n'
)

GUIDE_ZH_APPEND = (
    '\n'
    '**双库与选路的实现（本轮）**：AVX-512 版本单独编成 `libbitsandbytes_cpu_avx512.*`，和 AVX2\n'
    '版本一起分发。加载前由 `bitsandbytes/_cpu_caps.py` 判断能力：优先 torch 的\n'
    '`get_cpu_capability()`，Linux 上回退 `/proc/cpuinfo`；Windows 上无 torch 时返回"不支持"\n'
    '（**拒绝加载永远安全，猜错会 SIGILL**）。`cextension.LOADED_NATIVE_LIB` 记录实际加载的路径，\n'
    '`gdn_cpu._candidate_paths()` 用同一判定，`bitsandbytes-cpu doctor` 打印这三个判定。\n'
    '`BNB_CPU_DISABLE_AVX512` / `BNB_CPU_FORCE_AVX512` 是强制开关（后者只应在支持 AVX-512 的\n'
    '机器上用）。构建侧：Linux `build_linux.sh --avx512` 一次出两个 .so；Windows 用\n'
    '`BNB_BUILD_AVX512=1` 走 `build_manual.bat`。\n'
)
GUIDE_EN_APPEND = (
    '\n'
    '**The dual library and its selection (implemented this round)**: the AVX-512 backend is built as a\n'
    'separate `libbitsandbytes_cpu_avx512.*` artifact and shipped next to the AVX2 one. Before loading,\n'
    '`bitsandbytes/_cpu_caps.py` asks torch (`get_cpu_capability()`), falling back to `/proc/cpuinfo` on\n'
    'Linux; on Windows without torch the answer is "not supported" -- **refusing to load is always safe,\n'
    'guessing wrong SIGILLs**. `cextension.LOADED_NATIVE_LIB` records the chosen path,\n'
    '`gdn_cpu._candidate_paths()` uses the same query, and `bitsandbytes-cpu doctor` prints the decision.\n'
    '`BNB_CPU_DISABLE_AVX512` / `BNB_CPU_FORCE_AVX512` are the override switches (the latter only on\n'
    'hardware that really supports AVX-512). Build side: on Linux `build_linux.sh --avx512` emits both\n'
    '.so files; on Windows `BNB_BUILD_AVX512=1` makes `build_manual.bat` emit the second DLL.\n'
)

REPORT_LAST_ZH = (
    '**另一个必须记住的坑**：在本机（AVX2-only）强行编译 AVX-512 版，产物 1861 条 zmm 指令，一跑就\n'
    'SIGILL；zmm 归属证明崩溃点在**普通函数**（dequantize/rmsnorm/optimizer）而不是手写内核——\n'
    '`-mavx512f` 是整 TU 开关，这是"构建机必须自己支持 AVX-512"的物理原因。\n'
)
REPORT_APPEND_ZH = (
    '\n'
    '**双库落地（本轮）**：AVX-512 版本作为独立产物 `libbitsandbytes_cpu_avx512.*` 与 AVX2 版本\n'
    '一起分发，加载前由 `_cpu_caps.py` 判断 CPU 能力（torch `get_cpu_capability()` 优先，Linux\n'
    '回退 `/proc/cpuinfo`），`cextension` 与 `gdn_cpu` 共用同一判定，`doctor` 打印判定。\n'
    '构建侧：Linux `build_linux.sh --avx512` 一次出两个 .so（本机实测 293 KB / 330 KB，\n'
    '反汇编 zmm 行数 **0 / 1856**——AVX2 产物确实不含 EVEX）；Windows 用 `BNB_BUILD_AVX512=1`\n'
    '走 `build_manual.bat`（开发机没有 MSVC，这条产线未在本机验证，列入待办）。\n'
    '本机（AVX2）选择逻辑实测：正常导入选 AVX2 库；把一个同内容的假 AVX-512 DLL 放进包目录、\n'
    '并设 `BNB_CPU_FORCE_AVX512=1` 后，`cextension` 与 `gdn_cpu` 都选到 AVX-512 产物、`doctor`\n'
    '三个字段一致；清掉后恢复 AVX2。**真机 AVX-512 的端到端回归仍待做**（本机无 AVX-512）。\n'
)
REPORT_LAST_EN = (
    '**One more trap worth remembering**: forcing the AVX-512 build on the AVX2-only box yields 1861 zmm\n'
    'instruction lines and SIGILLs immediately; attributing the zmm lines shows the crash sites are the\n'
    '**ordinary** functions (dequantise/rmsnorm/optimizer), not the hand-written kernels -- `-mavx512f` is a\n'
    'whole-TU switch, and that is the physical reason the build host itself must support AVX-512.\n'
)
REPORT_APPEND_EN = (
    '\n'
    '**Dual library landed (this round)**: the AVX-512 backend is a separate\n'
    '`libbitsandbytes_cpu_avx512.*` artifact shipped next to the AVX2 one. Before loading,\n'
    '`_cpu_caps.py` queries the CPU (torch `get_cpu_capability()` first, `/proc/cpuinfo` fallback on\n'
    'Linux); `cextension` and `gdn_cpu` share the query and `doctor` prints it. Build side: on Linux\n'
    '`build_linux.sh --avx512` emits both .so files (measured here: 293 KB / 330 KB, zmm line counts\n'
    '**0 / 1856** -- the AVX2 artifact really contains no EVEX); on Windows `BNB_BUILD_AVX512=1` makes\n'
    '`build_manual.bat` emit the second DLL (untested on this machine: no MSVC here; on the todo list).\n'
    'Selection verified on this AVX2 box: a normal import loads the AVX2 library; dropping a dummy\n'
    'AVX-512 DLL into the package and setting `BNB_CPU_FORCE_AVX512=1` makes `cextension` and `gdn_cpu`\n'
    'both pick it and `doctor` agrees on all three fields; removing it restores AVX2. **The end-to-end\n'
    'regression on real AVX-512 hardware is still pending** (this machine has none).\n'
)

CLI_OLD = (
    '        ("the trap",\n'
    '         "-mavx512f is a whole-translation-unit switch: ordinary kernels get auto-vectorised to "\n'
    '         "zmm and SIGILL on an AVX2-only CPU", [\n'
    '             "# so an AVX-512 binary cannot be shipped to AVX2 machines; it has to be",\n'
    '             "# selected through a CPUID check at load time (installer wiring is a todo)",\n'
    '         ], SH),\n'
)
CLI_NEW = (
    '        ("the trap",\n'
    '         "-mavx512f is a whole-translation-unit switch: ordinary kernels get auto-vectorised to "\n'
    '         "zmm and SIGILL on an AVX2-only CPU", [\n'
    '             "# so an AVX-512 binary cannot be shipped to AVX2 machines; it is selected",\n'
    '             "# through a CPUID check at load time (bitsandbytes/_cpu_caps.py)",\n'
    '         ], SH),\n'
    '        ("how it is selected",\n'
    '         "the AVX-512 library ships next to the AVX2 one and is loaded only when the CPU can run "\n'
    '         "it; a missing file or an unknown CPU keeps the AVX2 library", [\n'
    '             "# capability: torch get_cpu_capability() first, /proc/cpuinfo fallback on Linux",\n'
    '             "# inspect the decision:",\n'
    '             "bitsandbytes-cpu doctor",\n'
    '             "#   cpu supports AVX-512  yes|no",\n'
    '             "#   AVX-512 library       libbitsandbytes_cpu_avx512.dll (present|absent)",\n'
    '             "#   this process will load  libbitsandbytes_cpu_avx512.dll|libbitsandbytes_cpu.dll",\n'
    '         ], SH),\n'
    '        ("build both artifacts",\n'
    '         "one source tree, two libraries; the AVX-512 one only runs on AVX-512 hardware", [\n'
    '             "bash build_linux.sh --avx512        # Linux: libbitsandbytes_cpu{,_avx512}.so",\n'
    '             "set BNB_BUILD_AVX512=1 && build_manual\\\\build_manual.bat   # Windows",\n'
    '             "# override switches: BNB_CPU_DISABLE_AVX512=1 / BNB_CPU_FORCE_AVX512=1",\n'
    '         ], SH),\n'
)

DOC_EDITS = [
    ('QUICKSTART.md', QS_INTRO_OLD_ZH, QS_INTRO_NEW_ZH, 'qs intro zh'),
    ('QUICKSTART.md', QS_TAIL_OLD_ZH, QS_TAIL_NEW_ZH, 'qs tail zh'),
    ('QUICKSTART_EN.md', QS_INTRO_OLD_EN, QS_INTRO_NEW_EN, 'qs intro en'),
    ('QUICKSTART_EN.md', QS_TAIL_OLD_EN, QS_TAIL_NEW_EN, 'qs tail en'),
    ('TECHNICAL_GUIDE.md', REPORT_LAST_ZH.replace('**另一个必须记住的坑**：在本机（AVX2-only）强行编译 AVX-512 版，产物 1861 条 zmm 指令，一跑就\nSIGILL；zmm 归属证明崩溃点在**普通函数**（dequantize/rmsnorm/optimizer）而不是手写内核——\n`-mavx512f` 是整 TU 开关，这是"构建机必须自己支持 AVX-512"的物理原因。\n',
                                               '**收益边界**：硬件峰值是真的（全宽 512 位机器 fp32 2.02×、BF16 点积 2.71×），但现有 AVX-512\n内核都是 memory-bound 的（反量化、优化器状态更新、4-bit GEMV），实测只有 0.97~1.02×。\n**要拿到峰值收益，必须新写 compute-bound 的 AVX-512 内核（BF16 / VNNI / GEMM），而不是把现有\n内核用 AVX-512 编译一遍。**\n'),
     None, 'guide zh anchor'),
]

# The engine section: append after its final paragraph (same text in both languages).
GUIDE_ANCHOR_ZH = (
    '**收益边界**：硬件峰值是真的（全宽 512 位机器 fp32 2.02×、BF16 点积 2.71×），但现有 AVX-512\n'
    '内核都是 memory-bound 的（反量化、优化器状态更新、4-bit GEMV），实测只有 0.97~1.02×。\n'
    '**要拿到峰值收益，必须新写 compute-bound 的 AVX-512 内核（BF16 / VNNI / GEMM），而不是把现有\n'
    '内核用 AVX-512 编译一遍。**\n'
)
GUIDE_ANCHOR_EN = (
    '**Where the gain stops**: the hardware ceiling is real (on a full-width 512-bit CPU, fp32 2.02x and\n'
    'BF16 dot product 2.71x), but today\'s AVX-512 kernels are all memory-bound (dequantise, optimizer state\n'
    'update, 4-bit GEMV), so the measured kernel speedup is only 0.97-1.02x. **Capturing the peak requires\n'
    'new compute-bound AVX-512 kernels (BF16 / VNNI / GEMM), not recompiling the existing ones with AVX-512\n'
    'flags.**\n'
)


def patch_file(name, old, new, label):
    path = os.path.join(DOCS, name)
    text = io.open(path, encoding='utf-8').read()
    count = text.count(old)
    if count != 1:
        print('ABORT %-22s %s anchor=%d' % (name, label, count))
        return False
    io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
    print('ok    %-22s %s' % (name, label))
    return True


def main() -> int:
    ok = True
    for name, old, new, label in DOC_EDITS[:4]:
        ok &= patch_file(name, old, new, label)
    ok &= patch_file('TECHNICAL_GUIDE.md', GUIDE_ANCHOR_ZH, GUIDE_ANCHOR_ZH + GUIDE_ZH_APPEND, 'guide zh append')
    ok &= patch_file('TECHNICAL_GUIDE_EN.md', GUIDE_ANCHOR_EN, GUIDE_ANCHOR_EN + GUIDE_EN_APPEND, 'guide en append')
    ok &= patch_file('TECH_REPORT.md', REPORT_LAST_ZH, REPORT_LAST_ZH + REPORT_APPEND_ZH, 'report zh append')
    ok &= patch_file('TECH_REPORT_EN.md', REPORT_LAST_EN, REPORT_LAST_EN + REPORT_APPEND_EN, 'report en append')

    # CLI section text, then regenerate the two API_REFERENCE files from the CLI.
    cli_path = os.path.join(REPO, 'bitsandbytes_cpu_cli.py')
    cli = io.open(cli_path, encoding='utf-8').read()
    if cli.count(CLI_OLD) != 1:
        print('ABORT CLI avx512 anchor=%d' % cli.count(CLI_OLD))
        return 1
    io.open(cli_path, 'w', encoding='utf-8', newline='').write(cli.replace(CLI_OLD, CLI_NEW, 1))
    print('ok    CLI avx512 section updated')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, 'bitsandbytes_cpu_cli.py', 'help', '--markdown'],
                       cwd=REPO, capture_output=True, text=True, encoding='utf-8', env=env)
    if r.returncode != 0 or not r.stdout.startswith('# CPU fork reference'):
        print('ABORT CLI markdown generation: %s' % r.stderr[:300])
        return 1
    body = r.stdout
    cn_header = ('<!-- 本文件由 `bitsandbytes-cpu help --markdown` 生成,请勿手改;'
                 '要改内容请改 bitsandbytes_cpu_cli.py。 -->\n')
    io.open(os.path.join(DOCS, 'API_REFERENCE.md'), 'w', encoding='utf-8', newline='').write(cn_header + body)
    io.open(os.path.join(DOCS, 'API_REFERENCE_EN.md'), 'w', encoding='utf-8', newline='').write(body)
    print('ok    API_REFERENCE regenerated (both languages)')
    for probe in ('how it is selected', 'build both artifacts'):
        print('      contains %-24s %s' % (probe, probe in body))

    # Mirror the shipped docs into the root docs_cpu/ so the two copies cannot disagree.
    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        data = io.open(os.path.join(DOCS, n), 'rb').read()
        io.open(os.path.join(ROOT, n), 'wb').write(data)
    print('ok    root docs_cpu/ mirrored (%d files)' % len(names))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
