"""Document the generic (no-AVX2) artifact and record its verification in the report.

What was built:
  * `build_linux.sh --generic` -> libbitsandbytes_cpu_generic.so, compiled with
    -march=x86-64 -DBNB_CPU_PORTABLE=1; BNB_CPU_PORTABLE turns the 31 AVX2 guard regions off at
    compile time, has_avx2_cpu() returns false, and cpu_gdn.cpp's VO<true> plus the <true> dispatch
    instantiations are not compiled. The build asserts objdump shows 0 AVX instructions (a leftover
    scalar-protection pragma used to force 53 of them; found and gated).
  * loader: `_cpu_caps.avx2_supported()` + three-way selection AVX-512 -> AVX2 -> generic, with
    BNB_CPU_FORCE_GENERIC=1 to force the portable artifact; doctor prints CPU AVX2 status, the
    generic library and the file it will load.
  * verified on this AVX2-only box: generic artifact 144 KB / 0 AVX instructions, portable selftest
    5/5 PASS, doctor/LOADED_NATIVE_LIB three-way decisions, Windows fallback when no generic DLL.

Docs: QUICKSTART 4.14, TECHNICAL_GUIDE 3.12 + section 6 item 6, TECH_REPORT 3.11, CLI avx512
section; report section 7.117.
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
DOCS = os.path.join(REPO, 'bitsandbytes', 'docs_cpu')
ROOT = os.path.join(REPO, 'docs_cpu')
CLI = os.path.join(REPO, 'bitsandbytes_cpu_cli.py')
REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

QS_ZH_ANCHOR = (
    '**产物契约（必须记住）**：AVX2 产物里**不许出现 zmm/EVEX 指令**（否则它在 AVX2 机器上一调用\n'
    '就 SIGILL）。`build_linux.sh` 现在构建后会数一遍 zmm 行数，非 0 直接构建失败；**不要在 AVX2\n'
    '产物上用 `-march=native`** —— 在 AVX-512 构建机上它会悄悄把"AVX2 基线"变成 AVX-512-only\n'
    '（2026-10-18 实测：1204 条 zmm）。\n'
)
QS_ZH_ADD = QS_ZH_ANCHOR + (
    '\n'
    '**通用 x86-64 产物（给没有 AVX2 的机器）**：第三条产物 '
    '`libbitsandbytes_cpu_generic.so`（`bash build_linux.sh --generic`）在**编译期**去掉全部\n'
    'AVX2/FMA（`-DBNB_CPU_PORTABLE`），只保留 SSE2 基线；构建后断言 **AVX 指令 = 0 条**（非 0\n'
    '直接失败）。载入端三级选择：**AVX-512 库 → AVX2 库 → generic 库**；`BNB_CPU_FORCE_GENERIC=1`\n'
    '可强制 generic（它在任何 x86-64 上都能跑，便于验证）。⚠️ Windows 目前只提供 AVX2 DLL，\n'
    '没有 generic DLL。\n'
)
QS_EN_ANCHOR = (
    '**Artifact contract (important)**: the AVX2 artifact must contain **no zmm/EVEX instructions**\n'
    '(otherwise it SIGILLs on an AVX2-only CPU as soon as a kernel is called). `build_linux.sh` now\n'
    'counts zmm lines after the build and fails the build when the count is not 0. **Do not use\n'
    '`-march=native` for the AVX2 artifact**: on an AVX-512 build host it silently turns the "AVX2\n'
    'baseline" into an AVX-512-only library (measured 2026-10-18: 1204 zmm lines).\n'
)
QS_EN_ADD = QS_EN_ANCHOR + (
    '\n'
    '**Generic x86-64 artifact (for machines without AVX2)**: the third artifact\n'
    '`libbitsandbytes_cpu_generic.so` (`bash build_linux.sh --generic`) removes all AVX2/FMA code at\n'
    '**compile time** (`-DBNB_CPU_PORTABLE`), keeping baseline SSE2 only; the build asserts **0 AVX\n'
    'instructions** (and fails otherwise). The loader selects **AVX-512 -> AVX2 -> generic**;\n'
    '`BNB_CPU_FORCE_GENERIC=1` forces the generic artifact (it runs on any x86-64, which makes the\n'
    'choice testable). NOTE: Windows currently ships only the AVX2 DLL, no generic DLL.\n'
)

GUIDE_ZH_ANCHOR = (
    '**双库与选路的实现（本轮）**：AVX-512 版本单独编成 `libbitsandbytes_cpu_avx512.*`，和 AVX2\n'
    '版本一起分发。加载前由 `bitsandbytes/_cpu_caps.py` 判断能力：优先 torch 的\n'
    '`get_cpu_capability()`，Linux 上回退 `/proc/cpuinfo`；Windows 上无 torch 时返回"不支持"\n'
    '（**拒绝加载永远安全，猜错会 SIGILL**）。`cextension.LOADED_NATIVE_LIB` 记录实际加载的路径，\n'
    '`gdn_cpu._candidate_paths()` 用同一判定，`bitsandbytes-cpu doctor` 打印这三个判定。\n'
    '`BNB_CPU_DISABLE_AVX512` / `BNB_CPU_FORCE_AVX512` 是强制开关（后者只应在支持 AVX-512 的\n'
    '机器上用）。构建侧：Linux `build_linux.sh --avx512` 一次出两个 .so；Windows 用\n'
    '`BNB_BUILD_AVX512=1` 走 `build_manual.bat`。\n'
)
GUIDE_ZH_ADD = GUIDE_ZH_ANCHOR + (
    '\n'
    '**第三产物：generic（无 AVX2）**。运行时守卫**不能**替代"二进制里没有 AVX2 指令"：常规构建\n'
    '整体用 `-mavx2 -mfma` 编译，编译器可以在任何函数（包括标量孪生和胶水代码）里发射 AVX2/FMA。\n'
    '`--generic` 用 `-march=x86-64 -DBNB_CPU_PORTABLE=1`：① 31 处 `__AVX2__ || x86` 守卫改成\n'
    '`BNB_AVX2_CODE && ...`，整块 AVX2 代码被预处理掉；② `has_avx2_cpu()` 直接返回 false；\n'
    '③ `cpu_gdn.cpp` 的 `VO<true>` 与 `<true>` 分派实例不编译，只剩标量孪生。构建后断言\n'
    '`objdump` 里 AVX 指令 = 0（本机实测：144 KB、**0 条**、便携版 selftest **5/5 PASS**）。\n'
    '选路三级：AVX-512 → AVX2 → generic（`BNB_CPU_FORCE_GENERIC=1` 强制 generic）。\n'
)
GUIDE_EN_ANCHOR = (
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
GUIDE_EN_ADD = GUIDE_EN_ANCHOR + (
    '\n'
    '**The third artifact: generic (no AVX2)**. Runtime guards cannot substitute for "the binary\n'
    'contains no AVX2 instructions": the regular build compiles the whole TU with `-mavx2 -mfma`, so the\n'
    'compiler may emit AVX2/FMA anywhere, including in the scalar twins and the glue. `--generic` uses\n'
    '`-march=x86-64 -DBNB_CPU_PORTABLE=1`: (1) the 31 `__AVX2__ || x86` guard regions become\n'
    '`BNB_AVX2_CODE && ...` and are preprocessed out; (2) `has_avx2_cpu()` returns false; (3)\n'
    '`cpu_gdn.cpp`\'s `VO<true>` and `<true>` dispatch instantiations are not compiled, leaving the\n'
    'scalar twins. The build asserts `objdump` shows 0 AVX instructions (measured here: 144 KB, **0\n'
    'instructions**, portable selftest **5/5 PASS**). Selection is three-way: AVX-512 -> AVX2 -> generic\n'
    '(`BNB_CPU_FORCE_GENERIC=1` forces generic).\n'
)

ITEM6_ZH_OLD = (
    '6. **非 AVX2 x86（当前不支持，需要单独的 generic 产物）**：**运行时**那一半是有的——'
)
ITEM6_ZH_NEW = (
    '6. **非 AVX2 x86（已支持：generic 产物）**：`build_linux.sh --generic` 产出\n'
    '   `libbitsandbytes_cpu_generic.so`（编译期去掉全部 AVX2/FMA，构建断言 AVX 指令=0），\n'
    '   载入端三级选择（AVX-512 → AVX2 → generic）；便携版 selftest 5/5 PASS，\n'
    '   `BNB_CPU_FORCE_GENERIC=1` 可强制。真·非 AVX2 的动态验证用 qemu-user（-cpu Nehalem，\n'
    '   e2e 脚本内置、可选）。⚠️ Windows 暂只有 AVX2 DLL。\n'
    '   （历史：**运行时**那一半本来就有——'
)
ITEM6_EN_OLD = (
    '6. **non-AVX2 x86 (not supported today; a separate generic artifact is needed)**: the\n'
)
ITEM6_EN_NEW = (
    '6. **non-AVX2 x86 (supported now: the generic artifact)**: `build_linux.sh --generic` emits\n'
    '   `libbitsandbytes_cpu_generic.so` (all AVX2/FMA removed at compile time, build asserts 0 AVX\n'
    '   instructions) and the loader is three-way (AVX-512 -> AVX2 -> generic); portable selftest\n'
    '   5/5 PASS, `BNB_CPU_FORCE_GENERIC=1` forces it. Dynamic validation on a real pre-AVX2 CPU is\n'
    '   done with qemu-user (`-cpu Nehalem`, built into the e2e script, optional). NOTE: Windows ships\n'
    '   only the AVX2 DLL for now. (History: the runtime half always existed --\n'
)

RPT_ZH_ANCHOR = (
    '⚠️ **同一次运行抓出一个发布包级 bug**：AVX2 产物里有 1204 条 zmm 指令（`-march=native` 在\n'
    'AVX-512 构建机上把"AVX2 基线"变成了 AVX-512-only）。修复与验证见 §7.116。\n'
)
RPT_ZH_ADD = RPT_ZH_ANCHOR + (
    '\n'
    '**第三产物：generic x86-64（无 AVX2）**：Linux `build_linux.sh --generic` 产出 144 KB 的\n'
    '`libbitsandbytes_cpu_generic.so`，编译期去掉 AVX2/FMA；反汇编 **AVX 指令 0 条**（构建脚本内置\n'
    '断言，非 0 直接失败）。便携版 selftest（同源码 + `-DBNB_CPU_PORTABLE=1`）**5/5 PASS**；\n'
    '`BNB_CPU_FORCE_GENERIC=1` 时 `doctor`/`LOADED_NATIVE_LIB` 选 generic，否则按 CPU 能力选\n'
    'AVX2/AVX-512。⚠️ Windows 目前只有 AVX2 DLL。详见 §7.117。\n'
)
RPT_EN_ANCHOR = (
    'WARNING: **the same run exposed a release-package bug**: the AVX2 artifact contained 1204 zmm\n'
    'instructions, because `-march=native` on an AVX-512 build host turns the "AVX2 baseline" into an\n'
    'AVX-512-only library. Fix and verification in 7.116.\n'
)
RPT_EN_ADD = RPT_EN_ANCHOR + (
    '\n'
    '**Third artifact: generic x86-64 (no AVX2)**: `build_linux.sh --generic` produces a 144 KB\n'
    '`libbitsandbytes_cpu_generic.so` with all AVX2/FMA removed at compile time; disassembly shows\n'
    '**0 AVX instructions** (asserted by the build, which fails otherwise). The portable selftest\n'
    '(same sources + `-DBNB_CPU_PORTABLE=1`) passes **5/5**; with `BNB_CPU_FORCE_GENERIC=1` doctor and\n'
    '`LOADED_NATIVE_LIB` pick the generic artifact, otherwise AVX-512/AVX2 per CPU capability. NOTE:\n'
    'Windows ships only the AVX2 DLL. Details in 7.117.\n'
)

NEW_SECTION = '''### 7.117 ★★★ 第三产物：generic x86-64（无 AVX2）——把"能跑"从运行时守卫变成二进制事实

#### 7.117.1 为什么运行时守卫不够

**§7.116 的教训是"构建环境决定了目标环境的指令集"。同一条逻辑还有第二个受害者：非 AVX2 的
x86 机器。** 代码里 17 处 `has_avx2_cpu()` 守卫 + 每内核标量孪生（`cpu_gdn.cpp` 的
`VO<true>/VO<false>` 模板）看上去"能降级"，但常规产物整体用 `-mavx2 -mfma` 编译 ⇒ 编译器
可以在**任何**函数（包括标量孪生、胶水代码）里发射 AVX2/FMA ⇒ 真·非 AVX2 上一调用就 SIGILL。
更糟的是用来兜底的 `build_linux.sh --no-avx2`（`-march=x86-64`）**根本编译不过**：两个 TU 的
intrinsics 用 `__GNUC__ && x86` / `GDN_X86` 守卫而不是 `__AVX2__`。

**⇒ ★【"运行时守卫"回答的是"调哪个内核"；"二进制里有没有 AVX2 指令"是另一个问题，
只能靠编译期隔离回答。】**

#### 7.117.2 实现：`BNB_CPU_PORTABLE` 编译期开关 + 第三产物

```
build_linux.sh --generic
  ⇒ -march=x86-64 -DBNB_CPU_PORTABLE=1
  ⇒ libbitsandbytes_cpu_generic.so  144 KB
```

三处改动：
1. `cpu_ops.cpp`：31 处 `#if defined(__AVX2__) || (x86…)` 守卫改为
   `#if BNB_AVX2_CODE && (...)`；`has_avx2_cpu()` 在便携构建直接 `return false`；
2. `cpu_gdn.cpp`：`VO<true>` 特化与两处 `<true>` 分派实例用 `!defined(BNB_CPU_PORTABLE)` 排除，
   只剩标量孪生；
3. `build_linux.sh`：构建后断言 **`objdump` 里 AVX 指令 = 0 条**，非 0 直接失败。

**过程中还抓到一个隐蔽的第二源**：第一版 generic 产物仍有 **53 条 AVX 指令**，全部落在
`quantize_cpu_impl` / `build_quantize_lut` / `LUTCache` —— 来源不是 AVX2 内核，而是
**"防止 AVX-512 构建发射 EVEX"的 `#pragma GCC target("avx2,...")`**（它本来只该在 AVX-512
构建里出现，却把 generic 构建的普通函数强行编成了 AVX）。修法：该 pragma 也用
`!defined(BNB_CPU_PORTABLE)` 关掉。**⇒ 这个断言不是形式主义：它一出现就抓到了真问题。**

#### 7.117.3 选路：三级（AVX-512 → AVX2 → generic）

```
_cpu_caps.py
  avx512_supported()  → AVX-512 库（存在时）
  avx2_supported()    → 常规库
  否则                 → generic 库（存在时；否则退回常规库并自然报错）
  BNB_CPU_FORCE_GENERIC=1 强制 generic（任何 x86-64 都能跑，所以可用来验证选择逻辑）
```

`cextension` / `gdn_cpu` 共用同一判定；`doctor` 现在打印"CPU 支持 AVX-512 / CPU 支持 AVX2 /
两个库在不在 / 会加载哪个"。

#### 7.117.4 本机验证（AVX2-only 开发机）

| 项 | 结果 |
|---|---|
| 三产物构建 | 288 KB（AVX2，zmm 0）/ 324 KB（AVX-512，zmm 1856）/ **144 KB（generic，AVX 0）** |
| 便携版 selftest | **5/5 PASS**（标量路径，无 AVX2 代码） |
| `doctor`（正常） | CPU AVX2 yes → will load `libbitsandbytes_cpu.so` |
| `doctor`（`BNB_CPU_FORCE_GENERIC=1`） | CPU AVX2 no → will load `libbitsandbytes_cpu_generic.so`，loads OK |
| Windows（无 generic DLL） | `BNB_CPU_FORCE_GENERIC=1` 时安全回退到 AVX2 DLL |
| e2e 脚本 | 已扩成 10 段（三产物 + 三级选路 + 便携 selftest + 可选 qemu-user `-cpu Nehalem`） |

#### 7.117.5 待办

1. **真机/模拟端到端**：在容器跑新 e2e（`avx512/dual_lib_e2e.sh`）——其中 qemu-user 段会尝试
   `-cpu Nehalem` 动态执行便携 selftest；若容器无密码 bash sudo 则跳过，静态证据仍在。
2. **Windows generic DLL** 未提供（MSVC 不加 `/arch:AVX2` 可编，但本机无 VS 无法验证）；
3. 发布脚本已能携带 generic `.so`（`build_release_linux.sh` 检测到就复制），默认仍只带 AVX2，
   保守默认不变。



'''

REPORT_EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.117'),
    ('**本文是 116 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 117 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 117'),
    ('**本报告的 116 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 117 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 117'),
    ('| **`-march=native` 毒化 AVX2 产物** | ⚠️ **AVX2 产物 1204 条 zmm（会 SIGILL）；已改显式 flags + 构建断言 zmm=0** | §7.116 |',
     '| **`-march=native` 毒化 AVX2 产物** | ⚠️ **AVX2 产物 1204 条 zmm（会 SIGILL）；已改显式 flags + 构建断言 zmm=0** | §7.116 |\n'
     '| **generic x86-64 第三产物** | ✓ **144 KB · AVX 指令 0 条 · 便携 selftest 5/5 · 三级选路** | `build_linux.sh --generic` · §7.117 |',
     'H table: generic row'),
]


def patch_doc(name, old, new, label):
    path = os.path.join(DOCS, name)
    text = io.open(path, encoding='utf-8').read()
    n = text.count(old)
    if n != 1:
        print('ABORT %-24s %s anchor=%d' % (name, label, n))
        return False
    io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
    print('ok    %-24s %s' % (name, label))
    return True


def main() -> int:
    ok = True
    ok &= patch_doc('QUICKSTART.md', QS_ZH_ANCHOR, QS_ZH_ADD, 'qs zh generic')
    ok &= patch_doc('QUICKSTART_EN.md', QS_EN_ANCHOR, QS_EN_ADD, 'qs en generic')
    ok &= patch_doc('TECHNICAL_GUIDE.md', GUIDE_ZH_ANCHOR, GUIDE_ZH_ADD, 'guide zh generic')
    ok &= patch_doc('TECHNICAL_GUIDE_EN.md', GUIDE_EN_ANCHOR, GUIDE_EN_ADD, 'guide en generic')
    ok &= patch_doc('TECHNICAL_GUIDE.md', ITEM6_ZH_OLD, ITEM6_ZH_NEW, 'guide zh item6')
    ok &= patch_doc('TECHNICAL_GUIDE_EN.md', ITEM6_EN_OLD, ITEM6_EN_NEW, 'guide en item6')
    ok &= patch_doc('TECH_REPORT.md', RPT_ZH_ANCHOR, RPT_ZH_ADD, 'report zh generic')
    ok &= patch_doc('TECH_REPORT_EN.md', RPT_EN_ANCHOR, RPT_EN_ADD, 'report en generic')

    # CLI: extend the avx512 section with the generic artifact + selection order.
    cli_old = (
        '        ("build both artifacts",\n'
        '         "one source tree, two libraries; the AVX-512 one only runs on AVX-512 hardware", [\n'
        '             "bash build_linux.sh --avx512        # Linux: libbitsandbytes_cpu{,_avx512}.so",\n'
        '             "set BNB_BUILD_AVX512=1 && build_manual\\\\build_manual.bat   # Windows",\n'
        '             "# override switches: BNB_CPU_DISABLE_AVX512=1 / BNB_CPU_FORCE_AVX512=1",\n'
        '         ], SH),\n'
    )
    cli_new = cli_old + (
        '        ("generic artifact: run on CPUs without AVX2",\n'
        '         "a third library, built with all AVX2/FMA removed at compile time, for pre-Haswell x86;"\n'
        '         " the loader picks it only when the CPU cannot run the AVX2 library", [\n'
        '             "bash build_linux.sh --generic       # libbitsandbytes_cpu_generic.so (Linux)",\n'
        '             "# build asserts: objdump shows 0 AVX instructions (else the build fails)",\n'
        '             "# selection order: AVX-512 -> AVX2 -> generic",\n'
        '             "#   BNB_CPU_FORCE_GENERIC=1 forces the generic library (any x86-64 can run it)",\n'
        '         ], SH),\n'
    )
    cli = io.open(CLI, encoding='utf-8').read()
    if cli.count(cli_old) != 1:
        print('ABORT CLI avx512 build-both anchor=%d' % cli.count(cli_old))
        ok = False
    else:
        io.open(CLI, 'w', encoding='utf-8', newline='').write(cli.replace(cli_old, cli_new, 1))
        print('ok    CLI avx512 section: generic artifact entry')

    # regenerate API_REFERENCE from the CLI
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, 'bitsandbytes_cpu_cli.py', 'help', '--markdown'],
                       cwd=REPO, capture_output=True, text=True, encoding='utf-8', env=env)
    if r.returncode != 0 or not r.stdout.startswith('# CPU fork reference'):
        print('ABORT CLI markdown generation: %s' % r.stderr[:300])
        return 1
    cn_header = ('<!-- 本文件由 `bitsandbytes-cpu help --markdown` 生成,请勿手改;'
                 '要改内容请改 bitsandbytes_cpu_cli.py。 -->\n')
    io.open(os.path.join(DOCS, 'API_REFERENCE.md'), 'w', encoding='utf-8', newline='').write(cn_header + r.stdout)
    io.open(os.path.join(DOCS, 'API_REFERENCE_EN.md'), 'w', encoding='utf-8', newline='').write(r.stdout)
    print('ok    API_REFERENCE regenerated')

    # report
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in REPORT_EDITS:
        if out.count(old) != 1:
            print('ABORT report %-20s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    report %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))

    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
