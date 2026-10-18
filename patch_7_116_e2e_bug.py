"""Real-machine E2E results, and the -march=native bug the run exposed.

What the AVX-512 container proved: the dual library is selected correctly (doctor yes/present/
avx512, LOADED_NATIVE_LIB=avx512), the forced-AVX2 switch works, the package selftest passes with
the AVX-512 library loaded (0 failures), and a loader-level 8-bit optimizer step is 4.296 ms vs
4.294 ms forced-AVX2 (1.00x, consistent with 7.113: the kernels are memory-bound).

What it broke: the AVX2 artifact contained 1204 zmm instruction lines, because build_linux.sh used
`-march=native` and the build host is an AVX-512 machine -- so the artifact meant for AVX2 machines
was AVX-512-only and would SIGILL there. Fixed in build_linux.sh (explicit -mavx2 -mfma -mf16c +
post-build zmm==0 assertion) and asserted in the E2E script; this section records it.

Docs: QUICKSTART 4.14 artifact contract, TECHNICAL_GUIDE 3.12 flags contract, TECH_REPORT 3.11
real-machine numbers + pointer to the bug.
"""
from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPO = r'D:\work\bitsandbytes-CPU'
DOCS = os.path.join(REPO, 'bitsandbytes', 'docs_cpu')
ROOT = os.path.join(REPO, 'docs_cpu')
REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

QS_ZH_ANCHOR = (
    '| `BNB_CPU_FORCE_AVX512=1` | 强制选 AVX-512 库（只应在支持 AVX-512 的机器上用） |\n'
)
QS_ZH_ADD = QS_ZH_ANCHOR + (
    '\n'
    '**产物契约（必须记住）**：AVX2 产物里**不许出现 zmm/EVEX 指令**（否则它在 AVX2 机器上一调用\n'
    '就 SIGILL）。`build_linux.sh` 现在构建后会数一遍 zmm 行数，非 0 直接构建失败；**不要在 AVX2\n'
    '产物上用 `-march=native`** —— 在 AVX-512 构建机上它会悄悄把"AVX2 基线"变成 AVX-512-only\n'
    '（2026-10-18 实测：1204 条 zmm）。\n'
)
QS_EN_ANCHOR = (
    '| `BNB_CPU_FORCE_AVX512=1` | force the AVX-512 library (only on hardware that supports it) |\n'
)
QS_EN_ADD = QS_EN_ANCHOR + (
    '\n'
    '**Artifact contract (important)**: the AVX2 artifact must contain **no zmm/EVEX instructions**\n'
    '(otherwise it SIGILLs on an AVX2-only CPU as soon as a kernel is called). `build_linux.sh` now\n'
    'counts zmm lines after the build and fails the build when the count is not 0. **Do not use\n'
    '`-march=native` for the AVX2 artifact**: on an AVX-512 build host it silently turns the "AVX2\n'
    'baseline" into an AVX-512-only library (measured 2026-10-18: 1204 zmm lines).\n'
)

GUIDE_ZH_ANCHOR = (
    '`BNB_CPU_DISABLE_AVX512` / `BNB_CPU_FORCE_AVX512` 是强制开关（后者只应在支持 AVX-512 的\n'
    '机器上用）。构建侧：Linux `build_linux.sh --avx512` 一次出两个 .so；Windows 用\n'
    '`BNB_BUILD_AVX512=1` 走 `build_manual.bat`。\n'
)
GUIDE_ZH_ADD = GUIDE_ZH_ANCHOR + (
    '\n'
    '**构建 flags 的可移植性契约（真机抓到的一次事故，2026-10-18）**：AVX2 产物必须用**显式**\n'
    '`-mavx2 -mfma -mf16c` 生成，**不能用 `-march=native`**。在 AVX-512 构建机上 `-march=native`\n'
    '的含义是 AVX-512，于是"AVX2 基线"产物里出现 **1204 条 zmm 指令** —— 这份 .so 发给 AVX2 机器\n'
    '会 SIGILL，而它正是默认分发给 AVX2 用户的那份。修复：`build_linux.sh` 代码生成固定显式 flags，\n'
    '并在构建后**断言 zmm 行数 = 0**（非 0 直接 exit 3）；`avx512/dual_lib_e2e.sh` 也做同样的断言。\n'
    '本机（AVX2 构建机）重编验证：0 / 1856 条。\n'
)
GUIDE_EN_ANCHOR = (
    '`BNB_CPU_DISABLE_AVX512` / `BNB_CPU_FORCE_AVX512` are the override switches (the latter only on\n'
    'hardware that really supports AVX-512). Build side: on Linux `build_linux.sh --avx512` emits both\n'
    '.so files; on Windows `BNB_BUILD_AVX512=1` makes `build_manual.bat` emit the second DLL.\n'
)
GUIDE_EN_ADD = GUIDE_EN_ANCHOR + (
    '\n'
    '**The build-flag portability contract (an incident found on real hardware, 2026-10-18)**: the AVX2\n'
    'artifact must be built with **explicit** `-mavx2 -mfma -mf16c`, **never `-march=native`**. On an\n'
    'AVX-512 build host, native means AVX-512, so the "AVX2 baseline" came out with **1204 zmm\n'
    'instruction lines** -- that .so SIGILLs on AVX2 machines, and it is exactly the artifact shipped to\n'
    'AVX2 users by default. Fix: `build_linux.sh` now pins explicit flags for code generation and\n'
    '**asserts zmm == 0 after the build** (exit 3 otherwise); `avx512/dual_lib_e2e.sh` asserts the same.\n'
    'Re-verified on this AVX2 build host: 0 / 1856 lines.\n'
)

REPORT_ZH_ANCHOR = (
    '**双库落地（本轮）**：AVX-512 版本作为独立产物 `libbitsandbytes_cpu_avx512.*` 与 AVX2 版本\n'
    '一起分发，加载前由 `_cpu_caps.py` 判断 CPU 能力（torch `get_cpu_capability()` 优先，Linux\n'
    '回退 `/proc/cpuinfo`），`cextension` 与 `gdn_cpu` 共用同一判定，`doctor` 打印判定。\n'
)
REPORT_ZH_ADD = REPORT_ZH_ANCHOR + (
    '\n'
    '**真机 e2e（AVX-512 容器，2026-10-18）**：`build_linux.sh --avx512` 双产物构建成功，`doctor`\n'
    '判定 `cpu supports AVX-512 yes / AVX-512 library present / will load ...avx512.so`；\n'
    '`LOADED_NATIVE_LIB` 实测为 AVX-512 产物，`has_avx512f_cpu` / `has_avx512bf16_cpu` 均 present；\n'
    '`BNB_CPU_DISABLE_AVX512=1` 强制回 AVX2 产物；包自带 selftest **0 failure**（4-bit 层 /\n'
    '8-bit 优化器 / GDN）。加载器级计时：auto **4.296 ms** vs forced AVX2 **4.294 ms**（**1.00×**）\n'
    '—— 与 §7.113"现有内核 memory-bound、命中 AVX-512 路径也不提速"的结论一致。\n'
    '⚠️ **同一次运行抓出一个发布包级 bug**：AVX2 产物里有 1204 条 zmm 指令（`-march=native` 在\n'
    'AVX-512 构建机上把"AVX2 基线"变成了 AVX-512-only）。修复与验证见 §7.116。\n'
)

NEW_SECTION = '''### 7.116 ★★★ AVX-512 双库真机 e2e：选中正确；并抓出一个会毒化发布包的 `-march=native`

#### 7.116.1 真机结果（AVX-512 容器：EPYC 9Y24 / 2 核 / gcc 11.4）

| 检查 | 结果 |
|---|---|
| `doctor` 判定 | `cpu supports AVX-512 yes` · `AVX-512 library ... (present)` · `will load libbitsandbytes_cpu_avx512.so` |
| 加载器实际选择 | `LOADED_NATIVE_LIB = .../libbitsandbytes_cpu_avx512.so` |
| 能力符号 | `has_avx512f_cpu` present · `has_avx512bf16_cpu` present |
| 强制 AVX2 | `BNB_CPU_DISABLE_AVX512=1` ⇒ `LOADED_NATIVE_LIB = .../libbitsandbytes_cpu.so` |
| 包自检 | **0 failure**（4-bit 层 rel err 0.0913 / 8-bit 优化器 / GDN 原生内核） |
| 加载器级单步耗时 | auto **4.296 ms** vs forced AVX2 **4.294 ms** ⇒ **1.00×** |

**⇒ ★【双库闭环：真机上"该选 AVX-512 就选 AVX-512、强制关就回 AVX2、选完还能跑对"三件事全部
成立。加载器级 1.00× 也再次确认 §7.113 的判断 —— 现有内核 memory-bound，换 AVX-512 不产生提速。】**

#### 7.116.2 ★★★ 但这次运行抓出一个发布包级 bug：AVX2 产物里有 1204 条 zmm

```
同一次构建的产物：
  libbitsandbytes_cpu.so        292,712 B · ★ 1204 条 zmm 指令   ← 【本应是 0】
  libbitsandbytes_cpu_avx512.so 284,520 B · 1196 条 zmm
```

**机制**：`build_linux.sh` 的 AVX2 分支原来用 `-march=native` 做代码生成。在 AVX2 开发机
（R5-4500U）上 native=Zen2，产物确实没有 AVX-512；但在 **AVX-512 构建机**上 native 的含义就是
AVX-512 ⇒ 编译器把普通内核自动向量化成 zmm ⇒ **"AVX2 基线"变成 AVX-512-only 的二进制**。

**⇒ ★【后果比"性能差"严重得多：这份 .so 就是默认发给 AVX2 用户的产物，而它在 AVX2 机器上
一调用内核就 SIGILL。仓库自己的 CMake 选项特意防着这件事（要求构建机真跑探针），
手工 Linux 构建脚本一直有这个洞 —— 直到借一台 AVX-512 机器才照出来。】**

**修复（两处）**：

1. `build_linux.sh`：AVX2 代码生成改为**显式** `-mavx2 -mfma -mf16c`（`/proc/cpuinfo` 检测只用来
   决定"开不开 AVX2"，不再参与指令集选择）；构建后**断言 zmm 行数 = 0**，非 0 直接 `exit 3`。
2. `avx512/dual_lib_e2e.sh`：同样断言"AVX2 产物 zmm=0、AVX-512 产物 zmm>0"，违反则输出
   `PORTABLE_ARTIFACT_FAILED`。

**验证**：本机（AVX2 构建机）重编 —— AVX2 产物 **0 条** zmm、AVX-512 产物 **1856 条**；
断言打印"可移植性检查：zmm 指令 0 条"。**容器侧需用修复后的提交重跑一次确认。**

#### 7.116.3 教训与待办

**⇒ ★【"构建机能力和目标机能力"必须分开写进构建系统；`-march=native` 在任何"要分发"的产物上
都是无条件的错误 —— 它让【构建环境】决定了【目标环境】的指令集。】**

**待办**：① 用修复后的 commit（`5146d4f0` 之后）在容器重跑 e2e，确认 AVX2 产物 zmm=0；
② 给 GitHub Actions 的 Linux 构建也加 zmm 断言（runner 若为 AVX-512 同样会中招）；
③ Windows 侧 `build_manual.bat` 用的是显式 `/arch:AVX2`，没有这个洞。



'''

REPORT_EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.116'),
    ('**本文是 115 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 116 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 116'),
    ('**本报告的 115 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 116 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 116'),
    ('| AVX-512 双库 + 自动选路 | ✓ **Linux 双构建 293 KB/330 KB · 选路本机验证 · 真机 e2e 待做** | `bitsandbytes/_cpu_caps.py` · §7.115 |',
     '| AVX-512 双库 + 自动选路 | ✓ **Linux 双构建 · 选路本机验证 · 真机选中/强制关/selftest 全过** | `bitsandbytes/_cpu_caps.py` · §7.115 · §7.116 |\n'
     '| **`-march=native` 毒化 AVX2 产物** | ⚠️ **AVX2 产物 1204 条 zmm（会 SIGILL）；已改显式 flags + 构建断言 zmm=0** | §7.116 |',
     'H table: e2e + bug rows'),
]


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
    ok &= patch('QUICKSTART.md', QS_ZH_ANCHOR, QS_ZH_ADD, 'qs zh')
    ok &= patch('QUICKSTART_EN.md', QS_EN_ANCHOR, QS_EN_ADD, 'qs en')
    ok &= patch('TECHNICAL_GUIDE.md', GUIDE_ZH_ANCHOR, GUIDE_ZH_ADD, 'guide zh')
    ok &= patch('TECHNICAL_GUIDE_EN.md', GUIDE_EN_ANCHOR, GUIDE_EN_ADD, 'guide en')
    ok &= patch('TECH_REPORT.md', REPORT_ZH_ANCHOR, REPORT_ZH_ADD, 'report zh')
    # EN report: append the real-machine paragraph after its dual-library paragraph.
    en_anchor = (
        '**Dual library landed (this round)**: the AVX-512 backend is a separate\n'
        '`libbitsandbytes_cpu_avx512.*` artifact shipped next to the AVX2 one. Before loading,\n'
        '`_cpu_caps.py` queries the CPU (torch `get_cpu_capability()` first, `/proc/cpuinfo` fallback on\n'
        'Linux); `cextension` and `gdn_cpu` share the query and `doctor` prints it. Build side: on Linux\n'
    )
    # The EN paragraph continues; append after its end instead (find the closing sentence).
    en_tail = ('regression on real AVX-512 hardware is still pending** (this machine has none).\n')
    en_add = en_tail + (
        '\n'
        '**Real-machine e2e (AVX-512 container, 2026-10-18)**: both artifacts built, `doctor` reported\n'
        '`cpu supports AVX-512 yes` / `AVX-512 library present` / `will load ...avx512.so`;\n'
        '`LOADED_NATIVE_LIB` was the AVX-512 artifact, `has_avx512f_cpu` / `has_avx512bf16_cpu` both\n'
        'present; `BNB_CPU_DISABLE_AVX512=1` went back to the AVX2 artifact; the package selftest passed\n'
        'with **0 failures** (4-bit layer / 8-bit optimizer / GDN). Loader-level timing: auto **4.296 ms**\n'
        'vs forced AVX2 **4.294 ms** (**1.00x**) -- consistent with 7.113: the kernels are memory-bound,\n'
        'so hitting the AVX-512 path does not speed them up.\n'
        'WARNING: **the same run exposed a release-package bug**: the AVX2 artifact contained 1204 zmm\n'
        'instructions, because `-march=native` on an AVX-512 build host turns the "AVX2 baseline" into an\n'
        'AVX-512-only library. Fix and verification in 7.116.\n'
    )
    ok &= patch('TECH_REPORT_EN.md', en_tail, en_add, 'report en')

    for name, old, new in []:
        pass

    # report section + index
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
