"""Record the Termux/ARM64 regression check and the compile fix it exposed.

Found while cross-compiling the current sources for aarch64 (zig clang, aarch64-linux-gnu):
  csrc/cpu_ops.cpp had an UNGUARDED `_mm_sfence()` in the fused-RMSNorm NT-store path. It is x86-only,
  so the ARM64 build failed with "use of undeclared identifier '_mm_sfence'". The guard around the
  NT-store loop did not cover the fence after the loop. Fixed by wrapping the fence in the same
  BNB_AVX2_CODE/x86 guard.

Verification after the fix:
  * all three TUs compile for aarch64 (cpu_ops, cpu_gdn, pythonInterface);
  * static musl ARM64 selftest runs under qemu-aarch64: 5/5 PASS;
  * x86 rebuild + selftest still 5/5 (no regression).

Docs: report 7.4, QUICKSTART 2A.2 (zh/en), TERMUX_GUIDE new regression section.
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
TERMUX = os.path.join(REPO, 'bitsandbytes', 'TERMUX_GUIDE.md')
GITIGNORE = os.path.join(REPO, '.gitignore')


def patch_doc(path, old, new, label):
    text = io.open(path, encoding='utf-8').read()
    if text.count(old) != 1:
        print('ABORT %-46s %s anchor=%d' % (os.path.basename(path), label, text.count(old)))
        return False
    io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
    print('ok    %-46s %s' % (os.path.basename(path), label))
    return True


REPORT_OLD = (
    '**Android / ARM64（Termux）。** 通过 `build_termux.sh` 完成移植，NEON 128-bit 路径（**无 AVX2**）'
    '经 C 层自检 4/4 通过。\n'
)
REPORT_NEW = (
    '**Android / ARM64（Termux）。** 通过 `build_termux.sh` 完成移植，NEON 128-bit 路径（**无 AVX2**）'
    '经 C 层自检 5/5 通过。**2026-10-18 回归**：在 x86 上做 aarch64 交叉编译（zig clang）时发现并修复'
    '一处 **ARM64 编译阻断**——fused RMSNorm 的 NT-store 路径里 `_mm_sfence()`（x86 专属 intrinsic）'
    '未加架构守卫，ARM 构建会直接报 `use of undeclared identifier \'_mm_sfence\'`；修复后三个 TU 全部'
    '通过 aarch64 编译，静态 musl ARM64 自检在 `qemu-aarch64` 下 **5/5 PASS**，同轮 x86 重建仍 5/5'
    '（无回归）。注：AVX-512 / generic 产物是 **x86-only**；ARM64 始终加载单份常规 `.so`'
    '（`avx2_supported()` 在 ARM 上为 False，generic 缺失时安全回退）——Termux 路径不受近期 x86 '
    '改动影响。\n'
)

QS_ZH_OLD = (
    '产物为 `bitsandbytes/libbitsandbytes_cpu.so`。Termux 的 clang 对 `-mcpu=native` 支持不稳定，'
    '脚本会先试编译再决定是否启用；OpenMP 需要单独 `pkg install libomp`，没有时自动降级为单线程。\n'
)
QS_ZH_NEW = QS_ZH_OLD + (
    '\n'
    '**2026-10-18 回归**：aarch64 交叉编译 + 静态 ARM64 自检在 `qemu-aarch64` 下 **5/5 PASS**；'
    '同轮发现并修复一处 ARM64 编译阻断（fused RMSNorm 的 `_mm_sfence()` 未加架构守卫）。'
    'x86 的 AVX-512 / generic 产物与 Termux 无关，ARM64 始终使用常规 `.so`。\n'
)

QS_EN_OLD = (
    'The artefact is `bitsandbytes/libbitsandbytes_cpu.so`. Termux\'s clang handles `-mcpu=native`\n'
    'unreliably, so the script probe-compiles before enabling it; OpenMP needs a separate\n'
    '`pkg install libomp` and the build degrades to single-threaded without it.\n'
)
QS_EN_NEW = QS_EN_OLD + (
    '\n'
    '**2026-10-18 regression**: aarch64 cross-compilation of all three TUs plus a static ARM64\n'
    'selftest under `qemu-aarch64` (**5/5 PASS**); the same pass found and fixed an ARM64 compile\n'
    'blocker (an unguarded `_mm_sfence()` in the fused RMSNorm NT path). The x86-only AVX-512 /\n'
    'generic artefacts are irrelevant to Termux: ARM64 always loads the single regular `.so`.\n'
)

TERMUX_SECTION = '''
---

## 八、跨平台回归（2026-10-18）：aarch64 交叉编译 + qemu 自检

这一轮不改 Termux 脚本，只做**回归验证**：在 x86 开发机上用 zig 自带的 clang 交叉编译当前源码，
再用 `qemu-aarch64` 跑静态 ARM64 自检。

**发现的真实 bug（ARM64 编译阻断）**：`csrc/cpu_ops.cpp` 的 fused RMSNorm NT-store 路径里
`_mm_sfence()`（x86 专属 intrinsic）**没有架构守卫**——NT-store 循环有守卫，但循环之后的那句
fence 落在守卫外面。aarch64 编译直接失败：

```
error: use of undeclared identifier '_mm_sfence'
```

**修复**：给 fence 套上与 NT-store 循环相同的 `BNB_AVX2_CODE && x86` 守卫。

**修复后验证**：

| 检查 | 结果 |
|---|---|
| 三个 TU 的 aarch64 交叉编译（cpu_ops / cpu_gdn / pythonInterface） | 全部通过 |
| 静态 musl ARM64 自检（`qemu-aarch64`，无 root） | **5/5 PASS** |
| x86 同源码重建 + 自检（回归） | **5/5 PASS** |

复现脚本：`arm64_selftest.sh`（下载 qemu-user .deb 到 /tmp 解包，无需 root）。

**边界**：这是 **aarch64 架构级**验证（glibc/musl + qemu），**不是** Android/bionic 真机验证；
真机（nova 5i Pro）此前的测试结论仍成立。Termux 的 `build_termux.sh` 不受影响，但**下一次
Termux 构建会包含这次 fence 修复**。
'''

EDITS = [
    (REPORT, REPORT_OLD, REPORT_NEW, 'report 7.4 Termux paragraph'),
    (os.path.join(DOCS, 'QUICKSTART.md'), QS_ZH_OLD, QS_ZH_NEW, 'quickstart zh termux note'),
    (os.path.join(DOCS, 'QUICKSTART_EN.md'), QS_EN_OLD, QS_EN_NEW, 'quickstart en termux note'),
]

ok = True
for path, old, new, label in EDITS:
    ok &= patch_doc(path, old, new, label)

# append the regression section to the Termux guide
termux = io.open(TERMUX, encoding='utf-8').read()
if '## 八、跨平台回归（2026-10-18）' not in termux:
    io.open(TERMUX, 'w', encoding='utf-8', newline='').write(termux.rstrip('\n') + '\n' + TERMUX_SECTION)
    print('ok    TERMUX_GUIDE.md regression section')
else:
    print('skip  TERMUX_GUIDE.md already has the section')

# ignore cross-compile output
gi = io.open(GITIGNORE, encoding='utf-8').read()
if 'build_arm64/' not in gi:
    io.open(GITIGNORE, 'w', encoding='utf-8', newline='').write(gi.rstrip('\n') + '\nbuild_arm64/\n')
    print('ok    .gitignore: build_arm64/')

# mirror docs_cpu to the repo root copy
names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
         'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
         'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
for n in names:
    io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
print('ok    root docs_cpu/ mirrored')
sys.exit(0 if ok else 1)
