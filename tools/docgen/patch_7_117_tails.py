"""Close the two remaining tails in the report/docs: qemu pre-AVX2 dynamic run + Windows portable DLL.

Evidence gathered on 2026-10-18:
  * qemu-user was installed WITHOUT root by downloading the .deb from the apt mirror and
    `dpkg-deb -x` extracting it into /tmp; `qemu-x86_64 -cpu Nehalem ./selftest_generic` (an
    emulated CPU with no AVX/AVX2) ran the portable selftest 5/5 PASS.
  * Windows portable DLL was cross-built with zig's bundled clang (x86_64-windows-gnu,
    -DBNB_CPU_PORTABLE=1): 0 AVX instruction lines in all three TUs, and on this Windows host the
    DLL loads via ctypes and a quantize->dequantize kernel round-trips at 0.39% relative error.
  * the MSVC path for the same artifact is wired into build_manual.bat (BNB_BUILD_GENERIC=1, no
    /arch:AVX2, /DBNB_CPU_PORTABLE) with export_generic.def and release packaging; it needs VS/CI.
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

RPT_OLD_TODO = (
    '1. ~~真机/模拟端到端~~ **已完成（2026-10-18，AVX-512 容器）**：三产物构建 + 三级选路 + 双\n'
    '   selftest 全部通过；qemu-user 段因容器无密码 sudo 跳过（权限限制，非失败），有 privileged\n'
    '   环境或 CI 时补跑 `-cpu Nehalem` 动态验证。\n'
    '2. **Windows generic DLL** 未提供（MSVC 不加 `/arch:AVX2` 可编，但本机无 VS 无法验证）；\n'
)
RPT_NEW_TODO = (
    '1. ~~真机/模拟端到端~~ **已完成**：AVX-512 容器三产物 + 三级选路 + 双 selftest 全通过；\n'
    '   **qemu-user 预 AVX2 动态验证也完成**——无 root 安装：从 apt 镜像 `apt-get download` 取\n'
    '   qemu-user 的 .deb、`dpkg-deb -x` 解包，`qemu-x86_64 -cpu Nehalem ./selftest_generic`\n'
    '   （模拟无 AVX/AVX2 的 CPU）→ **5/5 PASS**，`QEMU_PRE_AVX2_PASS`。\n'
    '2. ~~Windows generic DLL~~ **已闭环（无 MSVC 路线）**：用 zig 自带 clang 交叉编译\n'
    '   `x86_64-windows-gnu` portable DLL（三 TU 反汇编 **0 条 AVX 指令**），在本机 Windows 上\n'
    '   ctypes 加载并跑量化往返内核（相对误差 **0.39%** < 0.6% 判据）→ **KERNEL_OK**。\n'
    '   MSVC 官方产物入口已就位：`build_manual.bat` 的 `BNB_BUILD_GENERIC=1`（不加 `/arch:AVX2`、\n'
    '   `/DBNB_CPU_PORTABLE`）、`export_generic.def`、发布脚本复制；官方编译仍需 VS/CI（本机无 VS）。\n'
)

RPT_APPEND = (
    '\n'
    '**两条尾巴的验证（2026-10-18）**：\n'
    '\n'
    '| 尾巴 | 做法 | 结果 |\n'
    '|---|---|---|\n'
    '| qemu 预 AVX2 动态执行 | `apt-get download qemu-user` + `dpkg-deb -x`（**无需 root**），\n'
    '`qemu-x86_64 -cpu Nehalem` 跑便携 selftest | **5/5 PASS**（模拟 CPU 无 AVX/AVX2） |\n'
    '| Windows generic DLL | zig 自带 clang 交叉编译 `x86_64-windows-gnu`，`-DBNB_CPU_PORTABLE=1`；\n'
    '本机 Windows ctypes 加载 + 量化往返内核 | 反汇编 **0 条 AVX** · 往返误差 **0.39%** → KERNEL_OK |\n'
    '\n'
    '**⇒ ★【Linux 与 Windows 两条"非 AVX2 可运行"都有实测证据：Linux 是 e2e（容器三产物/三级选路）\n'
    '+ qemu 模拟老 CPU；Windows 是交叉编译产物 + 本机加载执行。剩下唯一没做的是【用 MSVC 编官方\n'
    'DLL】——本机没有 VS，入口已写好，留给有 VS 的机器/CI。】**\n'
)

DOC_ZH_ANCHOR = (
    '**真机验证（AVX-512 容器，2026-10-18）**：一次构建三产物（AVX2 256 KB / AVX-512 284 KB /\n'
    'generic 132 KB），指令审计 zmm **0 / 1196**、generic **AVX 0**；`doctor` 与\n'
    '`LOADED_NATIVE_LIB` 的三级选择全部符合预期；包 selftest 0 failure、便携版 selftest 5/5 PASS。\n'
)
DOC_ZH_ADD = DOC_ZH_ANCHOR + (
    '\n'
    '**非 AVX2 的动态验证（2026-10-18）**：无 root 装 qemu-user（下载 .deb + `dpkg-deb -x`），\n'
    '`qemu-x86_64 -cpu Nehalem`（模拟无 AVX/AVX2 的 CPU）跑便携 selftest → **5/5 PASS**；\n'
    'Windows 侧用 zig 自带 clang 交叉编译 portable DLL（三 TU **0 条 AVX**），本机 ctypes 加载并\n'
    '跑量化往返内核（**0.39%** 误差）→ OK。MSVC 官方 DLL 入口在 `build_manual.bat` 的\n'
    '`BNB_BUILD_GENERIC=1`（本机无 VS，未在本机编译）。\n'
)
DOC_EN_ADD = (
    '\n'
    '**Dynamic validation on non-AVX2 (2026-10-18)**: qemu-user installed without root (download the\n'
    '.deb, `dpkg-deb -x`), then `qemu-x86_64 -cpu Nehalem` (an emulated CPU without AVX/AVX2) ran the\n'
    'portable selftest **5/5 PASS**; on Windows, a portable DLL cross-built with zig\'s clang (three TUs,\n'
    '**0 AVX instructions**) loads through ctypes and round-trips a quantize kernel at **0.39%** error.\n'
    'The MSVC entry point for the official DLL is `BNB_BUILD_GENERIC=1` in `build_manual.bat` (this\n'
    'machine has no VS, so that build is not verified here).\n'
)

DOC_EN_ANCHOR = (
    '**Real-machine validation (AVX-512 container, 2026-10-18)**: one build produced all three\n'
    'artifacts (AVX2 256 KB / AVX-512 284 KB / generic 132 KB); instruction audit: zmm **0 / 1196**,\n'
    'generic **0 AVX**; doctor and LOADED_NATIVE_LIB agreed on all three choices; package selftest\n'
    '0 failures, portable selftest 5/5 PASS.\n'
)


def main() -> int:
    report = io.open(REPORT, encoding='utf-8').read()
    if report.count(RPT_OLD_TODO) != 1:
        print('ABORT report todo anchor=%d' % report.count(RPT_OLD_TODO))
        return 1
    report = report.replace(RPT_OLD_TODO, RPT_NEW_TODO, 1)
    # Append the validation block at the end of 7.117.5 (before the section separator that follows).
    tail_anchor = (
        '3. 发布脚本已能携带 generic `.so`（`build_release_linux.sh` 检测到就复制），默认仍只带 AVX2，\n'
        '   保守默认不变。\n'
    )
    if report.count(tail_anchor) != 1:
        print('ABORT report tail anchor=%d' % report.count(tail_anchor))
        return 1
    report = report.replace(tail_anchor, tail_anchor + RPT_APPEND, 1)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(report)
    print('ok    report 7.117 tails closed')

    for name, old, new in (('TECH_REPORT.md', DOC_ZH_ANCHOR, DOC_ZH_ADD),
                           ('TECH_REPORT_EN.md', DOC_EN_ANCHOR, DOC_EN_ADD)):
        path = os.path.join(DOCS, name)
        text = io.open(path, encoding='utf-8').read()
        if text.count(old) != 1:
            print('ABORT %s anchor=%d' % (name, text.count(old)))
            return 1
        io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %s tails note' % name)

    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')
    return 0


if __name__ == '__main__':
    sys.exit(main())
