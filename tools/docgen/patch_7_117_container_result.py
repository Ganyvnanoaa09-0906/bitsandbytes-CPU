"""Record the three-artifact e2e on the AVX-512 container: report 7.117 and the docs.

The run confirmed on real AVX-512 hardware:
  * three artifacts built (AVX2 256,360 B zmm 0 / AVX-512 284,520 B zmm 1196 / generic 131,896 B
    AVX 0);
  * doctor three-way decision correct, will load the AVX-512 artifact;
  * loader picks avx512 / forced AVX2 / forced generic as expected;
  * package selftest 0 failures with the AVX-512 library, portable selftest 5/5;
  * loader-level timing 4.167 vs 4.201 ms (1.01x, memory-bound kernels as in 7.113);
  * qemu-user was skipped (container lacks passwordless sudo) -- a permission limit, not a failure.
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

RPT_ANCHOR = (
    '| e2e 脚本 | 已扩成 10 段（三产物 + 三级选路 + 便携 selftest + 可选 qemu-user `-cpu Nehalem`） |\n'
)
RPT_ADD = RPT_ANCHOR + (
    '\n'
    '**真机 e2e（AVX-512 容器，2026-10-18，commit `8dcb085`）**：三产物一次构建成功——\n'
    'AVX2 256,360 B（zmm **0**）/ AVX-512 284,520 B（zmm **1196**）/ generic 131,896 B\n'
    '（AVX 指令 **0**）；`doctor` 三级判定正确（AVX-512/AVX2 均 yes、两库 present、will load\n'
    '`_avx512.so`）；加载器 auto→avx512、`BNB_CPU_DISABLE_AVX512=1`→`_cpu.so`、\n'
    '`BNB_CPU_FORCE_GENERIC=1`→`_generic.so`；包 selftest **0 failure**，便携版 selftest **5/5 PASS**；\n'
    '加载器级单步 auto 4.167 ms vs forced AVX2 4.201 ms（**1.01×**，memory-bound 内核，与 §7.113 一致）。\n'
    'qemu 段因容器无密码 sudo、装不上 qemu-user 而跳过（**权限限制，非测试失败**）——静态证据\n'
    '（generic 产物 0 条 AVX + 便携 selftest 通过）成立；有 privileged 环境或 CI 时可补跑。\n'
)
RPT_TAIL_OLD = (
    '1. **真机/模拟端到端**：在容器跑新 e2e（`avx512/dual_lib_e2e.sh`）——其中 qemu-user 段会尝试\n'
    '   `-cpu Nehalem` 动态执行便携 selftest；若容器无密码 bash sudo 则跳过，静态证据仍在。\n'
)
RPT_TAIL_NEW = (
    '1. ~~真机/模拟端到端~~ **已完成（2026-10-18，AVX-512 容器）**：三产物构建 + 三级选路 + 双\n'
    '   selftest 全部通过；qemu-user 段因容器无密码 sudo 跳过（权限限制，非失败），有 privileged\n'
    '   环境或 CI 时补跑 `-cpu Nehalem` 动态验证。\n'
)

DOC_ZH_ANCHOR = (
    '`BNB_CPU_FORCE_GENERIC=1` 时 `doctor`/`LOADED_NATIVE_LIB` 选 generic，否则按 CPU 能力选\n'
    'AVX2/AVX-512。⚠️ Windows 目前只有 AVX2 DLL。详见 §7.117。\n'
)
DOC_ZH_ADD = DOC_ZH_ANCHOR + (
    '\n'
    '**真机验证（AVX-512 容器，2026-10-18）**：一次构建三产物（AVX2 256 KB / AVX-512 284 KB /\n'
    'generic 132 KB），指令审计 zmm **0 / 1196**、generic **AVX 0**；`doctor` 与\n'
    '`LOADED_NATIVE_LIB` 的三级选择全部符合预期；包 selftest 0 failure、便携版 selftest 5/5 PASS。\n'
)
DOC_EN_ANCHOR = (
    'Windows ships only the AVX2 DLL. Details in 7.117.\n'
)
DOC_EN_ADD = DOC_EN_ANCHOR + (
    '\n'
    '**Real-machine validation (AVX-512 container, 2026-10-18)**: one build produced all three\n'
    'artifacts (AVX2 256 KB / AVX-512 284 KB / generic 132 KB); instruction audit: zmm **0 / 1196**,\n'
    'generic **0 AVX**; doctor and LOADED_NATIVE_LIB agreed on all three choices; package selftest\n'
    '0 failures, portable selftest 5/5 PASS.\n'
)


def main() -> int:
    report = io.open(REPORT, encoding='utf-8').read()
    for old, new, label in ((RPT_ANCHOR, RPT_ADD, '7.117 e2e result'),
                            (RPT_TAIL_OLD, RPT_TAIL_NEW, '7.117 todo 1')):
        if report.count(old) != 1:
            print('ABORT report %s anchor=%d' % (label, report.count(old)))
            return 1
        report = report.replace(old, new, 1)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(report)
    print('ok    report 7.117 updated')

    for name, old, new in (('TECH_REPORT.md', DOC_ZH_ANCHOR, DOC_ZH_ADD),
                           ('TECH_REPORT_EN.md', DOC_EN_ANCHOR, DOC_EN_ADD)):
        path = os.path.join(DOCS, name)
        text = io.open(path, encoding='utf-8').read()
        if text.count(old) != 1:
            print('ABORT %s anchor=%d' % (name, text.count(old)))
            return 1
        io.open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    %s real-machine note' % name)

    names = ['QUICKSTART.md', 'QUICKSTART_EN.md', 'TECHNICAL_GUIDE.md', 'TECHNICAL_GUIDE_EN.md',
             'TECH_REPORT.md', 'TECH_REPORT_EN.md', 'API_REFERENCE.md', 'API_REFERENCE_EN.md',
             'MOTION_BOUNDARY.md', 'MOTION_BOUNDARY_EN.md']
    for n in names:
        io.open(os.path.join(ROOT, n), 'wb').write(io.open(os.path.join(DOCS, n), 'rb').read())
    print('ok    root docs_cpu/ mirrored')
    return 0


if __name__ == '__main__':
    sys.exit(main())
