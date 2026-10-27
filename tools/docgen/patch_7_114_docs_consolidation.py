"""Add 7.114: this round's consolidation -- which workspace components became user docs, where the
numbers live, and the two-copies docs_cpu trap.

The docs update itself belongs in docs_cpu, but the *versions and sources* of the numbers belong in the
report: the shipped docs are generated copies, the root docs_cpu was one revision ahead, and the
research report is the only place that records which numbers were measured when.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW = '''### 7.114 ★ 把工作区内的组件收进用户文档（以及一个"两个 docs_cpu"的坑）

#### 7.114.1 这轮做了什么

**用户文档（`bitsandbytes/docs_cpu/`，CN+EN 各三份）此前落后于工作区：4-bit 优化器只存在于
自动生成的 API 参考里；2/3-bit 量化完全没有文字；核显章节停在 Windows DirectML 执行器；
AVX-512 后端没有任何用户向说明。本轮把它们补进了三份指南（提交 `cfdda7a1`）：**

| 组件 | 用户文档位置 | 数字出处 | 状态 |
|---|---|---|---|
| 核显 OpenCL/CLBlast | QUICKSTART §4.11 · TECHNICAL_GUIDE §3.9 · TECH_REPORT §3.8 | §7.104~§7.110 | 实验脚本（未进 wheel） |
| 4-bit 优化器 AdamW4bit | QUICKSTART §4.12 · TECHNICAL_GUIDE §3.10 · TECH_REPORT §3.9 | §7.76 · §7.80 · §7.107 | `0.50.2.dev3` 分支发布 |
| 2/3-bit 量化 | QUICKSTART §4.13 · TECHNICAL_GUIDE §3.11 · TECH_REPORT §3.10 | §7.23.7 · 模块 docstring 的损失表 | 脚本级（未进 wheel） |
| AVX-512 后端 | QUICKSTART §4.14 · TECHNICAL_GUIDE §3.12 · TECH_REPORT §3.11 | §7.113 | 实验性 opt-in 构建 |

**`check_docs.py` 通过：zh/en 节号 42/42、21/21、27/27 对齐，表格与代码围栏完整，
所有文件严格 UTF-8。**

#### 7.114.2 一个必须记下来的坑：仓库里有两个 `docs_cpu/`

```
docs_cpu/                       ← 根目录这份是"新的、精简版"（commit 828ade3b 更新的）
bitsandbytes/docs_cpu/          ← ★ 这份才是【发布包实际打包】的
                                   （build_release_linux.sh / build_release_windows.ps1
                                     都是 cd bitsandbytes 之后 cp docs_cpu/*.md）
```

**⇒ ★【更新了根目录那份，发布包里读到的仍是旧版本 —— 本轮真正修的就是这个错位。
判断标准不是"哪份更新"，而是"build_release_*.{sh,ps1} 从哪 cp"。】**

**另一个附带发现：`API_REFERENCE.{md,_EN.md}` 由 `bitsandbytes_cpu_cli.py` 生成
（`bitsandbytes-cpu help --markdown`），而 4-bit 优化器那一节此前只存在于生成结果里、
CLI 源里没有 ⇒ 下次重新生成会丢。已先把参考文件恢复，CLI 侧补源列入待办。**

#### 7.114.3 为什么"其他测试结果"不重复搬进报告

**核对结果：用户文档里引用的每个数字都已在报告里（`6.08` / `425.3` / `152.56` / `2.87` /
`5.6206` / `1.031` 等逐条 grep 确认），所以本轮只补了这份"组件→文档→出处"的对照表，
没有把同一份数据再写一遍 —— 否则下次改一处会漏改另一处。**



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.114'),
    ('**本文是 113 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 114 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 114'),
    ('**本报告的 113 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 114 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 114'),
    ('| AVX-512 真机验证 | ✓ **6/6 内核逐位一致；峰值 fp32 2.02×、BF16 2.71×；现有内核 0.97~1.02×** | `avx512/` · §7.113 |',
     '| AVX-512 真机验证 | ✓ **6/6 内核逐位一致；峰值 fp32 2.02×、BF16 2.71×；现有内核 0.97~1.02×** | `avx512/` · §7.113 |\n'
     '| 用户文档补全（核显/4-bit 优化器/2-3-bit/AVX-512） | ✓ **三份指南 CN+EN 已同步；check_docs 通过 42/21/27 节号对齐** | `bitsandbytes/docs_cpu/` · §7.114 |',
     'H table: docs row'),
]


def main(dry: bool) -> int:
    text = open(PATH, encoding='utf-8').read()
    out = text
    ok = True
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT %-18s occurrences=%d' % (label, n))
            ok = False
            continue
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    if not ok:
        return 1
    print()
    print('chars %d -> %d · lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    if dry:
        print('(dry run)')
        return 0
    open(PATH, 'w', encoding='utf-8', newline='').write(out)
    print('written')
    return 0


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
