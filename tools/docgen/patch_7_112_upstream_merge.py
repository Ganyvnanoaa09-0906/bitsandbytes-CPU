"""Record the upstream vendor merge: fix the stale base-version claims in the docs and add §7.112.

Facts established this round:
  * the fork's vendored tree (bitsandbytes/) was imported from upstream commit 2b6cfb79
    ("Bump dev version", 2026-08-13, __version__ = 0.50.2.dev0) -- not v0.45.1 as the docs claimed;
  * upstream main is 83364904 (2026-09-03), 12 commits ahead, including the 0.50.2 release;
  * net upstream delta touching the vendored tree: 8 files (CMakeLists, README, __init__,
    triton/kernels_4bit.py, csrc/common.cuh, csrc/gemm_4bit.cu, docs/source/installation.mdx, setup.py);
  * 6 merged cleanly, 2 conflicted on version strings only -> fork version kept.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\work\bitsandbytes-CPU'
REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

DOC_EDITS = [
    ('docs_cpu/TECHNICAL_GUIDE.md',
     '> 目的：说明本 fork 在 bitsandbytes v0.45.1 基础上**改了什么、为什么改、怎么用**。',
     '> 目的：说明本 fork 在 upstream bitsandbytes（0.50.2 发布线，main `83364904`，2026-09-03）基础上'
     '**改了什么、为什么改、怎么用**。'),
    ('docs_cpu/TECHNICAL_GUIDE.md',
     '## 2. 改动总览（相对 v0.45.1）',
     '## 2. 改动总览（相对 upstream 0.50.2 / main）'),
    ('docs_cpu/TECHNICAL_GUIDE_EN.md',
     '> Purpose: explain what this fork **changed relative to bitsandbytes v0.45.1,',
     '> Purpose: explain what this fork **changed relative to upstream bitsandbytes (0.50.2 release line,\n'
     '> main `83364904`, 2026-09-03),'),
    ('docs_cpu/TECHNICAL_GUIDE_EN.md',
     '## 2. Change Overview (relative to v0.45.1)',
     '## 2. Change Overview (relative to upstream 0.50.2 / main)'),
]

NEW = '''### 7.112 ★★ 把上游 bitsandbytes 的新版本合并进来：base 2b6cfb79 → main 83364904

#### 7.112.1 先纠正 fork 文档里一个持续了两个月的错误说法

**文档（`TECHNICAL_GUIDE.md` 开头）一直写"本 fork 在 bitsandbytes v0.45.1 基础上改了什么"。
本轮为了合并先做了来源核对，结论是这句话是错的：**

```
把 fork 的 vendored tree（仓库里的 bitsandbytes/ 子目录）与上游每个 tag、main 的每个 commit
逐一做"相同 blob 计数"（路径相同且内容哈希相同）：
  0.45.1  : 27/89  相同
  0.50.0  : 121/139
  0.50.1  : 126/139  ·  ★ 上游 commit 2b6cfb79（Bump dev version, 2026-08-13）: 127/185
  上游 main（83364904）: 122/139
⇒ ★【真实 base 是 2b6cfb79（__version__ = 0.50.2.dev0），不是 0.45.1。】
   （fork 的版本号 0.50.2.dev1 也与这条吻合；0.45.1 那个说法来自更早的 fork 文档，一直没更新。）
```

**⇒ 文档已改口为"相对 upstream 0.50.2 发布线 / main 83364904"。**

#### 7.112.2 合并范围：12 个 commit，净变更 8 个文件

**上游 main 自 base 以来 12 个 commit（含 0.50.2 release、ROCm CI、Triton 量化修复、
gfx9 Wave64、sm121 分派、文档修正）。其中 3 个是 fork 刻意删掉的 dev/CI 目录
（`.github/`），**不恢复**。净落到 vendored tree 的是 8 个文件：**

| 文件 | 上游改了什么 | 结果 |
|---|---|---|
| `CMakeLists.txt` | HIP 默认架构加 gfx908 / gfx1030 | ✓ 自动合并 |
| `README.md` | 支持表加 gfx101X / CDNA 行 | ✓ 自动合并 |
| `bitsandbytes/backends/triton/kernels_4bit.py` | ★ absmax 写入加 mask（越界修复） | ✓ 自动合并 |
| `csrc/common.cuh` | ★ gfx9 全部恢复 Warp64（原来只认 CDNA） | ✓ 自动合并 |
| `csrc/gemm_4bit.cu` | ★ sm121(G10) 加进 wide-N SIMT 分派 | ✓ 自动合并 |
| `docs/source/installation.mdx` | ROCm 安装表 / NVIDIA index URL 修正 | ✓ 自动合并 |
| `bitsandbytes/__init__.py` | 只有版本号 0.50.2.dev0→0.50.3.dev0 | ⚠️ 冲突·保留 fork（0.50.2.dev1 + gdn_cpu 挂载） |
| `setup.py` | 只有版本号同样一行 | ⚠️ 冲突·保留 fork（fork 重写过 setup.py） |

**⇒ 两个冲突都只是版本号，没有实质代码冲突；fork 的功能（cpu_ops / gdn_cpu / 自定义 setup.py）
全部保持原样。**

#### 7.112.3 方法：把 vendored tree 当第三方库做 3-way merge

```
ours   = 当前 bitsandbytes/<path>
base   = 2b6cfb79:<path>
theirs = upstream/main:<path>
⇒ git merge-file 逐文件三方合并；冲突文件不落盘，保留 fork 版本
⇒ 先建备份分支 backup/pre-upstream-merge-20261017，再在特性分支上提交
```

**验证：**
```
import bitsandbytes  ⇒ version = 0.50.2.dev1 · gdn_cpu 可访问 · optim 可访问
grep 上游改动       ⇒ CMake gfx908/gfx1030 ✓ · __GFX9__ ✓ · absmax_elements×6 ✓
fork 关键文件       ⇒ csrc/cpu_ops.cpp、__init__.py、setup.py 零改动
```

**⇒ ★ 而这次合并的真正产出不是那 8 个文件，是【一条可重复的流程】：
vendored tree 的 base 现在有据可查（2b6cfb79），下次上游更新照同一套做即可。**



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW + '# 附录 A：AR 训练路径的实测证据', 'insert 7.112'),
    ('**本文是 111 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     '**本文是 112 节的【时序记录】，而这不是一个适合查阅的顺序。下面按"一个读者会问什么"重排，',
     'header: 112'),
    ('**本报告的 111 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     '**本报告的 112 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
     'F header: 112'),
    ('| 文档（中英各四份） | ✓ | `docs_cpu/` |',
     '| 文档（中英各四份） | ✓ | `docs_cpu/` |\n'
     '| 上游合并（vendor） | ✓ **base 2b6cfb79 → main 83364904**（12 commit / 8 文件，2 个纯版本号冲突保留 fork） | `bitsandbytes/` · §7.112 |',
     'H table: upstream merge row'),
]


def main(dry):
    ok = True
    # docs
    for rel, old, new in DOC_EDITS:
        path = os.path.join(ROOT, rel)
        text = open(path, encoding='utf-8').read()
        n = text.count(old)
        if n != 1:
            print('ABORT docs %s: %d occurrences' % (rel, n))
            ok = False
            continue
        if not dry:
            open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new, 1))
        print('ok    docs %s' % rel)
    # report
    text = open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT report %-20s occurrences=%d' % (label, n))
            ok = False
            continue
        out = out.replace(old, new, 1)
        print('ok    report %s' % label)
    if not ok:
        return 1
    print()
    print('report chars %d -> %d · lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    if dry:
        print('(dry run)')
        return 0
    open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('written')
    return 0


import os  # noqa: E402  (kept late so the dry-run flag logic reads top-down)

if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
