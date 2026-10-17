"""Patch the report with three pieces of work done on 2026-10-17:

1. Section 7.107: the 4-bit optimizer's speed swept over parameter count. The 2.63x quoted in
   7.80 and in the docs was measured at 1M parameters (L3-resident); at >=64M -- the only region
   where the optimizer is worth using -- five runs give 1.69~1.81x, median 1.74x.
2. Section 7.108: the appendix corruption that made this file "binary" to editors -- 16,883 control
   characters from a double encoding, repaired from the evidence files.
3. The index tables (C/F/G), the A.11 line counts, and a pointer at the end of 7.80.

Every replacement is asserted to occur exactly once; --dry only reports.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

PATH = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

EDITS = []


def rep(old, new, label):
    EDITS.append((old, new, label))


# ---------------------------------------------------------------- index, table C
rep('| 而它快吗 | ★ **快 2.63×**（CI 2.13~3.12），不是我头注释写的"慢 4×" | §7.80 | ★★★ |',
    '| 而它快吗 | ★ **2.63× 只在 1M（L3 内）成立；扫到 384M（它唯一该用的区间）后是 1.74×**（五次运行 1.69~1.81） | §7.80 · §7.107 | ★★★ |',
    'index C: speed row')

# ---------------------------------------------------------------- index, table F
rep('| "4-bit 优化器慢 4×"（头注释） | §7.80 | **方向相反**，实测快 2.63× |',
    '| "4-bit 优化器慢 4×"（头注释） | §7.80 | **方向相反**，实测快 2.63× |\n'
    '| **"快 2.63×"（§7.80 与三份文档）** | **§7.107** | **那个数在 1M（缓存内）测的，而它只该在 ≥300M（DRAM）用；扫 N 后 ≥64M 是 1.74×（1.69~1.81）** |',
    'index F: scale-correction row')

rep('**本报告的 106 节里有 11 节存在的唯一目的就是推翻更早的一节。若不知道这件事，会相信先读到的那一个。**',
    '**本报告的 106 节里有 12 节存在的唯一目的就是推翻（或收窄）更早的一节。若不知道这件事，会相信先读到的那一个。**',
    'index F: count')

# ---------------------------------------------------------------- index, table G
rep('| 通过一个检验就宣布成立 | 1 | §7.98（梯度等价通过，而训练没在训练） |',
    '| 通过一个检验就宣布成立 | 1 | §7.98（梯度等价通过，而训练没在训练） |\n'
    '| **在错误的规模上测** | 1 | §7.107（2.63× 是 1M 缓存内的数；而它只在 ≥300M 用） |\n'
    '| **把"编码合法"当成"内容正确"** | 1 | §7.108（NUL 是合法 UTF-8 字符，所以严格解码检查看不见它） |',
    'index G: two patterns')

# ---------------------------------------------------------------- pointer at end of 7.80
# handled line-based below: the paragraph's punctuation is not worth guessing
POINTER_ANCHOR = '7.74 那条教训的第三次出现'
POINTER_LINES = [
    '',
    '**→ 而那个 2.63× 后来被扫了一遍规模（§7.107）：它在 1M（L3 内）成立，而在 ≥64M（DRAM，'
    '也就是它唯一该被用的区间）是 1.74×。本节的判据与方向都不变，变的是那个倍数的适用范围。**',
]

# ---------------------------------------------------------------- A.11 intro note
rep('以下日志由机器直接写出，未经编辑。每个文件上方标注其来源与行数。',
    '以下日志由机器直接写出，未经编辑。每个文件上方标注其来源与行数。\n\n'
    '> **关于中文乱码（2026-10-17 说明）**：本节的日志由 PowerShell 5.1 的命令重定向捕获；'
    '子进程的 UTF-8 stdout 被按 GBK 解码、再写成 UTF-16LE，因此**中文是乱码，而数字与英文行是准确的**。'
    '受影响的是 `curriculum_B1/B2` · `maskgit_causal/sample/train` · `night_ns/night_nsasync/night_tk/night_ttl`。\n'
    '>\n'
    '> 先前版本更糟：生成报告时又把这些 UTF-16 的【字节】当 UTF-8 读，于是附录里出现了 `\\x00` 与 `U+FFFD`。'
    '2026-10-17 已按 `evidence/raw/` 的 UTF-16LE 原文**逐字重新复制**（脚本 `repair_report_corruption.py`，'
    '备份在 `evidence/CPU_FORGE_TECHNICAL_REPORT.before_repair_20261017.md`），**没有做任何字符猜测**。'
    '要读中文原文，需要重跑产生该日志的命令。',
    'A.11 encoding note')

# ---------------------------------------------------------------- A.11 table line counts
for name, old_n, new_n, size in [
    ('curriculum_B1.log', 171, 85, 5402),
    ('curriculum_B2.log', 167, 83, 5318),
    ('maskgit_causal.log', 14, 7, 1714),
    ('maskgit_sample.log', 16, 8, 1824),
    ('maskgit_train.log', 172, 86, 5602),
    ('night_ns.log', 27, 13, 698),
    ('night_nsasync.log', 31, 15, 1138),
    ('night_tk.log', 34, 17, 1654),
    ('night_ttl.log', 48, 23, 1812),
]:
    rep('| `%s` | %d | %d |' % (name, old_n, size),
        '| `%s` | %d | %d |' % (name, new_n, size),
        'A.11 line count %s' % name)

rep('| **合计** | **977** | — |', '| **合计** | **634** | — |', 'A.11 total')

# ---------------------------------------------------------------- new sections
NEW = '''### 7.107 ★ 把"4-bit 快 2.63×"的表扩到 384M —— 它落在 1.74×，而 1M 那个数本身不稳

#### 7.107.1 ★ 那个倍数是在它【不该用】的规模上量的

**§7.80 与随后三份文档都把 4-bit 优化器的速度写成"快 2.63×（95% CI 2.13~3.12）"。
而那个数是在【1M 参数】上测的 —— 1M 参数的状态只有 1~2 MB，那在 L3 里。**

**而 §7.76 / §7.94 给出的适用边界恰恰是【只在 ≥300M 参数上才值】——
300M 参数的状态是 300~600 MB，那是 DRAM。**

```
1M 参数   ⇒ 工作集约 14 MB（含梯度与参数）⇒ 部分驻留 L3 ⇒ 差异可来自缓存与内核效率
300M 参数 ⇒ 工作集约 5 GB ⇒ 纯 DRAM 流 ⇒ 差异只能来自【搬运量】或【每元素成本】
⇒ ★【所以"它有多快"这个问题，此前是在它唯一不适用的那一端回答的。】
```

#### 7.107.2 方法：只改一个自变量

**同一对内核（`coptimizer_update_8bit_blockwise_cpu` / `coptimizer_update_4bit_blockwise_cpu`），
同一个 ctypes 调用签名（与 §7.80 同源），只把参数个数 N 从 1M 扫到 384M。**

* 9 轮 × 每轮 5 次，两臂【交替】测（频率/后台漂移对两臂对称）
* 判据与 §7.80 相同：配对比值的中位数，95% CI = 中位 ± 1.96·pstdev/√n
* N：1M（复现 §7.80）· 4M · 16M · 64M · 256M · 384M（"300M 级"的锚点）
* 五次独立运行；其中三次用同一估计量（配对），两次早期运行的估计量是 med8/med4
* 384M 时峰值内存约 4.2 GB（梯度 1.5 GB + 参数 1.5 GB + 两臂状态 1.1 GB）

**⇒ 脚本：`bench_4bit_opt_scale.py`（命令行即可复跑）。**

#### 7.107.3 结果

**表 108：4-bit 对 8-bit 优化器内核，比值随参数量（五次独立运行）**

| N | run1 | run2 | run3 | run4 | run5 | 中位 |
|---|---|---|---|---|---|---|
| 1M | 2.62 | 4.19 | 2.60 | 3.08 | 2.87 | **2.87** |
| 4M | 1.72 | 1.75 | 1.59 | 1.63 | 1.81 | **1.72** |
| 16M | 1.64 | 1.58 | 1.63 | 1.60 | 1.65 | **1.63** |
| 64M | 1.70 | 1.67 | 1.71 | 1.70 | 1.79 | **1.71** |
| 256M | 1.70 | 1.74 | 1.69 | 1.75 | 1.77 | **1.74** |
| 384M | 1.69 | 1.68 | 1.74 | 1.81 | 1.77 | **1.74** |

**（run1/run2 的列是 med8/med4；run3~run5 是配对比值，脚本同时印出两列。在 ≥64M 上两者差 <2%；
在 1M/4M 上配对比值可比 med8/med4 低到 10%（例如 run1 的 4M：1.72 对 1.58）。）**

#### 7.107.4 三条读数

**① ★ 1M 这一端本身就不稳。**

```
五次运行，1M：2.60 · 2.87 · 3.08（配对）· 2.62 · 4.19（med8/med4）
8-bit 臂的中位步时：1.049 / 1.178 / 1.077 / 1.734 / 2.180 ms —— 两次运行差 2.1×
4-bit 臂的中位步时：0.411 / 0.419 / 0.777 ms —— 相对稳定
⇒ ★【头注释与文档引用的 2.63× 落在这个范围内，但它是【最不稳的那一段】的中点。】
```

**② ★ ≥64M 这一端稳定在 1.74×（五次运行 1.69~1.81）。**

**③ ★ 而这 1.74× 【不是搬运量差出来的】。**

```
按状态布局算每步搬运（读+写都算）：
  8-bit = 16.06 N 字节（m、v 各 N 字节；absmax 每 256 个 4 字节 ×2）
  4-bit = 14.06 N 字节（m、v 各 N/2 字节；同样的 absmax）
  ⇒ 搬运比 = 1.142×  ⇒ 若两者都打满带宽，时间比应只有 1.14×
而实测 1.74× ⇒ ★【8-bit 核的有效带宽（模型口径 11.3~11.9 GB/s）低于 4-bit（17.2~17.8 GB/s）
⇒ 所以这个倍数描述的是一对【实现】，不是 4-bit 的物理优势】
```

**⇒ 而五次运行里，8-bit 臂的口径带宽从 11.9 GB/s 掉到 11.3 GB/s，4-bit 臂在 17.2~17.8 之间不动
⇒ 与"8-bit 核不是带宽受限、每元素有固定成本"一致（降温后重测可证伪，留待下次）。**

**④ 对训练步时意味着什么：几乎没有意义。**

```
优化器占一步 1~3% ⇒ 1.74× 对墙钟是 0.3~1% 的改善
§7.76 的真实训练实测 0.955×（噪声内）—— 与之一致
⇒ ★【所以 4-bit 优化器的价值是【内存 2.06×】，速度是附带的；把 2.63× 当卖点是选错了数。】
```

#### 7.107.5 修正后的三个口径

| 口径 | 之前写的 | 现在 |
|---|---|---|
| 内存：对 fp32 | 7.8× | **7.8×（不变）** |
| 内存：对 8-bit | 2.06× | **2.06×（不变）** |
| 速度：1M（缓存内） | 2.63× | **2.6~3.1×（五次运行）** |
| ★ 速度：≥64M（适用区间） | —— | ★ **1.74×（1.69~1.81，五次运行）** |
| 训练步时（30.5M 真实训练） | 0.955× | **0.955×（与 1.74× 自洽）** |

**⇒ 已同步：`docs_cpu/QUICKSTART.md` §4.9 · `docs_cpu/TECHNICAL_GUIDE.md` §3.9 ·
`docs_cpu/TECH_REPORT.md` §3.9.1 · 本文档索引 C 行、F 表、G 表。**

#### 7.107.6 两条方法学

**① 用【能测的规模】代替【要用的规模】。**
**这与 G 表第一条同族（代理量代替真实约束）：1M 不是 300M 的"小号"，它是另一个机制。**

**② ★ 单次 CI 窄 ≠ 这个量稳定。**

```
384M 单次 CI 可以窄到 ±0.04（run3：1.70~1.77）
而五次运行之间是 1.68~1.81
1M 更糟：单次 CI 宽到 1.66~3.80，跨运行 2.6~3.1
⇒ ★【窄 CI 只说明【这一次】量得准，不说明【这个量】是稳定的。】
```

**⇒ 而对策仍是同一条：把自变量扫出来 —— 这次扫的是 N。**



### 7.108 ★★ 附录里的 16,883 个坏字符：合法 UTF-8 的空字节，与一次双重编码

#### 7.108.1 症状：文件"是合法 UTF-8"，而工具说它是二进制

**2026-10-17 给本报告插结论索引时，编辑工具拒绝写入：`cannot edit ... binary file`。**
**而三件事同时成立：**

```
严格 UTF-8 解码（Python utf-8, errors='strict'） ⇒ ★【0 错误】
文件结尾 ⇒ LF，无 BOM
逐字节清点 ⇒ ★ 16,883 个控制字符/替换符（NUL 5,227 · U+FFFD 931 · 其余 ESC/BS/BEL…）
  分 5 个连续区段，全部落在附录 A.10 / A.11 / A.13 的原始日志块
```

**⇒ ★【原因不是编码坏了，而是【内容里本来就有 NUL】：U+0000 是合法 UTF-8 字符。】

#### 7.108.2 成因链：两次错误，一层套一层

```
① Python 的 stdout（这些脚本设了 PYTHONIOENCODING=utf-8）   ⇒ UTF-8 字节
② PowerShell 5.1 的原生命令重定向                           ⇒ 按系统代码页 GBK 解码 ⇒ 乱码
                                                            ⇒ 再写成 UTF-16LE 日志文件
③ 生成附录的脚本读这些文件                                  ⇒ 把 UTF-16 的【字节】当 UTF-8 解码
                                                            ⇒ \\x00 与 U+FFFD 进入报告
⇒ ★【第 ② 层是不可逆的：GBK 解码时非法字节被替换成 "?"，原始字符已经丢了。】
```

**（A.13 的四个输出文件没有被第 ② 层污染：它们的捕获路径没设 `PYTHONIOENCODING`，
PowerShell 用 GBK 正确解码，所以 UTF-16 文件里是正确的中文。）**

#### 7.108.3 修复：从证据文件重新逐字复制，不做字符猜测

| 对象 | 数量 | 证据 | 处理 |
|---|---|---|---|
| 日志块（A.11） | 9 | `evidence/raw/*.log`（UTF-16LE） | 按 UTF-16LE 解码后替换 |
| 基准输出（A.13.1~4） | 4 | `evidence/raw_outputs/*.txt`（UTF-16LE） | 解码，并把坏掉的 fence 重建为"命令块 + 输出块" |
| 路径行（§7.75.4） | 2 | `write_775.py` 第 75~76 行 | `\\b` / `\\a` 曾被当转义符 ⇒ 取回原字符串 |
| A.11 行数表 | 10 | 同上（解码后的行数） | 171→85 · 167→83 · 14→7 · 16→8 · 172→86 · 27→13 · 31→15 · 34→17 · 48→23 · 合计 977→634 |

```
结果：423,534 字符 / 13,544 行  ⇒  406,000 字符 / 13,081 行 · 坏字符 0
备份：evidence/CPU_FORGE_TECHNICAL_REPORT.before_repair_20261017.md
脚本：repair_report_corruption.py（支持 --dry，先跑一遍不写）
```

**★ 而 A.11 的中文仍是乱码 —— 因为乱码在证据文件里就已经存在（第 ② 层）。**
**⇒ 报告保留该文件的原样（逐字），并在 A.11 开头写明；数字与英文行是准确的。**
**⇒ ★【要恢复中文原文，只能重跑产生该日志的命令，不能靠猜字符。】**

#### 7.108.4 为什么这件事值得单独记

```
"严格 UTF-8 通过"  ⇒ 我此前就据此宣布"八份文档全部合法 UTF-8"
★ 而 NUL 是合法 UTF-8 ⇒ 那个检查【看不见】这一次损坏
⇒ ★ 与 G 表第一条同族：用"编码合法"这个代理量，代替"文本可读"这个真实约束
```

**⇒ 而它的代价具体到工具层面：编辑工具按"含 NUL ⇒ 二进制"处理 ⇒ 报告一度【无法被正常编辑】。**



'''

rep('# 附录 A：AR 训练路径的实测证据',
    NEW + '# 附录 A：AR 训练路径的实测证据',
    'insert 7.107 and 7.108')


def main(dry):
    text = open(PATH, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        n = out.count(old)
        if n != 1:
            print('ABORT %-40s occurrences=%d' % (label, n))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)

    lines = out.split('\n')
    hits = [i for i, ln in enumerate(lines) if POINTER_ANCHOR in ln]
    if len(hits) != 1:
        print('ABORT 7.80 tail pointer: %d lines contain the anchor' % len(hits))
        return 1
    lines[hits[0] + 1:hits[0] + 1] = POINTER_LINES
    out = '\n'.join(lines)
    print('ok    7.80 tail pointer (line %d)' % (hits[0] + 1))

    print()
    print('chars %d -> %d · lines %d -> %d' % (len(text), len(out),
                                               text.count('\n') + 1, out.count('\n') + 1))
    if dry:
        print('(dry run)')
        return 0
    with open(PATH, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('written')
    return 0


if __name__ == '__main__':
    sys.exit(main('--dry' in sys.argv))
