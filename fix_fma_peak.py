"""Fix the report's self-contradiction about this machine's FP32 peak.

Two mutually exclusive peaks are used in the same document, and one of them is used to conclude that
the GEMM has 19% of recoverable headroom. Measured, the answer is unambiguous.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

OLD = """**Zen2/Renoir 的 FPU 是 128-bit 宽**，因此每周期 1 个 256-bit FMA，6 核 @ 2.375 GHz 的理论上限是 **228 GFLOPS**；实测 L2 内到 371（超出该模型，说明单核可双发），而**不是**早期误算的 537（那假设了每周期两个 256-bit FMA）。"""

NEW = """**★ 2026-10-16 更正：这段文字原本写"FPU 是 128-bit 宽 ⇒ 每周期 1 个 256-bit FMA ⇒
理论上限 228 GFLOPS"，而它与本文档后面三处（L5743/L5841/L6325 用 456 峰值）矛盾。
实测判定：【每周期 2 个 256-bit FMA】，456 GFLOPS @2.375 GHz 是对的，"228" 是错的。**

```
实测（fma_width.cpp，8 条独立累加链，asm 屏障强制物化，无访存）:
    6 线程纯 FMA ⇒ 【500.2 GFLOPS】

而 R5-4500U 的标称最大 boost 是 4.0 GHz ⇒ 硬界:
    每周期 1 个: 上限 = 6 × 16 FLOP × 4.0 GHz = 【384 GFLOPS】 < 500.2 ⇒ ★【不可能】
    每周期 2 个: 上限 = 6 × 32 FLOP × 4.0 GHz = 768 GFLOPS      ⇒ 500.2 ✓
⇒ ★【判定不需要知道当时的频率 —— 它用 CPU 的标称上限做一个硬界】
⇒ 反推当时的频率 = 500.2 ÷ (6 × 32) = 【2.60 GHz】（纯 FMA 不访存，功耗低，可略高于满载的 2.375）
```

**⇒ 所以 456 GFLOPS 是正确的工作峰值，"537 是误算"那句话【本身才是误算】。**

**★ 而原文那句"实测 L2 内到 371（超出该模型，说明单核可双发）"—— 它的【方向是对的】
（它察觉了模型与实际不符），而结论却写成了"228 才是真实区间"。**
**⇒ 察觉了矛盾而没有改那个数，于是同一个错误在本文档后面又被用了三次。**
**⇒ 而这是本项目今天第四次遇到同一形状：【一个错的前提被下游引用多次】。**"""

if OLD in s:
    s = s.replace(OLD, NEW, 1)
    print('  ✓ 已更正 L8757（228 → 456，附判定过程）')
else:
    print('  ? L8757 锚点没找到')

# L5888 那条结论要标注
OLD2 = "| FMA 371 是机器上限（79.8%） | ✓ 成立 |"
NEW2 = "| FMA 371 是机器上限（79.8%） | △ **前半对、后半要改**：371 是实测上限 ✓，而 79.8% 的分母 456 是对的（见 A.7 的更正），只是**那个百分比要注明当时的频率** —— 371 对应的是 boost 频率下的比例，不是满载频率下的 |"

if OLD2 in s:
    s = s.replace(OLD2, NEW2, 1)
    print('  ✓ 已标注 L5888 那条结论')
else:
    print('  ? L5888 锚点没找到')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  报告 %d 行' % s.count('\n'))
