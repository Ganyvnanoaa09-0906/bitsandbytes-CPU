"""Append 7.70: the C latency test works, the DRAM figure is genuinely elevated, and I overcorrected.

In 7.69 I invalidated the 259 ns latency figure by showing the Python loop measures the interpreter,
and concluded that the claim 'the memory subsystem is misbehaving' had no evidence behind it. The C
version now separates the cache levels properly, and the DRAM figure it gives is 119 ns against a
normal 70 to 90 -- so the qualitative claim was right and only the number was wrong. That needs
stating plainly, because 7.69.6 overstated the correction.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.70 C 延迟测试可用；DRAM 确实偏高；而我在 7.69 里纠正过头了

#### 7.70.1 C 版本的延迟（`mem_true.cpp`，clang -O2）

| 工作集 | 延迟 ns | **周期** | 理论预期 |
|---|---|---|---|
| 寄存器链（下限） | 0.02 | **0.06** | ~1 周期/跳 |
| **L1 16 KB** | 1.26 | **2.99** | ~4~5 |
| L2 256 KB | 3.39 | **8.04** | ~12~15 |
| **L3 4 MB** | 20.96 | **49.79** | ~40~50 |
| **DRAM 256 MB** | **119.47** | **283.75** | ~170~200 |

**⇒ 下限归零、层级能分开、L1 与 L3 都落在预期内 ⇒ 这个测试【可用】。**

#### 7.70.2 ★ 而 DRAM 那条：我的推翻推翻了自己

```
Python 测出 259 ns  ⇒ 7.69 判定"那是解释器开销,作废"
C 测出    119 ns    ⇒ 而正常 DDR4-2667 是 70~90 ns
⇒ ★【119 对 75 ≈ 【1.6 倍 —— 偏高的【定性判断是真的】】
```

**⇒ 所以原来的结论【方向对、数字错】；而 7.69.6 那句"这个判断从头到尾没有证据"是【过头的】。**

**⇒ 正确的表述是：**延迟确实偏高，幅度是 1.6 倍而不是 3~4 倍；
**而那个 3~4 倍是 Python 开销（约 214 ns/次）叠加在一个真实的约 119 ns 上造成的。**

**⇒ 而这条本身也是一个教训：纠错时容易【连带否掉正确的部分】，
因为"这个数字是假的"和"这个结论是假的"是两件事。**

#### 7.70.3 而带宽那半【仍然坏着】

```
读      2684334 GB/s   ← ★ 荒谬(2.7 PB/s) ⇒ 编译器把求和提出循环了
写           9.50 GB/s
复制        17.88 GB/s  ⇒ 双通道理论的 42%
```

**⇒ 读那一条作废（和 7.69 里的读一样的问题：结果没被真正依赖）。**

**⇒ 而复制 17.88 是【单线程】的 ⇒ 单线程本来就到不了双通道上限 ⇒ 【通道数这个问题它答不了】。**

**⇒ 需要多线程版（6 线程），而对齐 Python 版测到的 27 GB/s —— 那个数字来自 `mem_diagnostic.py`，
而那个文件的 stride 部分已确认是坏的 ⇒ 所以它的带宽部分仍待独立复验。**

#### 7.70.4 所以内存这件事的当前状态

| 项 | 状态 |
|---|---|
| **缓存层级延迟** | ✓ **可信**（C 版，层级分明） |
| **DRAM 延迟** | ✓ **可信：119 ns，偏高 1.6 倍** |
| ~~Python 版 259 ns~~ | ✗ 作废（含 214 ns 解释器开销） |
| ~~读带宽~~ | ✗ 作废（编译器提出循环） |
| **复制带宽（单线程）** | 17.88 GB/s |
| **多线程带宽 / 通道数** | ★ **仍未测**（而它决定"27 GB/s"这个分母可不可信） |

#### 7.70.5 而 DRAM 延迟偏高 1.6 倍这件事，指向什么

**⇒ 一个可测的候选：内存通道。** 若两条 8 GB 里只有一条在跑，
* 带宽减半
* **而延迟【不会】减半**（延迟由时序决定，与通道数无关）

**⇒ 所以"延迟偏高"和"通道数"是两件事，需要分别测。**

**⇒ 另一个候选：BIOS 里的时序没有跑在 XMP/标称值上。** 而 7.41.3 已记录
`WMI: Speed = ConfiguredClockSpeed = 2667` ⇒ **频率是标称的，而【时序不在 WMI 里】。**

**⇒ 所以下一步：用 C 多线程量带宽（定通道数），并读 SPD 拿真实时序。**

'''

m = re.search(r'^### 7\.70 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.70'
else:
    a = re.search(r'^# 附录', s, re.M)
    if a:
        s = s[:a.start()] + SEC + '\n\n' + s[a.start():]
        how = '插在附录之前'
    else:
        s = s.rstrip() + '\n\n' + SEC
        how = '追加'
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('  %s；报告 %d 行' % (how, s.count('\n')))
