"""Append 7.71: memory latency is normal. The elevated figures were Python overhead and TLB misses.

Third correction on the same item. 7.69 showed the Python measurement reads the interpreter, 7.70
swept the working set. The sweep separates two effects that the earlier single-size measurements
conflated, and once separated the DRAM figure is ordinary.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.71 内存延迟正常 —— 三次测量，三次都是测量本身的假象

#### 7.71.1 工作集扫描把两个现象分开了

| 工作集 | 延迟 ns | 周期 |
|---|---|---|
| L1 16 KB | 1.26 | 3.00 |
| L2 256 KB | 3.14 | 7.46 |
| **L3 4 MB** | **21.79** | **51.74** |
| **8 MB** | ★ **65.13** | ★ **154.68** |
| 16 MB | 83.49 | 198.28 |
| 32 MB | 93.31 | 221.62 |
| 64 MB | 103.58 | 246.01 |
| 128 MB | 110.68 | 262.88 |
| **256 MB** | **119.87** | **284.70** |

#### 7.71.2 而曲线里叠着两个完全不同的现象

```
① 4MB → 8MB:  21.79 → 65.13 ns（跳 3 倍）
   ⇒ ★【这是真正的 L3 → DRAM 跨越】（L3 是 8 MB 共享,而 4 MB 装得下）

② 8MB → 256MB: 65.13 → 119.87 ns（涨 1.8 倍,渐变）
   ⇒ ★【这是 TLB】:8 MB 是 2048 页,256 MB 是 65536 页,
      而 TLB 只有约 1500~3000 项 ⇒ 大工作集下【几乎每次都是 TLB miss】
   ⇒ 每次 TLB miss 要一次页表遍历（约 20~40 ns）⇒ 与观测的增量吻合
```

**⇒ ★【所以真正的 DRAM 延迟是 65 ns —— 而 DDR4-2667 的正常区间是 70~90 ⇒【完全正常】】**
**⇒ 而 120 ns 是"大工作集下的【有效】延迟",受 TLB 限制,不是 DRAM 本身。**

#### 7.71.3 带宽：独立复验通过

| 线程 | 读 GB/s | 写 GB/s | 复制 GB/s |
|---|---|---|---|
| 1 | 15.44 | 7.76 | 12.23 |
| **2** | **24.58** | 8.84 | 13.49 |
| 6 | 24.44 | 8.50 | 12.98 |

**⇒ 读平台约 25 GB/s，而 Python 版（`mem_diagnostic.py`）报的是 27.17 ⇒ 【两者吻合 ✓】**

**⇒ ★ 而 25 > 21.3（单通道 DDR4-2667 的理论上限）⇒ 【单通道被排除 ⇒ 两条都在跑】。**

（写只有 8.5 GB/s 是 RFO 造成的：普通存储要先读缓存行 ⇒ 实际流量约翻倍 ⇒ 约 17 GB/s，与读的 24.5 一致。）

#### 7.71.4 ★ 而这是同一件事的第三次修正

| 版本 | 说法 | 状态 |
|---|---|---|
| 早期 | 延迟 259 ns（正常 70~90）⇒ 内存是片外瓶颈 | ✗ 含 214 ns Python 开销 |
| **7.69** | "这个判断从头到尾没有证据" | △ **过头了** |
| **7.70** | "119 ns 确实偏高 1.6 倍" | △ **把 TLB 算成了 DRAM** |
| **7.71** | **65 ns ⇒ 正常。内存没有异常** | ✓ **本次** |

**⇒ 三次的根源是同一个：【在没扫自变量的情况下下结论】。**

```
第一次 ⇒ 没排除 [解释器开销] 这个自变量
第二次 ⇒ 没排除 [TLB 覆盖范围] 这个自变量
第三次 ⇒ 扫了工作集 ⇒ 曲线立刻把两个现象分开
```

**⇒ 而这一条与 7.62 的核显是同一条教训的第二次出现**：
**那里是"CPU 强度"没被扫（扫出来是一条悬崖，说明是开关）；这里是"工作集大小"没被扫
（扫出来是两条斜率，说明是两个机制）。**

**⇒ 所以"把自变量扫出来"不是一条关于测量的建议，而是【唯一能把混在一起的机制拆开的手段】**。

#### 7.71.5 所以内存这条线关闭

| 项 | 最终值 | 判断 |
|---|---|---|
| 缓存层级延迟 | L1 3 · L2 7 · L3 52 周期 | ✓ 正常 |
| **DRAM 延迟** | **65 ns** | ✓ **正常** |
| 有效延迟（256 MB 随机） | 120 ns | TLB 限制，非 DRAM |
| **读带宽** | **~25 GB/s（两版吻合）** | ✓ |
| 通道数 | **双通道**（25 > 21.3） | ✓ |
| ~~"内存是片外瓶颈"~~ | **✗ 无依据** | **三次测量均为假象** |

**⇒ 而 GEMM 是计算受限这一点不变（算术强度 300 对门槛 9）——
所以内存这条线【既不支持也不否定】此前那些结论,它只是被排除为一个【嫌疑】。**

'''

m = re.search(r'^### 7\.71 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.71'
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
