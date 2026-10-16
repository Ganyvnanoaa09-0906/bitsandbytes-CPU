"""Append 7.69: the 259 ns latency figure is an artefact of the test, and thread scaling is the clock.

The latency measurement has been cited since early in the project as evidence that the memory
subsystem is misbehaving: 259 ns against a normal 70 to 90 for DDR4-2667. It was never audited, while
the stride test beside it was found broken in 7.41.3. Auditing it now shows it measures the Python
interpreter rather than memory.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.69 ★ 延迟 259 ns 是测试假象；线程扩展是频率不是内存

#### 7.69.1 逐级延迟测量

| 工作集 | 延迟 ns | 每跳周期 | 理论预期 |
|---|---|---|---|
| **L1 16 KB** | **214.5** | **509.5** | **~4~5 周期** |
| L2 256 KB | 213.7 | 507.6 | ~12~15 |
| L3 4 MB | 266.8 | 633.7 | ~40~50 |
| DRAM 256 MB | 332.7 | 790.2 | ~170~200 |

**★ L1 对 DRAM 的比值 = 【1.6×】（正常应 15~20×）。**

#### 7.69.2 所以它量的是 Python,不是内存

```
L1 只有 16 KB、应该是 4~5 个周期 ⇒ 而测出 509 个周期 ⇒ 【高 100 倍】
⇒ 根因:p = int(nxt[p]) 每次都要过 numpy 索引 + Python 对象转换
   ⇒ 约 214 ns/次 —— 而它被记成了"L1 延迟"
```

**⇒ 所以"延迟 259 ns（正常 70~90）"【不成立】。**

**⇒ 而它此前被用来支撑"内存是片外瓶颈"这个判断 —— 那个判断因此失去依据。**

#### 7.69.3 扣掉固定开销后仍然不成立

| 层级 | 实测 − 214 ns | 理论 |
|---|---|---|
| L1 | **~0** | ~2 ns |
| L2 | **~0** | ~5 ns |
| L3 | ~53 | ~20 ns |
| DRAM | ~118 | ~70~90 ns |

**⇒ L1 与 L2 【没有差别】⇒ 这个测试【连层级都分不出来】⇒ 分辨率不够,整条作废。**

**⇒ 正确的做法要用 C 写（而本项目现在有 `D:\\vs` 的 clang 可用）。**

#### 7.69.4 而线程扩展那 2.87× 不是内存

| 线程 | GFLOPS | 相对 1 线程 |
|---|---|---|
| 1 | 77.2 | 1.00× |
| 2 | 141.2 | 1.83× |
| **3** | **143.0** | **1.85×** ← 2→3 完全平坦 |
| 4 | 181.4 | 2.35× |
| 5 | 213.7 | 2.77× |
| 6 | 221.9 | **2.87×** |

**⇒ 而 2048³ 的算术强度是 341 FLOP/字节 ⇒ 只需 0.65 GB/s ⇒ 【不是带宽限制】。**

**⇒ 最可能是【全核频率】**：

```
单核可 Boost 到约 3.7 GHz,而 6 核满载掉到 2.375（15W 预算）
⇒ 6 × 2.375/3.7 = 3.85×  ⇒ 而实测 2.87× ⇒ 剩下的损失在别处
```

#### 7.69.5 两条一起看

**⇒ 7.68.6 里说"线程扩展饱和与延迟 259 ns 可能同源"—— 现在两条都不成立了：**

* **延迟那条是测试假象（本节）**
* **线程扩展那条是频率，而频率是 15W 预算的结果（与功耗仲裁同源）**

**⇒ 所以"内存子系统有问题"这个判断【从头到尾没有证据】——
而它此前是从一个坏掉的 stride 测试、加一个测错对象的延迟测试推出来的。**

#### 7.69.6 而这是第十七次同形错误，而这次它改了一个【贯穿全篇】的前提

**此前多处（7.47.2、7.52.2、7.62 等）写过"内存是片外瓶颈""只需 2 GB/s 对 27 GB/s"这类话 ——
而后者的**分母 27 GB/s 来自 `mem_diagnostic.py`，那个测试【也是同一作者写的】，
**⇒ 而它的 stride 部分已确认是坏的 ⇒ 那么它的【带宽部分】也需要重新审视。**

**⇒ 所以下一步很明确：用 C 重写内存测试（延迟 + 带宽 + 通道数），
把"内存子系统到底什么水平"这件事一次定下来 —— 因为它上面压着好几条结论。**

'''

m = re.search(r'^### 7\.69 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.69'
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
