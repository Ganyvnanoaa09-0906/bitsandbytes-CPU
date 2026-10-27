"""Append 7.52 final: the four measurements, including a contradiction that must not be smoothed over.

The task-split measurement said the CPU slows 4.3% while the iGPU runs GEMM. The process-based
concurrency retest said 94.4%. They measure the same thing and they disagree by a factor of twenty.
The honest record is both numbers and the admission that the difference was not isolated, not the
flattering one.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.52 四项测量：一个可用加速，一个未解的矛盾

#### 7.52.1 ★ L2 驻留分块：1.359× —— 立刻可用的训练加速

| 分块宽度 k | 每块 MB | GFLOPS | 相对 |
|---|---|---|---|
| 128 | 1.25 | 116.2 | 0.689× |
| **256** | 2.50 | **221.9** | **1.315×** |
| **512** | 5.00 | **229.4** | ★ **1.359×** |
| 684 | 6.68 | 116.2 | 0.689× |
| 1024 | 10.00 | 152.7 | 0.905× |

**⇒ torch 的一次性 `mm` 比手工按列分块【慢 36%】，最优宽度 512（输出块 5 MB，接近 L3 的 8 MB）。**

**⇒ 换算到训练**：GEMM 占一步 52.4% ⇒ **52.4% × 36% = 整步省约 19%。**
**⇒ 不需要写内核，只需在 MLP 处按 512 列分块调用。这是今天唯一一个直接可用的加速。**

#### 7.52.2 FMA 四因：371 是机器上限

| n | GFLOPS | 占 456 峰值 |
|---|---|---|
| **384** | **364.0** | **79.8%** |
| 1024 | 275.7 | 60.5% |
| 2048 | **235.5** | **51.6%** |

**⇒ torch 的 364 与此前记录的 371 几乎相同 ⇒ 判据不成立 ⇒ 371 是机器上限，不是实现产物。**
**⇒ 掉落点落在 L2（3 MB）与 L3（8 MB）之间 ⇒ 是【容量】不是带宽。**
**⇒ 剩下 20% 是发射气泡，而 4500U 无 SMT ⇒ 结构性，指令层面无路。**

**⇒ 两处算术都确认 GEMM 是计算受限**：算术强度约 300 FLOP/字节对门槛约 9，高出 33 倍 ⇒
**7.47 里"内存限制 GEMM"的说法是错的**（那只对逐元素算子成立）。

#### 7.52.3 ★★ 而 ① 与 ④ 矛盾，且未能解释

**两次测量本量的是同一件事，而结论相差二十倍：**

| 测量 | CPU 单独 | 并发时 CPU | 拖慢 |
|---|---|---|---|
| **① 任务分流**（`igpu_task_split.py`） | 4.533 s | 4.730 s | ★ **4.3%** |
| **④ GIL 对照**（`igpu_concurrency_procs.py`） | 3.172 s | 6.165 s | ✗ **94.4%** |

**而核显侧的两次数字也不同**：① 给出 506.9 GFLOPS，④ 给出 **789.5~818.4 GFLOPS**。

**⇒ 所以"核显重开"这件事【没有定论】。**

**⇒ 而 ④ 的进程版【没有】比 0.76× 的线程版更好，反而更差 ⇒ GIL 假说被否定。**
**⇒ 也就是说：0.76× 是真实的资源争抢，而 ① 的 4.3% 才是那个异常值。**

**⇒ 但 ① 也不是凭空来的**：它的 CPU 单独数字（4.533 s）与 ④（3.172 s）就不一致，
**⇒ 说明两次的 CPU 侧负载本身就不可比**，而那正是矛盾最可能的来源。

**⇒ 诚实结论：这条【待重测】。** 而重测前必须先把两件事对齐：

1. **CPU 侧用完全相同的循环与预热**（两次的"单独"数字差了 43%）
2. **核显侧用完全相同的强度**（506.9 对 789.5，差 56%）
3. **并且记录核显的图形时钟与功耗**，以排除 throttling

**⇒ 而这条纪律与前十一条同形**：**两次测量不一致时，不能挑对自己有利的那个。**
**7.46 的"核显关闭"和本节一度宣布的"核显重开"，两者都建立在单一测量上，而两次单一测量互相矛盾
⇒ 所以两者都不成立，正确状态是【未决】。**

#### 7.52.4 今晚净得到的

| 项 | 状态 |
|---|---|
| **L2 分块 1.359×** | ✓ **成立，可直接用** |
| FMA 371 是机器上限（79.8%） | ✓ 成立 |
| GEMM 是计算受限（高出带宽门槛 33 倍） | ✓ 成立 |
| ~~核显重开~~ | ✗ **未决** —— ①④ 矛盾且未解释 |
| ~~核显关闭~~ | ✗ **同样未决** —— 7.46 的依据是 0.76×，而 ④ 复现了它、① 否定了它 |

**⇒ 而唯一确定可用的结论是 L2 分块那 36%。**

'''

m = re.search(r'^### 7\.52 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.52'
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
