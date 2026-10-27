"""Append 7.36: the re-measurement list the user asked for.

The user's point: several numbers in this report were taken while something else was running,
and those numbers underpin later conclusions. Re-measuring them on an idle machine does not
produce new findings -- it makes the existing ones stand on solid ground.

The selection rule is stated first so it can be applied to future measurements too: any figure
whose log records concurrent load gets re-measured; anything that passed measure_guard without
an exemption does not. That rule cuts the list to four items and explicitly clears the rest,
which matters because "re-measure everything" would waste a day.

Both the old and new values are to be kept. A number that changes without a record of what it
changed from is worse than the contaminated original.
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

SEC = '''### 7.36 受污染测量的重测清单（用户要求）

用户指出：报告里有一部分数字是在机器有别的负载时采的，而它们支撑着后面的结论。
**重测不产生新结论，它把已有结论的地基换结实。**

#### 7.36.1 判定规则

**凡日志里记录了并发负载的数字，一律重测；凡通过 `measure_guard` 且未使用
`allow=True` 豁免的测量，不动。**

这条规则把清单压到四项，其余明确排除——否则"全部重测"会浪费一整天。

#### 7.36.2 需要重测的四项

| # | 数字 | 出处 | 为什么污染 | 影响什么 |
|---|---|---|---|---|
| **①** | **batch 8 的算子占比**（mm 53.56% · attention 18.8%） | 1.20 节 profiler | 当时核显实验在跑 | **所有算子分析都源于此** |
| **②** | **3000 步基线的 3.671 s/步** | 1.20.4 | 无守卫，未确认空闲 | 报告中被引用最多的数字之一 |
| **③** | **Muon 臂 2000 步的总墙钟 11883.8 s** | 7.27 | 核显测试与其并行 | 进了 1.96× 的计算 |
| **④** | **ar_long 各段耗时** | 7.33 之后 | VS 下载器自 07:55 持续读盘 14.25 GB | 长跑的墙钟记录 |

**①②的期望值**：比例（53.56% / 18.8%）很可能稳定，因为它们是相对值；**但绝对耗时会变**，
而绝对耗时才是用来算"某算子占多少秒"的。**重测后两个都要报。**

**③已被部分修正**：每步成本已有干净的补充测量（`clean_step_timing.py`：AdamW 3.316 s /
Muon 5.942 s，7.31.3），**但那次 A/B 的总时长仍是污染的**，而 1.96× 用的是每步成本而非
总时长——所以结论应不变，需要重测确认的是这个推断本身。

**④最简单**：长跑重跑的代价太高，改为**用干净条件下的短测标定 `s/step`，再乘步数**得到
干净墙钟的估计，并注明是估计而非实测。

#### 7.36.3 明确不需要重测的

| 测量 | 为什么清白 |
|---|---|
| 内存诊断（7.34.2） | 23:31 空闲时跑，守卫确认"无其他大任务" |
| NS 审计与微基准（7.31.4.2） | 训练前跑，未开任何作业 |
| TK 扫描、occupancy 扫描（7.28） | 均通过守卫（TK 扫描那次是补跑的，第一次因污染已作废） |
| `clean_step_timing.py` | 专门为干净条件写的，守卫通过 |
| Muon vs AdamW 的 600 步与 2000 步**步数比** | 步数比只依赖 loss 曲线，与每步耗时无关 |

**最后一行值得强调**：步数比（3.19× / 4.98×）**不受时间污染影响**，因为它比较的是 loss
曲线上的位置。受影响的只有时间口径。

#### 7.36.4 重测的两个前提

1. **机器完全空闲**——守卫通过，**不使用 `allow=True` 豁免**。豁免只对"负载不影响所测对象"
   的情形正当（例如测核显时 CPU 忙），而重测的目的恰恰是消除负载影响。
2. **新旧数字都保留**——报告里并列，标明各自的条件。**一个改了却没有记录原值的数字，
   比原来那个被污染的数字更糟。**

#### 7.36.5 并入周四队列

原队列（7.35.1）的 ①②③ 之后加为第 ④ 项：

| 顺序 | 内容 | 预计 |
|---|---|---|
| ① | Muon 隔步正交化 | 3 小时 |
| ② | QKV 融合 | 4 小时 |
| ③ | flash attention 后端 | 30 分钟 |
| **④** | **本节的四项重测** | **2 小时** |

**排在最后，但不是最不重要。** 前面所有分析建立在这些数字上，只是它不产生新结论，
所以在"一天能做完多少"的排序里让位；若时间充裕，它比新实验更该先做。

'''

m = re.search(r'^### 7\.36 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.36'
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
