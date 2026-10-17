"""Append 7.63: the switch cost is small, so time slicing is viable at about 1.6x.

Concurrency fails because power arbitration makes the two engines mutually exclusive at full rate.
Time slicing avoids that by never running them together, and its arithmetic gives 1/(1-0.513f) if
switching is free. This measures whether it is free.

After a CPU burst the iGPU's next call costs 31-35 ms against a 14.8 ms idle baseline, so the switch
costs about 16-20 ms and only once. The second call is back to 18-21 ms, so recovery takes two calls.
And the burst length makes no difference -- 50 ms and 1000 ms behave identically -- which means this is
a state transition rather than thermal accumulation, and therefore a fixed cost that amortises.

Steady state inside a slice is 18.5 ms against 14.8 idle, a 25% residual penalty. The plausible cause
is that torch's OpenMP pool keeps spinning after a matmul, so the CPU is not actually idle during the
GPU slice and the power arbitration is still partly in effect. That is testable and likely fixable.
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

SEC = '''### 7.63 切换成本很小 ⇒ 时间切片可行（约 1.6×）

#### 7.63.1 测量

| CPU 忙多久 | 之后连续 6 次核显调用（ms） |
|---|---|
| （空闲基准） | **14.803** ⇒ 387.6 GFLOPS |
| **50 ms** | **30.9** · 21.4 · 18.0 · 17.8 · 17.7 · 17.7 |
| **200 ms** | **34.9** · 20.3 · 19.0 · 18.7 · 18.7 · 18.3 |
| **1000 ms** | **34.8** · 20.2 · 19.3 · 19.3 · 19.1 · 19.0 |

#### 7.63.2 三条读数

```
① 第一次调用 31~35 ms 对基准 14.8 ⇒ 【切换成本约 16~20 ms,而【只付一次】】
② 第二次就回到 18~21 ms          ⇒ 【两次调用内恢复】
③ ★ CPU 忙 50 ms 与忙 1000 ms【结果一样】
   ⇒ ★【没有热积累,纯粹是状态切换 ⇒ 固定成本,可以摊薄】
```

**⇒ ③ 是最重要的一条**：**它说明这不是"用久了会变糟"的热问题，而是一个能靠切片长度摊掉的固定开销。**

#### 7.63.3 而切片的收益可以算了

```
切片内核显稳态: 18.5 ms ⇒ 【308 GFLOPS】（空闲时 387.6）
CPU 单独:        190.0 GFLOPS

⇒ 加速 = 1 / (1 - f + f×190/308) = 1 / (1 - 0.383f)
     f = 0.5  ⇒ 【1.24×】
     f = 1.0  ⇒ 【1.62×】
```

**⇒ 所以时间切片成立，而收益是 1.24~1.62×，不是满速的 2.05×。**

#### 7.63.4 ★ 而那个 25% 残余惩罚很可能可修

```
切片内 18.5 ms 对空闲 14.8 ms ⇒ 残余 【+25%】

★ 怀疑原因:torch 的 OpenMP 线程池在 matmul 之后还会【空转】一段时间
⇒ 所以"CPU 空闲"其实不是真的空闲 —— 它还在烧电,于是功耗仲裁仍部分生效
```

**⇒ 修法可能只有一行**：GPU 切片期间把 torch 的线程数降到 1，或设 `OMP_WAIT_POLICY=PASSIVE`，
**让 CPU 真的停下来。**

**⇒ 若残余惩罚消失 ⇒ 切片内回到 387.6 GFLOPS ⇒ 加速变成 `1/(1 - 0.51f)`：
f=0.5 ⇒ 【1.34×】· f=1.0 ⇒ 【1.96×】。**

#### 7.63.5 而与"全给核显"的关系

**"全给核显"（f=1.0）的上限是 1.62~1.96×**，**而它的门票是**：

```
核显上要有一套完整的前向+反向:
  autograd · attention · RMSNorm · SiLU · cross-entropy —— 全部 OpenCL
⇒ ★【代价不是"写一个内核",而是在这台机器上重写一个 PyTorch】
```

**⇒ 而对照库那侧那条路：值 10.6×，代价是一个 C++ 文件。**

#### 7.63.6 结论

| 方案 | 理论 | 实测/修正后 | 代价 |
|---|---|---|---|
| 并发分工 | — | **负**（功耗仲裁） | — |
| **时间切片** | 2.05× | **1.24~1.62×（修掉 OMP 空转后 1.34~1.96×）** | **重写训练框架** |
| **库：静默回退修复** | — | **10.6×** | **一个 C++ 文件** |

**⇒ 所以核显这条线的正确形态是【微批次时间切片】，而这值得记下来 —— 它是这个硬件上唯一成立的用法。
⇒ 但按性价比，库那条线仍然是优先的。**

'''

m = re.search(r'^### 7\.63 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.63'
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
