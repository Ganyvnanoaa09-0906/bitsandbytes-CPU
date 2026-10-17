"""Append 7.64: the residual is not the OpenMP pool. The 1.62x slice bound is hard.

The guess in 7.63 was that torch's OpenMP workers keep spinning after a matmul, so the CPU is not
really idle during a GPU slice and power arbitration stays partly engaged. Dropping torch to one
thread before the slice moves it from 18.5 ms to 18.17, about 1.8%, so the guess is wrong.

Which matters because it makes the slice figure a property of the hardware rather than of the
software configuration. 308 GFLOPS inside a slice against 190 on the CPU gives 1.62x at f=1.0 and
that is not going to improve by tuning thread pools.
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

SEC = '''### 7.64 那 25% 残余不是 OMP 空转 —— 1.62× 是硬上限

#### 7.64.1 验证

| 条件 | 切片稳态 ms | 切片 GFLOPS |
|---|---|---|
| 6 线程，不做处理（对照） | 18.5 | 308 |
| **切片前降到 1 线程** | **18.173** | **315.7** |

**⇒ 只快 1.8% ⇒ 【假说否掉】。**

#### 7.64.2 所以残余在硬件状态里

**⇒ 不是"CPU 在假装空闲"，而是【功耗状态本身没有完全回来】——
一旦有 CPU 活动发生过，iGPU 就不会立刻回到最高档。**

**⇒ 而这意味着 7.63 的切片收益是【硬】的**：

```
切片内 308 GFLOPS 对 CPU 190 ⇒ f=1.0 时 1.62×
⇒ ★【而它不能靠调线程池、设环境变量、或任何软件配置抹掉】
```

#### 7.64.3 三条路的最终数字

| 方案 | 收益 | 性质 | 代价 |
|---|---|---|---|
| 并发分工 | **负** | 功耗仲裁，开关式 | — |
| **时间切片（f=1.0）** | **1.62×（硬上限）** | 每切片内 308 对 190 | 重写整套前反向 |
| **库：静默回退修复** | **10.6×** | 已实现已验证 | 一个 C++ 文件 |

**⇒ 而这条差距是 6.5 倍 —— 所以优先级没有悬念。**

#### 7.64.4 而这一整条核显线，值得记住的是它的【形状】

**从 7.46 到 7.64，核显这个问题测了十几次，而结论反复了四次：**

```
关闭(0.76×) → 重开(4.3%) → 重开(1.78×) → 关闭(2~5×) → 拖慢4倍 → 功耗仲裁 → 切片 1.62×
```

**⇒ 而每一次反复的原因都是同一个：****在【不同的机器状态】下测了同一个东西，而没把状态当作自变量。**

* 0.76× 和 4.3% 是线程/进程之别（GIL 假说，后被否）
* 409 和 90 是"CPU 闲"与"CPU 忙"之别（**分母也不同**）
* 1.78× 建立在"CPU 闲时的速度"上，而训练里 CPU 从不闲
* 而最终答案在**把 CPU 强度扫出来**之后才出现 —— 而扫出来的是**一条悬崖，不是一个数**

**⇒ 所以这条线的真正教训不是"核显能不能用"，而是：**

> **当同一个量在不同条件下给出 20 倍差异时，正确做法不是"取哪个"，而是【把那个条件扫出来】。
> 而扫出来的往往不是一条平滑曲线，而是一个说明机制的形状 —— 悬崖意味着开关，而不是比例。**

'''

m = re.search(r'^### 7\.64 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.64'
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
