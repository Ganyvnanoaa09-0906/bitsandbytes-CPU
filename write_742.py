"""Append 7.42: the first borrowed technique tried -- torch.compile -- and it does not help.

Negative, clean, and cheap (about 5 minutes). Two implementation notes worth keeping, because
both cost time to discover and will recur.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.42 抄第一件：torch.compile —— 不成立，−2.4%

CUDA 最大的结构性优势是**融合**（不为每个逐元素算子单独起核）。CPU 上的对应物是 Inductor。
目标很具体：**490 次 `aten::mul`（占 11.6%）**，全部来自 RMSNorm 展开，外加调度开销。

#### 7.42.1 结果

| 项 | 值 |
|---|---|
| 未编译（10 步，含 3 步预热） | **3.8513 s/步** |
| 已编译（10 步，含 3 步预热） | **3.9428 s/步** |
| 加速比 | **0.9768×（−2.4%）** |
| 编译耗时 | **仅 6.1 s** |

**判据（7.39.4 事先固定：稳态步时降 > 5% 才留）⇒ 不通过，该方向关闭。**

#### 7.42.2 为什么没作用

**编译只花 6.1 秒，说明 Inductor 把大部分算子都回退了**——CPU 上它能编译的模式有限。
所以它基本没做事，而不是做了事却没效果。

而更根本的原因与本报告早先的发现一致：**那 490 次 `mul` 已经在 `F.rms_norm` 的融合核心里**，
不是散落的逐元素调用（见 7.31.4 与 7.41.2）。**没有留给 Inductor 的剩余空间。**

**⇒ 这与 7.41.2 的"比例稳定"是同一件事**：算子结构没有可融合的余量。

#### 7.42.3 两个实现坑（记下来避免重复踩）

1. **必须设 `PYTHONUTF8=1`（或 `python -X utf8`）。** 否则 Inductor 起子进程读输出时用系统
   默认的 GBK 编码，直接抛 `'gbk' codec can't decode byte 0x94`。第一次运行就是这么失败的，
   而报错信息完全看不出是编码问题。
2. **CPU 上编译很快（6 秒），而不是几分钟。** 7.39.4 里我预判"编译是分钟级、容易被首步骗"，
   那个预判错了——所以要按实测调整：**编译开销在这里可以忽略，不构成使用障碍**，
   真正的问题是收益本身为零。

#### 7.42.4 队列的调整

**⑤ 从队列移除**（已证否）。剩下四项不变，而顺序按 7.41.4 的新发现调整为：

| 顺序 | 内容 | 理由 |
|---|---|---|
| **①** | **核显内核正确性**（当前 914~1793 个元素算错） | **基准不可信 ⇒ 任何分块/布局实验都测不出收益** |
| **②** | **Swizzle / padding → 核显 LDS** | 七次失败里唯一没试的维度 |
| **③** | **attention 分块**（占 18.3% 而效率未测） | 唯一性质未知的大项 |
| **④** | **软件预取**（延迟 259 ns、在飞约 7.5 KB） | 打片外那一层 |
| ✗ | ~~torch.compile~~ | **已证否，−2.4%** |

**注意①排在最前**：7.41.4 测出核显 GEMM 有 0.03%~0.15% 的元素错误，**在这种状态下测试任何
优化都是在错误的地基上盖楼**。

'''

m = re.search(r'^### 7\.42 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.42'
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
