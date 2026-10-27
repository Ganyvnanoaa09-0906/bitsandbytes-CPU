"""Append 7.54: 7.52.1's L2 blocking recommendation is retracted. The 1.359x was allocation.

The chunked GEMM was never compared against a like-for-like full call. torch.mm(A, B) allocates a
fresh 22.4 MB output on every invocation while the chunked version wrote slices into one buffer, so
the measurement compared an allocating path against a non-allocating one and attributed the
difference to cache behaviour.

Against a preallocated full call, chunking is slower: 0.904x, 1.006x and 0.870x at the three model
shapes. The recommendation and the 19% step saving are withdrawn.

This is the fourteenth instance of the project's recurring error, but the first caught before
implementing rather than after.
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

SEC = '''### 7.54 撤回 7.52.1：L2 分块那 1.359× 是内存分配，不是缓存

#### 7.54.1 混淆在哪

```python
基线:  torch.mm(A, B)                          # ★ 每次【新分配】22.4 MB 输出
对照:  Cc[:, j:j+k] = torch.mm(A, B[:, ...])   # ★ 只分配一次 Cc,再写切片
```

**两次调用的差别不止分块 —— 还有【分配次数】。** 而 22.4 MB 的分配不便宜。
**⇒ 那次测量是把"分配路径"和"非分配路径"比，然后把差异归给了缓存。**

#### 7.54.2 排掉混淆之后

| 形状 | 全量新分配 | 全量预分配 | 分块预分配 | **分块/预分配** |
|---|---|---|---|---|
| 2048×2736×512 | 191.8 | 194.4 | 175.7 | **0.904×** |
| 2048×512×2736 | 242.8 | 264.5 | 266.1 | **1.006×** |
| 2048×2048×512 | 183.2 | 200.5 | 174.5 | **0.870×** |

**⇒ ★ 分块【比预分配的全量调用更慢】（0.870~1.006×）。**
**⇒ 判据（> 5%）不成立 ⇒ 【7.52.1 的建议作废，那"整步省 19%"撤回】。**

#### 7.54.3 而顺带发现的真实（但用不上的）效应

**预分配本身确实有收益**：194.4 对 191.8（+1.3%）、264.5 对 242.8（+8.9%）、200.5 对 183.2（+9.4%）。

**⇒ 但训练的前向【必须新分配**（autograd 不支持 `out=` 参数），**⇒ 这个收益拿不到。**
**⇒ 所以它只是解释了那次测量的来源，不是一个可用优化。**

#### 7.54.4 与 7.53 的关系：速度账要重算

```
7.53 的 1.78× 建立在"核显卸载 GEMM"上,而【不含】L2 分块
⇒ 所以 1.78× 【不受本次撤回影响】✓
⇒ 而 7.53.3 里"叠上 L2 分块 ⇒ 约 2.16×"那句【作废】
⇒ 修正后的工程上限仍是 【1.78×】
```

**⇒ 而修正后的 epoch 账**：

| 项 | 7.53 曾记 | **修正后** |
|---|---|---|
| 工程加速 | 2.16× | **1.78×** |
| 一个 epoch | 约 8 小时 | **约 9.6 小时** |
| 10 个 epoch | 约 3.3 天 | **约 4 天** |

#### 7.54.5 第十三次纪律的第一次"提前生效"

**这是本项目第十四次同形的错误**（判据匹配了不是它的东西），
**但第一次【在实现之前】被拦住** —— 而拦住的唯一原因是：

> **在动手改代码之前，先问一句"对照组的条件真的只差我要测的那一个变量吗"。**

**⇒ 而 7.52.1 当时的报告里其实写着两边的代码**，差别（分配 vs 不分配）就摆在那里，
**⇒ 所以它本来是可以当场发现的。** 而我没看，因为"分块更快"这个结论符合预期。

**⇒ 这给出一条新的、更具体的纪律：**

> **凡是要写进"可执行建议"的结论，必须先把它准备施加的代码和对照组的代码并排看一遍，
> 确认两者只差那一个变量。** 单看数字不够，因为符合预期的数字最容易被放过。

'''

m = re.search(r'^### 7\.54 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.54'
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
