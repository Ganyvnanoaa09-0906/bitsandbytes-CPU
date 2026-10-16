"""Append 7.68: chunking the batch dimension is K-dependent, and that itself is the finding.

Chunking N failed in 7.54. Chunking M keeps B fixed across the loop, so it should be the better
scheme, and it is for large K and worse for small K. The interesting part is not the net result, which
is about zero for the model, but that a manual scheme beats torch by 14.6% on one shape.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.68 切 M 维：K 依赖 —— 而"torch 非形状最优"才是发现

#### 7.68.1 为什么试这个

**切 N（7.52/7.54）失败，因为【B 的块和 C 的块都在变 ⇒ 每轮都要重新取 A】。**

**⇒ 而切 M 是另一回事：**

```python
C[i:i+m] = A[i:i+m] @ B      # ★ B 在整轮里【保持不变】⇒ 可以驻留
```

**⇒ 而模型的 MLP 输出 [2048, 2736] 是 22.4 MB，装不进 8 MB 的 L3；而切 M 之后行块可以。**

#### 7.68.2 结果（对预分配全量，避开 7.54 那个分配混淆）

| 形状 MxNxK | 全量预分配 | 切 M 最优 | **倍数** | 最优 m |
|---|---|---|---|---|
| 2048×2736×**512** | 185.7 | 165.0 | **0.889×** ✗ | 512 |
| 2048×512×**2736** | 252.3 | 289.2 | ★ **1.146×** | 512 |
| 2048×512×**2048** | 247.9 | 266.3 | ★ **1.074×** | 512 |
| 2048×2048×**512** | 195.7 | 178.9 | **0.915×** ✗ | 512 |

**⇒ 规律：K 大时切 M 有效，K 小时反而更慢。**

#### 7.68.3 而套到模型的 MLP 上，净效果约等于零

```
w12:  [2048, 2736] × K=512   ⇒ 0.889×  ✗
w3 :  [2048, 512]  × K=2048  ⇒ 1.074×  ✓

而 w12 占 MLP 的 5.7/10 GFLOP、w3 占 4.3/10
⇒ 净 ≈ 5.7/10×0.889 + 4.3/10×1.074 = 【0.968×】⇒ 反而略慢
⇒ ★【与切 N 同判：不划算】
```

#### 7.68.4 ★ 但这一跑证明了一件更要紧的事

```
切 M 在 2048x512x2736 上比 torch 的【预分配全量】快 【14.6%】
⇒ ★【torch 自己的分块【不是对所有形状都最优】】
⇒ 而这意味着【形状本身】是一个还没被用的杠杆
```

**⇒ 而它同时说明为什么库需要一张【分派表】**：

```
最优是【分段且非单调】的 —— K=512 输、K=2048 赢
⇒ 任何"推论出来的最优"都会在某一段上错
⇒ ★【只能实测建表】（这一条在 7.59.7 已提过,而这是它的第二个实例）
```

#### 7.68.5 所以 CPU 侧现在的完整清单

| 项 | 状态 |
|---|---|
| 切 N 维 | ✗ 0.87~1.006×（7.54 撤回） |
| **切 M 维** | ✗ **净 0.968×**，**但证明了 torch 非形状最优** |
| 四个速度杠杆 | ✗ 全关 |
| **★ 内存延迟异常** | **未解：259 ns 对正常的 70~90** |
| **★ 线程扩展饱和** | **未解：1 线程 72.9 → 6 线程 190.0，只有 2.6×（而非 6×）** |
| **形状选择** | ★ **新线索：速率依赖 K/N,而形状是我们能选的** |

#### 7.68.6 而最后两条可能是同一个原因

```
线程扩展只 2.6×（而非 6×）  ⇒ 说明有东西在 2 线程之后就饱和了
指针追逐延迟 259 ns（正常 70~90）⇒ 说明访存子系统本身不正常
⇒ ★【两条都指向【内存子系统】,而如果它有解,杠杆会比任何分块都大】
⇒ 而 7.41.3 已记录过 stride 测试是坏的 —— 而【延迟测试没被同样审视过】
```

**⇒ 所以下一步该做的是：【把延迟测试本身验一遍】—— 因为如果它和 stride 测试一样有缺陷，
那 259 ns 这个数字就不可信，而"内存子系统有问题"这个判断也就没有依据。**

'''

m = re.search(r'^### 7\.68 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.68'
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
