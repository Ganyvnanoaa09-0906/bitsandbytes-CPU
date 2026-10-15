"""Append 7.50: the MaskGIT sampler bug, and the epoch arithmetic that reframes the project.

Two findings. The first is a fix: MaskGIT sampling ran over the full 1024-way distribution while the
AR path filtered to the top 100, which is why the step-4000 samples were chaotic fragments. With
filtering the colours become coherent.

The second is more important and came out of trying to interpret that result. Counting tokens
against the training set shows that no run in this project has ever completed a single epoch. The
6000-step MaskGIT run is at 0.43 epochs. The grid-16 run reached 0.83. And image generation models
are normally trained for a hundred epochs or more.

Which means every conclusion of the form "the model cannot learn structure" was drawn far too early,
and the real constraint is not the kernel, the objective, the tokenizer or the architecture. It is
that ten epochs would take seven days on this machine, and ten epochs is still an order of magnitude
short of normal practice.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.50 MaskGIT 采样器的 bug，以及把整个项目重新框一遍的 epoch 算术

#### 7.50.1 采样器 bug：MaskGIT 从 1024 个词表裸采样

step 4000 的出图是**高频碎片噪声**——小色块拼贴。**它不是 AR 那种平滑色块，而是"并行预测互相
不一致"的形态**，而这种形态的根源通常在采样而非训练。

对照两条路径，问题一目了然：

```python
# L189 AR 路径      ⇒ model.generate(..., top_k=a.top_k, ...)         ✓ 传了
# L183 MaskGIT 路径 ⇒ model.generate_masked(...)                      ✗ 没传
#                                而 generate_masked【根本没有 top_k 参数】
```

```python
p = torch.softmax(lg, dim=-1)
pred = torch.multinomial(p.view(-1, V), 1).view(_B, T)   # ★ 从全部 1024 个词表裸采样
```

**⇒ AR 采样过滤到最可能的 100 个 token，MaskGIT 采样不过滤。**
而验证 loss 6.19 对随机 6.93 ⇒ 分布接近均匀 ⇒ **裸采样就是噪声。**

**修法**：给 `generate_masked` 加 `top_k`（默认 0 保持原行为），过滤后重归一化再采样，
置信度取重归一化后的概率。`sample_ar.py` 沿用 `--top-k`（默认 100）。

**⇒ 而这里有一个差点误判的细节**：第一次改完，调用点的锚点没匹配上（缩进不同），
**top_k 其实没传进去**。而输出立刻暴露了它：

```
生成 token: 使用 1024/1024 个码, 熵 6.858/6.931 (98.9%)   ← 【噪声水平的熵】
```

**修对之后**：

```
修复前: 1024/1024 个码 · 熵 6.858（98.9%）· 碎片噪声
修复后:  980/1024 个码 · 熵 6.467（93.3%）· 【粉肤色调的连贯纹理】
训练数据本身: 1024/1024 · 熵 6.694（96.6%）
```

**⇒ 生成分布现在比修复前更接近真实数据分布,而颜色崩塌消失了。**

#### 7.50.2 但仍然没有结构——而原因不在目标函数

**top_k 修好了颜色,没有修好结构。** 而出图仍然是"皮肤/花瓣的近摄纹理",不是人物。

**于是我算了一个此前从没算过的数：**

| 实验 | 步数 | 每步 token | 累计 token 预测 | 训练集 token | **epoch** |
|---|---|---|---|---|---|
| **MaskGIT 6000 步** | 6000 | 2×1024 | 1230 万 | 2830 万 | **0.43** |
| **grid16 12000 步** | 12000 | 2×256 | 610 万 | 740 万 | **0.83** |
| AR 长跑 12000 步 | 12000 | 2×1024 | 2460 万 | 2830 万 | **0.87** |
| AR + wd 8500 步 | 8500 | 2×1024 | 1740 万 | 2830 万 | **0.61** |

**⇒ ★ 这个项目里【没有任何一次运行跑满过一个 epoch】。**

**而图像生成模型的通常做法是 100 个 epoch 以上。**

#### 7.50.3 于是此前所有的"模型学不出结构"都下得太早

**不是目标函数不行**（MaskGIT 与 AR 都在 0.4~0.5 epoch 时只学到色调）
**不是分词器不行**（7.48 证明重建接近无损）
**不是架构不行**（未验证，但那正是下面那个测试要答的）
**⇒ 是根本没训够。**

#### 7.50.4 而硬件把这条路堵死了

| 目标 | 步数 | 时间（4.5 s/步） |
|---|---|---|
| 1 个 epoch | 13800 | **17 小时** |
| 10 个 epoch | 138000 | **172 小时 = 7.2 天** |
| 100 个 epoch | 1380000 | **72 天** |

**⇒ 10 个 epoch 要 7.2 天,而那相对正常做法还差一个数量级。**

**⇒ 这才是这个项目真正的边界,比任何内核优化都硬。** 四个速度杠杆全关（7.47.3）意味着
**这条边界无法靠工程手段移动**——它由"30M 参数 × 1024 token × 14432 张图 × 4.5 秒/步"决定。

#### 7.50.5 所以剩下的决定性实验是"能不能拟合"

**既然训不够是算力问题,那么唯一还能在今晚出结果的问题是：模型到底能不能拟合？**

**设计**：100 张图（200 行）· AR 目标 · 2000 步 = **20 个 epoch**。

| 结果 | 结论 |
|---|---|
| 采样出那 100 张的可辨认版本 | **架构没问题,问题是算力** ⇒ 加数据/缩模型都不解决,只能接受边界 |
| 仍然只是色块/纹理 | **架构或采样流程有结构性问题** ⇒ **与数据量无关,加数据也救不了** |

**⇒ 这两者是完全不同的结论,而此前的推理一直默认了前者而没有验证。**

**参照**：7.48 的 VQGAN 重建图（同一表示空间里应该能达到的样子）。

**已准备好**：`tokens_overfit100.pt`（100 张图彼此足够不同：两两 token 一致率均值 0.23%、
最大 16.4%，随机基线 0.098%）。

#### 7.50.6 与 7.43 那条纪律的呼应

**这次的 top_k 差点又成一次"改完没生效"**：锚点没匹配，改动没落地，
**而如果不去看输出里的熵（98.9% 是噪声水平），我会以为"修了 top_k 也没用"，
从而得出"采样器不是问题"的错误结论。**

**⇒ 与前六次同形：改动的【验证】必须落在【改动的效果】上，而不是"我改过了"。**

'''

m = re.search(r'^### 7\.50 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.50'
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
