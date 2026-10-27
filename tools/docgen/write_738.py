"""Append 7.38: the weight decay result, the dataset inspection, and the timing discrepancy.

Three findings from 10-14, in order of importance:

  1. Muon had no weight decay at all. Adding it moved validation from 5.3546 (the previous best,
     which was itself a peak that then reversed) to 5.0678 and still descending, with the
     train/validation gap going from +2.08 to negative. This is the largest quality improvement
     in the project so far, and it came from a missing line rather than from speed work.

  2. The training data is cosplay photography, and had never been looked at. It reframes the
     flat samples: 32x32 is four to eight tokens across a face, and 14432 complex images is a
     small corpus. The warm colour dominance in the samples is the dataset's own statistics.

  3. mean_step_s under-reports wall clock by about 20%: 6000 steps at 4.758 s reports 7.93 h
     while elapsed_s is 9.55 h. Anything quoting per-step time as a total is too optimistic.
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

SEC = '''### 7.38 权重衰减：项目至今最大的一次质量提升

#### 7.38.1 发现

7.37 的 12000 步长跑暴露过拟合后，我去查了优化器的实现，发现：

```
Muon.step 的全部内容:
    buf.mul_(momentum).add_(p.grad)
    d = p.grad.add(buf, alpha=momentum)
    d = zeropower_via_newtonschulz5(d, ns_iters)
    p.add_(d, alpha=-lr)          ← 只有更新，【没有任何衰减】
```

**Muon 管着 60 / 65 个 2D 权重，而它没有权重衰减。** 模型里也没有 dropout
（`cond_dropout` 只在有 pooled 条件时生效，而 `cond_mode='none'`），`--token-dropout` 默认 0。

**这个缺陷是从 `ab_muon_v2.py` 照搬的**——那个 harness 的 Muon 同样没有衰减。但它只跑过
600~2000 步，那时过拟合尚未显形，所以一直没暴露。**直到 12000 步的长跑把它照出来。**

#### 7.38.2 修复

解耦权重衰减（与 AdamW 同形）：**先向零收缩，再施加更新**。这样正交化的更新方向不受影响
——而那正是 Muon 的全部意义。

```python
if _wd > 0:
    p.mul_(1.0 - g['lr'] * _wd)     # 收缩
p.add_(d, alpha=-g['lr'])            # 更新
```

验证：`wd=0` 时权重**逐位不变**（默认路径安全）；`wd=0.5` 配 `lr=0.1` 时权重变化**正好 0.05**
（= lr × wd × weight）。参数 `--muon-wd`（默认 0）与 `--adamw-wd`，备份 `train_ar_v2.py.pre_wd`。

#### 7.38.3 结果：6000 步，只改这一个变量

| 步数 | 有 wd（0.05） | 无 wd |
|---|---|---|
| 261 | 5.7908 | 5.7762 |
| 1041 | 5.5243 | 5.4994 |
| 2081 | 5.4320 | 5.3920 |
| 2601 | 5.3914 | 5.3618 |
| 3381 | **5.3087** | — |
| **4001** | — | **5.3546（其最佳值）** |
| 4421 | 5.2423 | — |
| 5201 | 5.2099 | — |
| **5981** | **5.0678** | — |
| 7001 | — | 5.4900 |
| 11501 | — | 6.0113 |

**⇒ 5.0678 对 5.3546，好 0.2868，而且 5981 步时【仍在下降】。**

**早期反而更慢**（261~2601 步每点落后 0.01~0.04）——这是衰减的代价，符合预期。**转折发生在
3381 步：此时首次低于无 wd 的全程最佳值，而后者在同一步数上已经开始变差。**

#### 7.38.4 过拟合缺口消失

| 阶段 | 无 wd 的 gap | 有 wd 的 gap |
|---|---|---|
| 中期 | +0.51 / +0.89 / +0.16 | +0.64 / +0.35 / +0.51 |
| 后期 | **+1.34 / +1.70 / +2.08** | **−0.56 / −0.36 / −0.14** |

**无 wd 的 gap 单调增长到 +2.08；有 wd 的在后期转为负值。** train 与 val 基本重合，
说明模型不再是在背训练集。

#### 7.38.5 但图仍是色块——而这次先看了数据

修正指标之后的第一件事是采样，结果：**橙红为主的纹理场，有分层与块状，但没有可辨认的物体。**

于是我去看了训练数据本身——**这件事此前从未做过**。

**数据集是真人 cosplay 摄影**：人物、精致服饰、棚拍布光、复杂背景。

这改变了整个读法：

* **32×32 网格对人物照片极其粗糙**——一张脸大约只有 4~8 个 token 宽；
* **14432 张这种复杂度是偏小的语料**；
* **样本里的橙红主导正是这个数据集的色调统计**，不是失败模式。

**⇒ 所以"色块"不是 bug，也不只是训练不足，而是任务难度与数据量的共同结果。** 权重衰减让
模型学出更准的**色调统计**（指标从 5.35 到 5.07），**但那不等于结构。**

#### 7.38.6 顺带查清的一处口径错误

`mean_step_s` **低估了真实墙钟约 20%**：

| 项 | 值 |
|---|---|
| `mean_step_s` × 步数 | 4.758 × 6000 = 28548 s = **7.93 h** |
| `elapsed_s`（实测） | 34354 s = **9.55 h** |
| 差 | **1.61 h** |

**因为 `mean_step_s` 只统计被计时的训练步，不含模型构建、数据加载、每次验证的前向和
checkpoint 保存。** 所以**报告里凡是用"每步耗时 × 步数"得出的总时长都偏乐观 20%**，
正确口径是 `elapsed_s`。**这一条并入 7.36 的重测清单。**

#### 7.38.7 紧接着的续训

无 wd 的曲线在 4001 步见顶反弹，那正是"训更久是错的"的依据。**有 wd 的曲线到 6000 步
从未反弹**，所以加步数重新变得有意义——问题从"会不会反弹"变成"在哪里反弹"。

已启动：**从 6000 步续训到 18000 步**（4 段 × 3000，每段采样 8 张），配置不变，预计 19 小时。

#### 7.38.8 这一轮的方法论收获

**三次"指标与直觉冲突"，三次都是先测量后判断赢了：**

1. 无 wd 的 12000 步：**图看起来更丰富而 loss 更差** ⇒ 感知丰富度不等于质量；
2. 有 wd 的 6000 步：**指标明显更好而图仍是色块** ⇒ 指标改善不等于结构出现；
3. 而解开这两次的钥匙，都是**去看一个此前没人看过的东西**——第一次是验证曲线，第二次是
   **训练数据本身**。

**⇒ 两次都说明：在解释任何结果之前，先确认自己看的是不是那个东西。**

'''

m = re.search(r'^### 7\.38 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.38'
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
