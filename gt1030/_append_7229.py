# -*- coding: utf-8 -*-
"""追加 §7.222.9：训练语义对齐（weight decay）+ iGPU 100 步 loss 与 CPU 同量级。"""
import io

TEXT = r"""

#### 7.222.9 语义对齐：weight decay 与"和 CPU 同量级"的定量证据

**一个容易漏掉的语义差**：CPU trainer 用的是 `bnb.optim.AdamW8bit(...)`，没显式传 `weight_decay`
⇒ bnb 默认 **0.01**；我这边最初一直传 `wd=0`。10 步轨迹看不出来（每步只差 `lr·wd=5e-7`），
但几百步后会有 0.5% 量级的系统性漂移。已改成默认 `wd=0.01`（`adamw_single` 的衰减形式
`w -= lr·wd·w` 和 torch.AdamW 的 decoupled 形式一致），并复测 10 步轨迹：
**最大偏差 2.00e-3、均值 9.3e-4**，仍远优于 0.02 的验收线。

**"iGPU 训练和 CPU 训练是不是同一个东西"的定量对照**（同一步数附近，loss 都是单 batch 快照）：

| 来源 | 步数 | loss |
|---|---|---|
| CPU `ar_full3.log` | 13201 / 13801 / 14401 | 5.5293 / 3.8162 / 4.4609（单 batch，噪声极大） |
| iGPU 续训 100 步（`ckpt_14000` → 14100） | 100 个 batch | mean **5.3382**、median **5.4999**、range 3.5855~6.4036 |

两者的分布同量级、同抖动幅度 ⇒ iGPU 侧的前向/反向/优化器在"统计意义上"复现了 CPU 的训练行为，
而不只是"十步数值对齐"。这也解释了为什么 CPU 的 `train=` 一列（3.8~5.5）会和 `val=5.36` 差那么多：
train 是**单 batch** 的快照，val 是 382 张图的平均；`gap` 列的正负跳变（+1.56 / −0.11）全是这个原因，
不是模型在抖。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))
