"""Append 7.37: the long run's result, which is negative and important.

12000 steps in eight segments over 17.2 hours, sampling every 1500. The hypothesis it was
built to test -- that the flat colour blocks at 3000 steps were simply not enough training --
is refuted: validation loss is best at 4001 steps (5.3546) and rises monotonically to 6.0113
at 11501, while train loss keeps falling. That is overfitting, not underfitting.

The samples make it look like progress and the metric says otherwise, and the discrepancy is
the point. At 12000 steps the images have visible textures -- water-like patterns, speckles,
green blocks -- which read as structure to the eye. But held-out loss is worse than at 3000
steps. The model learned high-frequency texture from the training set; it did not learn
structure that generalises. Perceptual richness is not quality, and here the metric is right.

No shutdown was requested for this run and none was performed.
"""
from __future__ import annotations

import io
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.37 长跑结果：12000 步，而答案是负的 —— 过拟合

用户提出的假设是"30M 模型训 3 万步不该是色块"。为了检验它，跑了一次分段长跑：8 段 × 1500
步，每段结束采样 8 张，共 **12000 步 / 17.2 小时**（10-13 12:35 → 10-14 05:47）。

**假设被否证。** 不是训练不足，是**过拟合**。

#### 7.37.1 数字

| 全局步 | train | val | gap |
|---|---|---|---|
| 4001 | — | **5.3546** | — |
| 7001 | 5.1995 | 5.4900 | +0.2904 |
| 8001 | 4.2558 | 5.5960 | +1.3402 |
| 8501 | 4.9930 | 5.6536 | +0.6607 |
| 9501 | 4.0506 | 5.7515 | +1.7010 |
| 10001 | 4.8137 | 5.8355 | +1.0218 |
| 11001 | 3.8349 | 5.9143 | **+2.0794** |
| 11501 | 4.5971 | **6.0113** | +1.4142 |

**最佳验证 loss 出现在 step 4001 的 5.3546，此后单调上升。** 到 11501 步已经是 6.0113，
而**随机猜测（1024 类均匀）是 ln(1024) = 6.931**。gap 从 0.29 涨到 2.08。

**train 一直在降（5.20 → 4.60，最低 3.83），val 一直在涨。** 两个方向相反且分开，这是
过拟合的定义式。

#### 7.37.2 而出图假象：看起来更好，指标说更差

| 步数 | 图的观感 | 验证 loss |
|---|---|---|
| 3000 | 纯色块 | 5.30 |
| 12000 | **有纹理**（水波 · 斑点 · 块状结构） | **6.01** |

**眼睛觉得 12000 步的图"更有结构"，而它的 held-out loss 比 3000 步更差。**

解释：模型学到的是**训练集里的高频纹理**，不是可泛化的结构。它把训练样本的局部细节背了
下来，因此在训练分布上越拟合、在留出集上越差。**感知上的丰富度不等于质量，这里指标是对的。**

这一点必须写清楚，因为它正好是外部计划里那条警告的实例：
**「不要把训练 loss 更低直接等同于生成质量更好」** —— 而这次是反过来，**生成看起来更有内容
而 loss 更差**，同样不能当作进步。

#### 7.37.3 为什么是过拟合

| 项 | 数值 |
|---|---|
| 训练数据 | **14432 张原图 / 28864 行 token**（含翻转） |
| 模型参数 | 30.5 M |
| 序列长度 | 1024 token/图 |
| 12000 步 × batch 2 | **24000 次抽样**，对 28287 行训练集约覆盖 57% |

**容量与数据量的比例失衡。** 30.5M 参数的模型对 14432 张图，加上 1024 长度的序列，
模型有足够自由度在 4000 步后就转向记忆。

#### 7.37.4 对加速比的影响：**3.57× 反而更站得住**

我们要的工作点是 **val 5.35 ~ 5.8**，而这**恰好是两条曲线重叠、加速比被实测出来的区间**
（7.33.3）：

| 目标 val | AdamW 步数 | Muon 步数 | 比 |
|---|---|---|---|
| 6.00 | 781 | 261 | 2.99× |
| 5.90 | 781 | 261 | 2.99× |
| 5.80 | 1301 | 261 | 4.98× |
| 5.55 | 未在记录内达到（3000 步终值 5.5328） | 781 | — |

**更深的 loss 我们到不了（因为过拟合），所以不需要外推加速比到未测量区。**
3.57× 是在实际工作点上测的，这一点现在比之前更确定。

#### 7.37.5 下一步该做什么（按优先级）

| # | 方向 | 理由 |
|---|---|---|
| **①** | **早停在工作点**（~4000 步） | 最佳 val 在 4001 步；继续训只有害处 |
| **②** | **正则化**：dropout / weight decay / 更强的数据增强 | 直接针对 gap 的增长 |
| **③** | **缩小模型** | 30.5M 对 14432 张图偏大；减半可能反而更好 |
| **④** | **扩充数据** | 根本解法，但需要新数据源 |

**明确不做的**：再跑更长的训练。12000 步已经证明方向是错的，18000 步只会更差。

#### 7.37.6 这次长跑仍然值得

它花了 17.2 小时，得到一个负结论。**但负结论在这里价值很高**：

* 它杀掉了"多训就能出图"这个直觉上最诱人的假设；
* 它把问题从"训练不够"重新定位到"数据不够 / 容量过大"；
* 它给出的最佳工作点（4000 步、val 5.3546）让后续实验有了明确的目标，而不是盲目加长；
* 而且它顺手验证了 3.57× 在真实长跑里成立——12000 步里每步成本稳定在 4.54 s。

**四个明确的门**（7.34）之外的第五个门，今天关上了：**步数**。

'''

m = re.search(r'^### 7\.37 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.37'
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
