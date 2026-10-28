"""Record the final Muon result: the LR bug, the fix, and 3.57x.

This supersedes the preliminary 2.51x in 7.32.3, which was wrong and whose retraction has to be
as prominent as the new number.

The sequence:

  7.32   Muon wired into the trainer. First cost measured (5.00 s/step), and a preliminary
         2.51x quoted by multiplying the harness's 3.19x step ratio by the trainer's cost
         ratio -- two different runs, which was a mistake regardless of what followed.
  then   The trainer's own comparison ran: Muon LOST at every checkpoint by a nearly constant
         0.5 and never reached loss 6.2. Constant offset, never closing.
  cause  The cosine schedule. Muon was built with lr=0.02, then lr_at() drove it down to
         --lr-min = 1e-5, a factor of 2000 -- a schedule tuned for AdamW's 1e-4 applied
         unchanged to a rate twenty times larger.
  fix    Muon's param group is tagged and the schedule skips it, matching the constant 0.02
         that ab_muon_v2.py used and that the 3.19x was measured under.
  result 4.98x fewer steps to validation 5.80, 1.37x cost per step, 3.57x net.

31 hours becomes about 8.7 hours, which is the target the user set at the start: get it under
nine.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import io
import os
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

SEC = '''### 7.33 Muon 的 lr bug、修复，以及最终结果 3.57×

**本节取代 7.32.3 的初步值 2.51×，那个数字是错的。** 作废理由写在 7.33.1。

#### 7.33.1 我犯的错：把 AdamW 的学习率计划套到了 Muon 上

训练器里的第一次对比结果是这样：**Muon 在每个检查点都落后，差值几乎恒定在 0.5，
而且从未达到 loss 6.2。**

| 步数 | AdamW | Muon（退火 lr） | 差 |
|---|---|---|---|
| 261 | 6.4318 | 6.8141 | +0.3823 |
| 521 | 6.0381 | 6.5579 | **+0.5198** |
| 781 | 5.8964 | 6.4144 | +0.5180 |
| 1041 | 5.8288 | 6.3392 | +0.5104 |
| 1301 | 5.7982 | 6.3040 | +0.5058 |

**差值恒定且从不收窄** ⇒ 这不是调优差异，而是**两个不同的优化问题**。

**根因**：`Muon(muon_p, lr=0.02)` 构造后，`lr_at()` 用 **AdamW 的余弦计划**把它一路退到
`--lr-min = 1e-5`——**相差 2000 倍**。那个计划是为 AdamW 的 `1e-4` 调的，直接套到
**大 20 倍**的 Muon 学习率上，等于让它在训练后段几乎不动。

**而这也解释了 7.32.3 为什么错**：那个 2.51× 是拿 **harness 的步数比（3.19×）**乘上
**训练器的成本比**——**两次不同运行的数字相乘**，无论后面发生了什么，这个算法本身就不成立。

#### 7.33.2 修复：让 Muon 不参与退火

```python
# Muon.__init__
for g in self.param_groups:
    g['is_muon'] = True          # 标记

# 学习率写回
for g in opt_groups(opt):
    if g.get('is_muon'):
        continue                 # Muon 保持常数 0.02
    g['lr'] = lr_at(step)        # 只有 AdamW 退火
```

**为什么是"让它不参与"而不是"给 Muon 设计一个计划"**：`ab_muon_v2.py` 用的是**常数 0.02**，
而 3.19× 就是在那个设置下测的。**匹配已测过的配置是唯一站得住的选择**，而且只改了一个变量。
AdamW 那一支的余弦**完全不变**，所以基线臂未受影响。
备份：`train_ar_v2.py.pre_lrfix`。

#### 7.33.3 修复后的结果：3.57×

三条曲线，全部走 `train_ar_v2.py`，同配置（1500 步 · batch 2 · T=1024 · 每 250 步验证同
一个留出集）：

| 步数 | AdamW | Muon（退火 lr） | **Muon（固定 lr）** | 固定 vs AdamW |
|---|---|---|---|---|
| 261 | 6.4318 | 6.8141 | **5.7764** | **−0.6554** |
| 521 | 6.0381 | 6.5579 | **5.6275** | **−0.4106** |
| 781 | 5.8964 | 6.4144 | **5.5434** | **−0.3530** |
| 1041 | 5.8288 | 6.3392 | **5.4876** | **−0.3412** |
| 1301 | 5.7982 | 6.3040 | **5.4519** | **−0.3463** |

**到同一验证 loss 所需步数**：

| 目标 val | AdamW | Muon（退火） | **Muon（固定）** |
|---|---|---|---|
| 6.40 | 521 | 1041 | **261** |
| 6.20 | 521 | **未达到** | **261** |
| 6.00 | 781 | **未达到** | **261** |
| 5.90 | 781 | **未达到** | **261** |
| 5.80 | 1301 | **未达到** | **261** |
| 5.70 | **未达到** | **未达到** | **521** |
| 5.60 | **未达到** | **未达到** | **781** |
| 5.55 | **未达到** | **未达到** | **781** |

**每步成本（中位）**：AdamW **3.333 s** · Muon 固定 **4.568 s** ⇒ 成本比 **1.37×**。

#### 7.33.4 净加速

```
步数比（到 val 5.80）   1301 / 261 = 4.98×
每步成本比              4.568 / 3.333 = 1.37×
⇒ 净加速                4.98 / 1.37 = 3.57×
```

**⇒ 31 小时 → 约 8.7 小时。**
**⇒ 用户一开始设的目标是"压到 9 小时里"——达到了。**

#### 7.33.5 必须保留的限制

1. **单种子**（计划要求 ≥3）；
2. **只有验证 loss，没有生成质量评估**；
3. **1500 步**——步数比在长跑里可能变化（Muon-vs-AdamW 的早期对照曾显示差距随长度收窄）；
4. **8.7 小时是按单一 loss 目标外推的**，不是完整 3000 步训练的实际墙钟；
5. **Muon 的 lr=0.02 从未调优**——固定 lr 是从 harness 照搬的，不是搜出来的。

#### 7.33.6 只推荐一个下一步

**按 3.57× 的配置跑一次完整的 3000 步训练，并同时做生成质量评估。**

理由：加速比现在是在 1500 步、单种子、只看验证 loss 的条件下测的。要把它当作可信的训练
配置，需要（a）完整长度下的实际墙钟，（b）生成质量不掉，（c）至少第二个种子。这三件事
可以一次跑完，而在此之前不该把 8.7 小时当作既定事实。

'''

m = re.search(r'^### 7\.33 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.33'
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
