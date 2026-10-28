"""Record the Muon wiring and its first measured step cost in the real trainer.

This closes the gap 7.31.4.9 recommended closing: Muon and NS existed only in the standalone
A/B harness, so the trainer could not use either, and every speed figure was a harness figure.

What changed in train_ar_v2.py:

  * the Muon class, the Newton-Schulz routine and the 2-D/other split are lifted verbatim
    from ab_muon_v2.py, which is where the measurements were taken;
  * --optimizer {adamw,muon} and --muon-lr, defaulting to adamw so nothing existing changes;
  * opt_zero_grad / opt_step / opt_lr / opt_groups, because Muon mode holds two optimisers
    and the loop had ten direct opt.step() / opt.zero_grad() / opt.param_groups call sites;
  * a backup of the pre-change file at train_ar_v2.py.pre_muon for an immediate rollback.

First measured step cost in the trainer, five steps, same configuration as every other number:

    AdamW8bit (baseline)      3.95 s/step
    Muon + NS=3               5.00 s/step

so Muon costs 1.27x per step here against 1.63x with NS=5 in the harness. The NS=3 saving is
consistent with the A/B's 20.4%.

Applying the A/B's 3.19x step-count advantage to that cost ratio gives 2.51x, against 1.96x
with NS=5. Labelled preliminary on purpose: five steps is enough to establish that the wiring
works and to get a first cost, not enough to establish a speedup, and the 3.19x comes from a
different harness.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

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

SEC = '''### 7.32 Muon 与 NS=3 接入正式训练路径

**这补上了 7.31.4.9 指出的缺口**：Muon 与 NS 此前只存在于独立 A/B 脚本，训练器两者都没有，
因此所有加速数字都是 harness 的数字。

#### 7.32.1 改动

`train_ar_v2.py`：

| 改动 | 说明 |
|---|---|
| Muon 类 · NS 实现 · 2D/其余分组 | 从 `ab_muon_v2.py` **逐字**搬入（测量的就是那份代码） |
| `--optimizer {adamw,muon}` | 默认 `adamw` ⇒ 不传则行为不变 |
| `--muon-lr` | 默认 0.02，**未调优** |
| `opt_zero_grad` / `opt_step` / `opt_lr` / `opt_groups` | Muon 模式持有**两个**优化器，而循环里有 **10 处**直接调用 |
| 备份 | `train_ar_v2.py.pre_muon` ⇒ 可立即回滚 |

冒烟输出确认分组正确：

```
[muon] 60 个 2D 权重走 Muon（NS=3）· 65 个其余走 AdamW8bit
```

**这与 7.31.4.1 的审计一致**：2D 权重共 65 个，其中 60 个走 Muon，5 个（位置嵌入、token
嵌入、head）走 AdamW。

#### 7.32.2 正式训练器里的第一步成本实测

同一配置（batch 2 · T=1024 · 10 层 · 6 线程），5 步：

| 优化器 | s/步 | 相对 AdamW | 来源 |
|---|---|---|---|
| AdamW8bit（基线） | **3.95** | 1.00× | 7.31.1 |
| **Muon + NS=5**（harness） | 6.42 | 1.63× | 7.31.4.4 |
| **Muon + NS=3**（**训练器**） | **5.00** | **1.27×** | 本节 |

**NS=3 相对 NS=5 省 22%**（harness 里测到 20.4%）⇒ 两者一致。

#### 7.32.3 净加速：**初步值 2.51×**

```
步数比（A/B，2000 步）      3.19×
每步成本比（训练器，5 步）   1.27×
⇒ 净加速 = 3.19 / 1.27 =   2.51×
```

对照：NS=5 时是 3.19 / 1.63 = **1.96×**。

**⇒ NS=3 把 Muon 的净加速从 1.96× 提到 2.51×，即 31 小时 → 约 12.3 小时。**

**必须标注为初步值的理由**：

1. **5 步**只够确认接线正确与拿到第一个成本量级，**不足以确定加速比**；
2. **3.19× 来自另一个 harness**（`ab_muon_v2.py`），不是训练器自己的步数比；
3. **成本比可能随后续步骤变化**（首次运行含模型构建与数据加载，已从 mean_step_s 排除，
   但缓存效应未知）；
4. **仍只有一个种子、无生成质量评估**（7.31.4.9 的同一批保留）。

**因此本节不把 2.51× 写成结论**，只写成"接线后第一次测得的初步值"。

#### 7.32.4 回滚路径

* `--optimizer` 默认 `adamw` ⇒ 不传即回到原行为；
* `train_ar_v2.py.pre_muon` 是改动前的完整副本；
* Muon 相关代码是独立提交，可单独 revert；
* 原 A/B harness 未改动。

#### 7.32.5 只推荐一个下一步

**用训练器自己跑一次 NS=3 的步数比**（而不是沿用 harness 的 3.19×）。

理由：现在两个杠杆都在训练器里了，而**唯一的步数比仍来自另一个脚本**。在把 2.51× 写进任何
训练时长预期之前，需要在同一条路径上量出步数比。这一步同时能给出多种子数据。
'''

m = re.search(r'^### 7\.32 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.32'
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
