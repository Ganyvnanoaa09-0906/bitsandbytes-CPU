"""Append 7.41: the clean re-measurement results, and what they change.

Ran 07:59-08:16 on 10-15 with zero Python processes at start and no allow=True exemption. Four
items from 7.36, each with a contaminated predecessor.

The headline: every time-based figure was 10-13% too high, and the iGPU's 300 GFLOPS was never
real. The proportions, by contrast, held up almost exactly -- which is the useful part, because
the qualitative conclusions were built on proportions.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

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

SEC = '''### 7.41 干净重测结果（10-15，机器完全空闲）

07:59:56 启动，启动前确认 **0 个 python 进程**，四个脚本全部通过 `measure_guard` 且**未使用
`allow=True` 豁免**。原始日志在 `evidence/clean_2026-10-15/`。

#### 7.41.1 ① 步时分项：旧值偏高 10~13%

| 阶段 | 旧值（10-12） | **新值（10-15）** | 差 |
|---|---|---|---|
| AdamW 前向 | 1.166 | **1.0593** | −9% |
| AdamW 反向 | 2.647 | **2.2801** | −14% |
| AdamW 优化器 | 0.138 | **0.1066** | −23% |
| **AdamW 合计** | **3.952** | **3.4460** | **−13%** |
| Muon 前向 | 1.155 | **1.0385** | −10% |
| Muon 反向 | 2.706 | **2.2783** | −16% |
| Muon 优化器 | 2.562 | **2.4941** | −3% |
| **Muon 合计** | **6.423** | **5.8109** | **−10%** |

**⇒ 所有时间口径的结论都要按 −10~13% 修正。** 而 **Muon 的成本比从 1.63× 变为 1.686×**，
因此**时间口径的净加速从 1.96× 修正为 3.19 / 1.686 = 【1.89×】**。

**⇒ 但步数比（3.19×）不受影响**，因为它比的是 loss 曲线上的位置。**修正只动时间那一栏。**

#### 7.41.2 ② 算子占比：比例稳定，绝对值下降

| 算子 | 旧 | **新** | 差 |
|---|---|---|---|
| `aten::mm` | 53.56% / 6438.6 ms | **52.39% / 5907.9 ms** | −8% |
| attention（前） | 13.93% / 1675.0 | **13.43% / 1514.5** | −10% |
| `aten::mul` | 10.74% / 1290.7 | **11.61% / 1309.0** | +1% |
| attention（后） | 4.87% | **4.87% / 549.7** | 0% |
| `aten::sum` | 2.11% / 253.8 | **2.32% / 261.2** | +3% |
| `aten::copy_` | 2.14% / 257.3 | **1.80% / 203.4** | −21% |
| `aten::pow` | 1.28% / 154.4 | **0.96% / 108.2** | −30% |
| `aten::cat` | 1.23% / 148.4 | **1.30% / 146.1** | −2% |

**⇒ 分类比例几乎不变：GEMM 52~54% · attention ~18% · elementwise ~19% · 其余 ~9%。**

这很重要，因为 **7.34 和 7.39 的策略判断是建立在比例上的**（"mm 已达该形状的 79~85%，剩余
空间小"、"attention 占 18.8% 而效率未测"）。**比例没变 ⇒ 那些判断不用改；只有引用绝对秒数的
地方需要 −6~30%。**

`aten::mul` 仍是 490 次，仍是 RMSNorm 内部展开——**这一点也没变。**

#### 7.41.3 ③ 内存：延迟 259.3 ns

| 项 | 旧 | **新** |
|---|---|---|
| 随机访问延迟 | 277.6 ns | **259.3 ns** |
| 线程扩展 | 1⇒16.2 · 3⇒24.75 · 6⇒27.17 | （本次未重复） |

**同一量级，仍然远高于正常 DDR4 的 70~90 ns。** 结论不变：**这是片外那一层的问题。**

**而 stride 测试仍然是坏的**（1⇒3.51 · 8⇒3.44 · 64⇒3.90 · 512⇒14.85）。**原因已查明并写在
7.34：分母用了 `sub.nbytes`，而大步长读的元素少得多，所以那些 GB/s 不可比。**
**这条测试应当删除或重写，不该继续引用。**

#### 7.41.4 ④ 核显 GEMM：**300 GFLOPS 从来不存在**

| 形状 | 新测 GFLOPS | 报告记录的 | 错误 |
|---|---|---|---|
| 1024×2736×512 | **155.5** | 300 | 914 / 2801664 错 |
| 1024×512×2736 | **152.9** | 300 | 1079 / 524288 错 |
| 2048×2048×2048 | **163.1** | 300 | 1793 / 4194304 错 |
| 1024×512×512 | **128.2** | — | 1460 / 524288 错 |

**⇒ 核显在本机真实形状上的单内核 GEMM 上限约 155 GFLOPS，而且结果算错。**
**⇒ 它比 CPU 的 218~240 GFLOPS 还低**，与 7.28.3 记录的"300 GFLOPS，CPU 的 1.51×"相反。

**300 这个数字来自一个已经不存在于仓库中的内核版本**（7.28.5 已记录为不可复现）。**本次是第二次
确认，且是在机器完全空闲的条件下。** 因此：

* **报告中所有"核显 300 GFLOPS"的引用作废**，包括 7.28 的正面结论；
* **核显在非立方形状上的可用性也需要重新表述**——它仍能跑（不像 CLBlast），但**比 CPU 慢**；
* ★ **而它的错误（0.03%~0.15% 的元素）意味着即使这个 155 也不能作为可信上限。**

#### 7.41.5 四项重测之后，哪些结论要改、哪些不改

| 结论 | 是否受影响 |
|---|---|
| **Muon 步数比 3.19× / 4.98×** | ✗ **不受影响**（比的是 loss 曲线位置） |
| **Muon 时间口径 1.96×** | ✓ **修正为 1.89×** |
| **GEMM 占 53.6%，已达该形状 79~85%** | ✗ 比例未变，判断不变 |
| **attention 占 18.8% 而效率未测** | ✗ 比例未变，仍是最大未知项 |
| **elementwise 18.8%，490 次 mul** | ✗ 未变 |
| **内存是片外瓶颈（延迟 259~277 ns）** | ✗ 未变，仍是正常值的 3 倍 |
| **★ 核显 300 GFLOPS** | ✓ **作废，实际约 155 且算错** |
| **核显并发五条全负** | ✗ 未变（那测的是并发，不是单内核速度） |
| **stride 测试** | ✓ **作废（分母错），应从报告中移除** |

**⇒ 好消息是：所有【定性】结论都活了，因为它们是建立在比例和结构上的；**
**⇒ 坏消息是：所有【绝对时间】要下调 10~13%，而核显那条正面结论彻底没了。**

#### 7.41.6 方法论的第四次确认

**这是同一种错误的第四次实例**：比较的基础不对，而比较看起来很合理。

1. 非方阵用了 `‖XᵀX − I‖`（不适用）⇒ 错判"迭代数不能减"；
2. 用熵判断出图好坏（噪声熵最高）⇒ 错判 MaskGIT 有效；
3. 用错的解码器采样别的模型 ⇒ 造出一条不存在的"结构线索"；
4. **本次：在别的负载下测"干净"基线 ⇒ 所有时间偏高 10~13%。**

**四次的形状完全一样：测量本身没有报错，只是量错了东西。**

**⇒ 所以 7.36 那条规则（凡日志记录了并发负载的必重测）应当升级为更一般的纪律：**
**任何数字在进入结论之前，先问一句"它量的真的是我要的那个东西吗"。**

'''

m = re.search(r'^### 7\.41 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.41'
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
