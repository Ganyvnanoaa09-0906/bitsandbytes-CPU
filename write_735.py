"""Append 7.35: the experiment queue for Thursday, as the user requested.

The user has classes all day Thursday and has offered the machine for the whole day, so the
queue is ordered by what can actually be finished in one sitting rather than by theoretical
value alone. Each entry carries a hypothesis, a decision rule fixed in advance, a cost, and a
risk, because the whole point of the earlier failures was that judgements made after seeing
the numbers tend to bend.

Priority is decided by expected wall-clock gain per hour of work, not by elegance. Two entries
are explicitly marked as not worth doing, with reasons, so a future round does not rediscover
them.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.35 待做实验队列（用户指定：周四执行）

用户周四满课，机器可整日占用。因此本节按**"一天能做完多少"**排序，而不是单看理论价值。

#### 7.35.1 队列

| # | 实验 | 预期 | 成本 | 风险 | 判据（事先固定） |
|---|---|---|---|---|---|
| **①** | **Muon 隔步正交化**（每 2 步做一次 NS） | **1.2~1.5×** | 1 小时实现 + 2 小时 A/B | 中 | 验证 loss 与逐步版之差 ≤ 0.05（在 1500 步、loss 5.5 附近）⇒ 采用；否则回滚 |
| **②** | **QKV 融合**（三个投影合成一个） | 1.1~1.2× | 2 小时实现 + 2 小时 A/B | 低 | 前向输出与拆分版逐位误差 < 1e-5，且整步省时 > 3% ⇒ 采用 |
| **③** | **flash attention 的 CPU 后端** | ≤ 9.4% | 30 分钟 | 低 | 若 SDPA 比手写慢 20% 以上 ⇒ 换；否则关闭该方向 |
| **④** | 若 ①② 都在当天完成且仍有时间 | — | — | — | **不追加新实验**，改为把 ①②③ 的结果写进报告并复核 |

#### 7.35.2 ① 的设计细节（Muon 隔步正交化）

**假设**：Newton-Schulz 的作用是给出一个谱平坦的更新方向，而这个方向在相邻步之间变化
很小。若如此，每 2 步正交化一次，中间步复用上一次的方向，质量应当不变。

**已知依据**：NS 占 Muon 一步的 38%（2.42 s / 6.42 s，7.31.4）。Muon 论文本身允许这种近似。

**已试过而失败的**：批量化（0.89×）、真异步（比串行差 5%）、减少迭代到 3（条件数 ×12
但训练等价）。**这三条都是在"把每一步做得更快"，而本条是"少做几步"。** 方向不同。

**实现**：`--muon-ns-every N`，N=1 为现状。NS 的动量缓冲保持不变，只在第 N 步调用
`zeropower_via_newtonschulz5`，其余步复用缓存的 `d`。

**判据**：1500 步、单种子，验证 loss 与逐步版相差 ≤ 0.05 ⇒ 采用并延长验证；否则回滚。

#### 7.35.3 ② 的设计细节（QKV 融合）

**观察**：配置里有 `fuse_qkv`，当前为 `False`。而 **MLP 已经在做这件事**——`mlp.w12` 的
宽度是 **2736** 而不是 2048，说明 gate 与 up 已经拼成一个矩阵。注意力那边还没拼。

**假设**：三个 (512, 512) 和 (256, 512) 的投影合成一个 (d, d + 2·kv) 的 GEMM，在 CPU 上
效率更高。依据是同一份内核只改尺寸时的落差：n=384 → 371 GFLOPS，n=2048 → 230。大 GEMM
的每字节计算量更高，而这台机器的瓶颈正是取数。

**判据**：前向输出与拆分版逐位误差 < 1e-5（先验证正确性），且完整一步省时 > 3%。

#### 7.35.4 ③ 的设计细节（注意力后端）

**唯一的未知大项**：attention 占一步的 18.8%，而它的 CPU 实现效率**从未测过**。

**做法**：T=1024 · 8 头 · 4 KV 头 · d=64，对比 `F.scaled_dot_product_attention` 与手写
matmul + softmax + matmul。30 分钟。

**上限 9.4%**（若快 2 倍）。若 SDPA 已经最优，这条关闭。

#### 7.35.5 明确不做的（附理由，避免下一轮重新发现）

**AVX 加速库**：torch 底层是 MKL/oneDNN，我们的 mm 已达该形状独立水平的 **79~85%**
（218 对比 218~235 GFLOPS）。自己写很难更好，而收益只在前向的 53.6% 里再挤出几个百分点。

**低秩近似 W ≈ UV**：成本从 mnk 降到 (m+n)kr，看着很美。但**我们实测过量化到 2 bit 才
损失 0.117 的 loss**，同等压缩率下低秩会掉得多得多；而且模型已用满秩训练，中途替换会破坏
已有的动量状态。数学成立，工程不值得。

**bf16/fp16 前向**：**AVX2 没有 fp16/bf16 的乘加指令**，只能省内存不能省算力，而转换本身
有成本。除非换到有 AVX-512 或 AVX512-BF16 的平台。

**核显**：7.28 与 7.34 已定罪——LDS 缺口 8 倍（片上），UMA 无独立显存（并发五条全负）。
**DDR5 平台可能改变并发结论，但那是换硬件，不是换代码。**

#### 7.35.6 排在最前面的那条数学判断

这一轮最有价值的结论不是任何单个实验，而是一个方向性的认识：

**在固定架构与数据下，GEMM 的 FLOPs 是固定的**（2·m·n·k·层数·3）。数学上能减的只有
三件事——**达到目标质量所需的步数、每步的 FLOPs、以及表示精度**。第三条在 AVX2 上走不通。

**而最大的一次收益已经拿到了**：Muon 没有让任何一次 GEMM 变快，它让步数少了 **4.98 倍**。
**⇒ 下一个同量级的想法也应该在这一层，而不在 GEMM 内核里。** 队列里的 ①② 都只是个位数到
1.5 倍的量级，这一点必须诚实地写在前面。

'''

m = re.search(r'^### 7\.35 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.35'
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
