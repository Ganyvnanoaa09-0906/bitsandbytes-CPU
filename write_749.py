"""Append 7.49: a consolidation of 10-15, once the day's work spread across nine sections.

The report is past 8300 lines and a reader arriving at today's work has to walk 7.41 through 7.48
to assemble the picture. This section is that picture in one place, and it also names a pattern
that has now appeared three times and is worth treating as a standing checklist item rather than a
coincidence.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.49 10-15 汇总：关掉的门、睡觉的资产、以及剩下的一条路

本节把分散在 7.41~7.48 的结论收在一处。

#### 7.49.1 今天关闭的方向（都带数字）

| 方向 | 结果 | 依据 |
|---|---|---|
| ~~torch.compile~~ | **0.9768×（−2.4%）** | 编译仅 6.1 秒 ⇒ Inductor 大部分算子回退；而 490 次 `mul` 已在 RMSNorm 融合核心里 |
| ~~CPU attention 手写分块~~ | **比 SDPA 慢 5.8×** | 14.23 ms 对 83.05 ms。**要抄的东西 PyTorch 已经抄完了** |
| ~~GQA 展开消除~~ | **仅占 0.3%** | 80 MB / 1150 ms。生成时值 44%，但生成走 cache 路径已避开 |
| ~~核显与 CPU 并用~~ | **0.76× CPU 单独** | 即使核显单内核快 1.34 倍。**共享 DDR4 是根因** |
| ~~寄存器分块~~ | **与 swizzle 互相替代** | 8×8+swizzle = 320 < 只做 swizzle 的 350 |
| ~~1024 token 太少~~ | **排除** | 7.45 的 grid16 对照 + 7.48 的重建质量 |
| ~~分辨率/码本容量~~ | **排除** | 重建接近无损；码本 1024/1024 使用、熵 96.6% |

#### 7.49.2 今天成立的（同样带数字）

| 结论 | 数字 |
|---|---|
| **Swizzle 是唯一抄成功的一项** | **2.0×**（183 → 361.5 GFLOPS），核显单内核 **超过 CPU 1.34 倍** |
| **而它之所以成功，是因为动的是地址映射** | 此前七次内核尝试改的都是访问方式或访问量 |
| **核显真实数字** | **361.5 GFLOPS**（报告曾记 300，那从来不存在） |
| **干净重测：时间口径全部下调** | **10~13%**；Muon 净加速 **1.96× → 1.89×**（步数比 3.19× 不受影响） |
| **而【比例】全部存活** | GEMM 52~54% · attention 18.3% · elementwise ~19% |
| **分词器天花板很高** | 重建几乎无损、码本全用、熵 96.6% |
| **四个速度杠杆全关** | GEMM 79~85% 形状最优 · attention 已最优 · elementwise 已融合 · 核显否掉 |

#### 7.49.3 ★ 一个出现了三次的模式：答案在仓库里，但睡着

| 次 | 资产 | 状态 | 后果 |
|---|---|---|---|
| ① | **循环 Transformer** | 挂载了但 `random=False` | 变深度训练从未生效 |
| ② | **4-bit 优化器** | 实现+验证完成（0.658 ms 对 1.682 ms） | **从未接进训练** |
| ③ | **MaskGIT 完整管线** | `--maskgit` · `masked_loss` · `forward_masked` · `generate_masked` · `sample_ar --maskgit` | **从未被执行** |

**而第③次的误导性最强**：`i5build/maskgit1` 这个目录名说明"MaskGIT 跑过"，
**但它的 `vocab_size` 是 1024 而不是 1025** ⇒ 根本没传 `--maskgit` ⇒ 它是那个"双向注意力 +
完整真实序列输入 + loss 算在所有位置"的坏版本 ⇒ **模型能看见自己的答案。**

**⇒ 所以"跑过了"和"跑对了"是两件事，而目录名不能作为证据。**

**⇒ 这应当成为一条固定检查：** 每当要重复一个"以前做过"的实验时，
**先验证那次实验的配置里真的含有它声称使用的那个开关。**

#### 7.49.4 而 7.41.6 的纪律今天又触发了两次

**第六次**：并发脚本的正确性检查放在了内核第一次启动之前 ⇒ 读了没写过的缓冲区 ⇒ 假失败。

**第七次**：参考图代码调用了 VQVAE 的 `forward`，**而那是编码器，吃图像不吃 token** ⇒ 报错。

**两次都是同一个形状：检查/调用写在了【假设的接口或状态】上，而不是真实的那个。**

**⇒ 纪律：任何检查都要先确认被检查的对象【已经存在或已经运行过】；
任何调用都要先确认【接口的真实签名】。**

#### 7.49.5 剩下的一条路

```
分词器天花板高 ✓ · 码本健康 ✓ · 四个速度杠杆全关 ✓
⇒ 而 AR 出的是色块
⇒ 瓶颈在【模型或目标函数】,不在表示
```

**而栅格顺序自回归对图像本来就不合适**——预测下一个图块只能看前缀，**那要求相邻图块相关，
而照片里不相关**（文本相邻词强相关，图像相邻块弱相关）。

**⇒ 正在运行的 MaskGIT 是唯一一条有理论依据的方向**：双向上下文是图像的自然归纳偏置，
而且出图只要 16 次前向而不是 1024 次。

**而判读已经准备好两个参照**：

1. **7.48 的 VQGAN 重建图** —— 同一表示空间里"应该能达到"的样子。
   **模型出图明显差于它，差距就在模型身上，没有别的借口。**
2. **正在准备的过拟合 100 张诊断** —— 若连 100 张都记不住，问题在架构或采样流程，
   **而那就与数据量无关，加数据也救不了。**

#### 7.49.6 时间线

| 时间 | 事件 |
|---|---|
| 07:59 | 干净重测四项（机器完全空闲） |
| 13:22 | attention 对照 + v4 参考（2048³ 达 200 GFLOPS） |
| 13:25 | swizzle 扫描 ⇒ 均方 2.0× |
| 13:38 | 组合测试 ⇒ **替代而非叠加** |
| 13:41 | 并发重测 ⇒ **0.76×，核显关闭** |
| 13:46 | **★ MaskGIT 第一次被执行** |
| 13:52 | 分词器天花板检查 ⇒ **很高** |
| 16:15（预计） | MaskGIT 第一张出图（step 2000） |
| 21:35（预计） | MaskGIT 6000 步完成 |

'''

m = re.search(r'^### 7\.49 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.49'
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
