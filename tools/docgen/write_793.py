"""Append 7.93: a dense bf16 LLM for comparison, and the headline it produces.

The second model for the classification table, and the one that isolates what quantisation buys,
because it is the same kind of model without any.
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

SEC = '''### 7.93 ★★★ 未量化 LLM 的对照：量化让【更大的模型跑得更快】

#### 7.93.1 第二个对照模型

**`deepseek-coder-1.3b-base`（2.57 GB 磁盘 · 1.35B 参数 · **bf16，未量化**）**

**⇒ 选它的理由：它与 NF4 的 Qwen3 是同一类模型，唯一的区别是【没有量化】。
⇒ 所以两者的差就隔离出了"量化买到什么"。**

#### 7.93.2 ★★ 第一个对照：磁盘

```
NF4 Qwen3-1.7B    【2.03B 参数】· 磁盘 【1.59 GB】
bf16 DS-Coder     【1.30B 参数】· 磁盘 【2.57 GB】
⇒ ★【量化的 2.03B 比未量化的 1.3B 【小 38%】,而参数【多 56%】】
```

#### 7.93.3 ★★★ 而更出人意料的是【速度】

| | NF4 Qwen3-1.7B | bf16 DS-Coder-1.3B | **比值** |
|---|---|---|---|
| 参数 | **2.03B** | 1.30B | 0.64× |
| 磁盘 | **1.59 GB** | 2.57 GB | 0.62× |
| **prefill** | ★ **29.6 token/s** | 10.9 token/s | ★ **2.7×** |
| **decode** | ★ **9.4 token/s** | 5.65 token/s | ★ **1.66×** |

**⇒ ★【参数多 56% 的那个模型,prefill 快 2.7 倍、decode 快 1.66 倍,而且磁盘小 38%。】**

**⇒ 而原因是可算的：**

```
NF4:  每参数读 【0.55 字节】(4-bit + 每 64 元素一个 fp32 absmax)
bf16: 每参数读 【2 字节】
⇒ ★【3.6 倍的访存差】
⇒ 而这台机器上 LLM 推理是【访存受限】的 ⇒ 少读 3.6 倍的数据直接换成速度
```

#### 7.93.4 ★★ 而算子构成把这条解释了

**bf16（prefill）：**

| 类别 | 占比 |
|---|---|
| **矩阵乘** | ★ **96.5%** |
| elementwise/搬运 | 2.4% |
| 注意力 | 0.3% |
| 归一化/激活 | 0.3% |
| 其余 | 0.5% |
| **dequantize** | ★ **无** |

**⇒ 与 NF4 那个（矩阵乘 71.0% + 反量化 21.8% + 融合 3.2%）对照：**

```
未量化:  96.5% 全是【全精度 GEMM,读 2 字节/参数】⇒ 一次访存喂一次 FMA
NF4:     71% GEMM + 21.8% 反量化,而 GEMM【读 0.55 字节/参数】
⇒ ★【NF4 那条路【多花】了 21.8% 在反量化上,却仍然快 2.7× ——
   因为它把 71% 那段的访存减到了 1/3.6】
```

#### 7.93.5 ★★★ 而顺带抓到一个更该被知道的事：bf16 比 fp32 还慢

| dtype | 权重 | prefill | decode |
|---|---|---|---|
| **bf16（模型原样）** | 2.69 GB | **10.9 token/s** | **5.65 token/s** |
| **fp32（对照）** | 5.39 GB | ★ **14.4 token/s** | 2.74 token/s |

**⇒ ★【bf16 的 prefill 比 fp32 【慢 32%】,而它只占一半的内存。】**

**⇒ 原因：这颗 CPU 有 AVX2 但【没有 AVX512-BF16】⇒ bf16 矩阵乘被【模拟】。**

**⇒ ★★★ 而这条的适用范围很广：**

```
★ 现在绝大多数开源模型默认发【bf16】⇒ 而在 AVX2 机器上,
  把它们【转成 fp32】反而更快(代价是 2 倍内存),或者【量化成 NF4】快得多
⇒ ★【而"bf16 是默认"这件事在无 AVX512-BF16 的机器上是一个【静默的性能陷阱】——
   用户拿到一个 bf16 模型,装上就用,而它比 fp32 还慢】
⇒ ★ 这与本项目在别处反复遇到的形状一样:一个【看起来最优的默认值,实际是错的】。
```

#### 7.93.6 ★★ 五行分类表

| 负载 | 主子 | 次子 | 第三 | **GEMM-like 合计** |
|---|---|---|---|---|
| LLM decode（NF4） | 融合 4-bit GEMV **73.5%** | — | — | — |
| LLM prefill（NF4） | 稠密 GEMM **71.0%** | 反量化 21.8% | 融合 3.2% | **71%** |
| **LLM prefill（bf16）** | **矩阵乘 96.5%** | elementwise 2.4% | — | ★ **96.5%** |
| 扩散推理 | 卷积 **41.6%** | 矩阵乘 27.7% | 注意力 15.2% | **69.3%** |
| 扩散训练 | 卷积 **32.1%** | 矩阵乘 24.5% | 注意力 **25.5%** | **56.6%** |

**⇒ 五行里 GEMM-like 占 57~96.5% ⇒ ★【§7.89 那把钥匙（分块并行核显）的适用范围比预期更宽】**
**⇒ 而最宽的那一行恰恰是【未量化的 bf16】——因为反量化那 21.8% 在它那里不存在。**

#### 7.93.7 一条我自己的作业纪律：两个大任务不能并发

**本次训练作业第一次启动后 OOM 退出，原因是我把它挂在后台的同时跑了这个探针，
而探针要加载 2.7~5.4 GB 的模型 ⇒ 合计超过 15.4 GB。**

**⇒ ★【本项目的规矩写着"绝不并发跑两个大任务",而那条规矩正是为这台机器立的。
⇒ 而违反它的代价不是慢,是【静默的 OOM】—— 日志停在"=== 训练前出图 ===",没有 traceback。】**

**⇒ 所以那条规矩要写成可执行的判据，而不是一句原则：**

```
并发前先算:【A 的峰值 RSS】+【B 的峰值 RSS】< 可用内存 × 0.8
⇒ 本次: 2.6 GB(pipeline) + 5.4 GB(fp32 探针) = 8.0 GB,而当时可用【7.2 GB】⇒ 超了
⇒ 而不算这一步,失败的形态是"日志断了",不是"报内存不足"
```

'''

m = re.search(r'^### 7\.93 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.93'
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
