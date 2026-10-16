"""Append 7.59: the instantiation matrix, and what a library's job actually is.

The user reframed the problem: a library cannot choose the user's model, so it has to make a range of
models faster without knowing what they are. That rules out everything based on shrinking the problem
and leaves a specific and repeatable shape of work, which this section names and then enumerates.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.59 库的视角：实例化矩阵，以及"让一堆模型都快"到底是什么活

#### 7.59.1 先修正一个定位错误

**我在 7.58 里说："我们的 GEMM 是计算受限，所以只拿 2×；4.5× 是带宽密集场景，而我们不是那个场景。"**

**⇒ 这个定位对【训练】成立，对【库】是反的。** 因为：

```
bnb 的用户主要跑【推理】
而推理 batch 小、序列长 ⇒ 算术强度低 ⇒ 【就是带宽受限的】
⇒ ★【所以对库来说 4.5× 才是那个数字,2× 才是特例】
```

#### 7.59.2 而库不能决定模型 ⇒ 能做的只有一类事

**用户模型不可选 ⇒ "缩小问题规模"这条路对库不成立。剩下的活有一个固定形状：**

```
① 某个内核【只对新指令集】提供了实例      （例如 AVX512-BF16）
② 在旧硬件上【静默回退】到通用慢路径      （不报错,只是慢）
③ 而回退是无声的 ⇒ 用户不知道自己在慢 10 倍
④ ★【找到并填掉这些静默回退 = 库的核心工作】
⇒ 而它对【所有走那条路的模型】生效,完全不需要知道对面是什么
```

#### 7.59.3 `cpu_ops.cpp` 的完整实例化矩阵

| 函数 | 实例数 | 覆盖 | 状态 |
|---|---|---|---|
| `dequantizeBlockwise4bitCpu` | **6** | float/fp16/bf16 × FP4/NF4 | ✓ 全覆盖 |
| `dequantizeBlockwise8bitCpu` | **3** | float/fp16/bf16 | ✓ 全覆盖 |
| **`gemv_4bit_inference`** | **2** | **【只有 bf16 × FP4/NF4】** | ✗ **我们已填（见下）** |
| **`tinygemm_kernel_nn`** | 声明通用 | **只特化了 `bf16_t`** | ✗ **未填** |

**⇒ 模式清楚**：**反量化函数三种类型全有（因为它们简单），而矩阵乘内核只有 bf16（因为它们才需要 SIMD 特化）。**

**⇒ 上游作者瞄准的是 AVX512-BF16 硬件 ⇒ 只给 bf16 写了快路径 ⇒ 其他所有机器静默回退。**

**⇒ 而"反量化全都有"正好解释了回退路径的形态**：`把整个权重反量化 + 跑稠密 GEMM` ——
**而那正是实测 12.3 ms 对 1.16 ms 的那条路。**

#### 7.59.4 而我们【已经填过一个】，值 10.6×

`bitsandbytes/csrc/avx2_gemv_4bit.h` 的头注释完整记录了它：

```
上游提供的两个实例都是 AVX-512 + AVX512-BF16 路径
⇒ 没有 AVX512-BF16 的 CPU 上,代码回退成"反量化整个权重矩阵 + 稠密 GEMM"
⇒ 而那覆盖【每一台 AVX2-only 机器】:
   Zen2(Ryzen 4000/5000U) · Comet Lake(i5-10400 一类) · 以及几乎所有消费级笔记本

在 R5-4500U 上实测（同形状、同进程）:
   反量化+稠密GEMM 回退 : 12.3 ms/次
   我们的融合 AVX2 内核  :  1.16 ms/次      ⇒ ★【10.6×】
```

**⇒ 所以"让一堆模型都快"的第一个真实收益，已经在仓库里了，而且是 10.6×。**

#### 7.59.5 而它还记下了一条可复用的技术洞察

**为什么 4-bit 是 AVX2 上的甜点，而 2-bit 不是：**

```
16 项的 NF4/FP4 表【正好填满 pshufb 的 16 字节索引空间】
⇒ 一条 _mm256_shuffle_epi8 解决 32 个码字,LUT 查表免费
⇒ 而 2-bit 的表只有 4 项,填不满那个索引空间
⇒ 必须【另付】层级选择的代价
```

**⇒ "哪种量化在 AVX2 上划算"不是经验猜测，是由 `pshufb` 的索引宽度决定的。**
**⇒ 而这正是一条该写进库文档、而不是让每个用户自己撞的东西。**

#### 7.59.6 所以库的优化清单（按"缺了就慢一个数量级"排序）

| # | 缺口 | 类别 | 值 |
|---|---|---|---|
| **①** | `gemv_4bit_inference` 只有 bf16 | 静默回退 | ★ **10.6×（已填）** |
| **②** | `tinygemm_kernel_nn` 只特化 bf16 | 静默回退 | 未测，同一形状 |
| **③** | **整个后端零个整数点积指令** | **指令未用** | 2~4.5× |
| ④ | fp32/fp16 输出的 4-bit GEMV | 静默回退 | 头注释标为"机械扩展" |
| ⑤ | 按形状选内核（n=384 到 2048 差 1.5×） | 分派 | 实测分段 |

**⇒ ①②④ 是同一类：模板通用而实例化只覆盖新硬件。**
**⇒ ③ 是另一类：有实例，而实例里没有该用的指令。**

**⇒ 而两类共同点：****都是无声的** —— **没有任何报错，只是慢。**

#### 7.59.7 而库专属的方法论

```
对一个模型,最优配置可以手调
对一堆模型,只能靠【按形状/类型分派】
⇒ ★【而分派表必须【扫】出来,不能推出来】
   今天已撞到三次:
     · n=384→2048,GFLOPS 从 364 掉到 235 ⇒ 阈值在哪?
     · 分块宽度 64→1024,最优是 512 而 684 反而最差(0.689×)⇒ 【非单调】
     · 4-bit GEMV 在 bf16 下有实例,fp32 下没有
⇒ 非单调 ⇒ 任何"推论出来的最优"都会在某一段上错
```

**⇒ 所以库的正解是：建一张【形状 × 类型 × 后端】的分派表，用实测填它，并配一个能持续扫的基准工具。**

'''

m = re.search(r'^### 7\.59 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.59'
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
