"""Append 7.57: from the library's point of view, the quantised path has no quantised arithmetic.

bnb's CPU backend contains no integer dot-product instruction at all. The AVX2 histogram is entirely
FP32 arithmetic -- mul_ps, fmadd_ps, add_ps -- plus bit manipulation to unpack 4-bit values into FP32
lanes. So the quantised path is a dequantise-then-FP32-GEMM, which is why quantisation measured
slower than dense rather than faster.

That is a real library defect with a real 2x behind it, and it is worth separating from what it does
for this project's training.
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

SEC = '''### 7.57 从 bnb 的角度：量化路径里没有量化算术

#### 7.57.1 发现

查 `bitsandbytes/csrc/cpu_ops.cpp` 的 AVX2 指令直方图：

| 指令 | 次数 | 类别 |
|---|---|---|
| `_mm256_mul_ps` | 80 | FP32 乘 |
| `_mm256_set1_ps` | 68 | FP32 |
| `_mm256_loadu_ps` | 38 | FP32 |
| `_mm256_add_ps` | 31 | FP32 |
| `_mm256_fmadd_ps` | 21 | FP32 FMA |
| `_mm256_castsi256_ps` | 30 | **位操作（4-bit 解包到 FP32 通道）** |
| `_mm256_and_si256` | 20 | 同上 |

**而整数点积指令【一个都没有】：**

```
_mm256_maddubs_epi16   ★ 不存在
_mm256_madd_epi16      ★ 不存在
_mm256_dpbusd          ★ 不存在
_mm256_mullo_epi16     ★ 不存在
_mm256_madd52          ★ 不存在
_mm256_mul_epi32       ★ 不存在
```

**⇒ 所以 bnb 的 CPU 量化路径是【解包成 FP32 再算】⇒ "FP32 GEMM + 额外解包开销"。**
**⇒ 这解释了那个记录在案的"量化 0.80~0.85×"—— 量化反而更慢，而那不是量化的性质，是实现的。**

（`avx2_gemv_4bit.h` 是本项目自己加的文件，头注释写着 "upstream gap, measured"。）

#### 7.57.2 而真正的 2 倍在哪

```
FP32 FMA:   一条指令 8 个 float     ⇒ 长度 K 的点积要 K/8 条指令
INT8:       一条 32 组 int8 乘加    ⇒ K/32 条 maddubs + K/32 条 madd
                                    = K/16 条指令
⇒ ★【2× 更少的指令】

★ VPMADDUBSW 属于 AVX2，【Zen2 就有】
  （需要 AVX512-VNNI 的是 VPDPBUSD，那个才是 Zen4）
```

**⇒ 所以修复方向明确**：**用 `_mm256_maddubs_epi16` + `_mm256_madd_epi16` 写一条真正的
INT8 GEMM**，替换现在的"解包→FP32"路径。

#### 7.57.3 但必须说清它【对本次训练】值多少

| 路线 | 对一步的作用 | 效果 |
|---|---|---|
| INT8 GEMM 2× | GEMM 2.36 s ⇒ 1.18 s | 一步 4.53 ⇒ **3.32 s = 1.36×** |
| **核显卸载** | GEMM 整个交给核显 | 一步 ⇒ **2.55 s = 1.78×** |

**⇒ 两者打的是同一块（GEMM），【不能叠加】⇒ 取大的 ⇒ 仍是 1.78×。**

**⇒ 而更关键**：**卸掉 GEMM 之后，瓶颈是 CPU 那 47.6%（attention 18.3% + elementwise 19%），
而 INT8 改不了它 ⇒ 天花板不变。**

#### 7.57.4 所以这条的价值定位

**它是一个【真实的库级缺陷修复】：一个叫"量化"的路径没有量化收益，这件事本身就不对，
而且 2× 是从指令集里现成拿的。**

**⇒ 但它的收益属于 bnb 这个库，不属于这个项目的训练。** 而这一点要说清楚，
**否则就会重复今天已经犯过的错：把一个真实的数字，套到它不适用的问题上。**

**⇒ 而如果目标是【让 bnb 更好】，这是清单上最实在的一条 —— 比 4-bit 优化器更直接**
（那个已经做好了，而且它省的是内存不是时间）。

#### 7.57.5 bnb 角度的完整清单

| 项 | 状态 | 对训练 | 对库 |
|---|---|---|---|
| **INT8 GEMM（VPMADDUBSW）** | ✗ **未实现，2× 现成** | 1.36×（与核显重叠） | ★ **最大的一项** |
| 4-bit 优化器 | ✓ 已实现已验证 | ~0（省内存） | 已完成 |
| RMSNorm 两遍 | ✓ 已做 | — | 已完成 |
| 量化路径的速度 | ✗ **实测 0.80~0.85×** | — | **缺陷，见上** |
| NF2/NF3 | ✓ 有分支 | — | 待评 |

'''

m = re.search(r'^### 7\.57 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.57'
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
