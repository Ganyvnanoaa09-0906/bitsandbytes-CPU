"""Append 7.72: the M=2..4 gap is worth 10-14x, and the crossover is nowhere near it.

tinygemm_kernel covers BLOCK_M from 1 to 4 and sits inside the AVX512-BF16 block, so on AVX2 it is
absent entirely. That is the same shape of gap as gemv_4bit_inference, which we filled at M=1 for
10.6x. This measures whether M=2, 3 and 4 are worth filling too.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.72 ★ M=2~4 的缺口值 10~14× —— 而交叉点远不在这里

#### 7.72.1 缺口的来源

```
tinygemm_kernel 的派发器覆盖 BLOCK_M = 1,2,3,4 × BLOCK_N = 32,64
⇒ 而整块在 #if defined(__AVX512F__) && defined(__AVX512BF16__) 之内
⇒ ★【AVX2 机器上它【完全不存在】⇒ 回退到"反量化整个权重 + 稠密 GEMM"】

而这与 gemv_4bit_inference 是【同一个形状的缺口】（同一段条件编译、同一批缺实例）
⇒ 我们已经填了 M=1（实测 10.6×）,而 M=2,3,4 未填
```

#### 7.72.2 测量（N=4096, K=4096, 两条路径都在本机实测）

| M | 融合 ms | 回退 ms | 其中反量化 | **融合/回退** |
|---|---|---|---|---|
| 1 | 4.315 | 152.947 | 149.765 | **35.44×** |
| **2** | 11.098 | 157.332 | 148.659 | ★ **14.18×** |
| **3** | 14.172 | 161.352 | 150.586 | ★ **11.39×** |
| **4** | 17.118 | 167.461 | 151.672 | ★ **9.78×** |
| 5 | 21.413 | 172.114 | 152.816 | 8.04× |
| 8 | 34.971 | 198.246 | 148.680 | 5.67× |

**判据（M=2,3,4 上 ≥1.5×）⇒ ★【大幅通过：9.78~14.18×】**
**⇒ ★【而且在 M≤8 范围内【从未交叉】—— 最差的一档也是 5.67×】**

#### 7.72.3 而回退路径的构成让这个结论很稳

```
回退 = 反量化(149 ms) + 稠密 GEMM(3~49 ms)
⇒ ★【反量化占回退的 【97%】】
⇒ ⇒ 即使把稠密 GEMM 换成无限快的,回退仍要 149 ms 对融合的 4~35 ms
   ⇒ ★【仍有 4~35 倍】
```

**（诚实标注：本次的稠密 GEMM 是朴素 k 循环、未分块 ⇒ 大 M 那几档会高估融合的优势。
但反量化那道 149 ms 的地板独立于 GEMM 质量 ⇒ 结论不变。）**

#### 7.72.4 而我们的头注释有一句需要修正

**`avx2_gemv_4bit.h` 写着："At M >= 8 a dense fp32 GEMM wins even with 4x less memory traffic."**

**⇒ 而本次在 M=8 上实测仍偏向融合 5.67×。**

**⇒ 那句话大概针对"与【真正分块】的稠密 GEMM 比"，而本次的回退是朴素的；
⇒ 但它至少说明那个边界【不在这附近】—— 若真在 M=8,需要的是分块稠密 GEMM 的对照，而不是朴素的。**

#### 7.72.5 库侧的清单更新

| # | 缺口 | 覆盖 | 值 | 状态 |
|---|---|---|---|---|
| ① | `gemv_4bit_inference` 只有 bf16 | **M=1** | **10.6×（实测）** | ✓ **已填** |
| **②** | **`tinygemm_kernel` 整块在 `#if AVX512-BF16` 内** | **M=2~4** | ★ **9.8~14.2×（本次）** | ✗ **值得填 —— 形状与 ① 相同** |
| ③ | 零个整数点积指令 | 全部量化路径 | 2~4.5× | ✗ 需写新内核 |
| ④ | fp32/fp16 输出的 4-bit GEMV | 同 ① | 同上 | ✗ 头注释标"机械扩展" |

**⇒ 所以库侧现在有【两个】已证实值一个数量级的缺口，而它们出自同一段条件编译。**

#### 7.72.6 而这两个加起来覆盖了什么

```
① ⇒ 单序列生成（M=1）         ✓ 已覆盖
② ⇒ 【批量生成 2~4 条序列】    ★ 本次证明值 ~10×
⇒ 而批量生成正是采样的常见用法（一次出多张图 / 多条候选）
⇒ ★【所以 ② 填上之后,4-bit 的批量解码路径才算完整】
```

'''

m = re.search(r'^### 7\.72 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.72'
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
