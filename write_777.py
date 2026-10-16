"""Append 7.77: the 4-bit optimizer's AVX2 path is the identified next piece, and its exact shape.

The docstring's last line says "the AVX2 version is the next piece of work". Locating it confirms both
the claim and the template to follow.
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
P = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'
s = io.open(P, encoding='utf-8').read()

SEC = '''### 7.77 4-bit 优化器缺的那条 AVX2 路径，以及它的准确形状

#### 7.77.1 缺口已在文档里写明

`adamw4bit.py` 头注释最后一句：

```
What it does NOT buy: speed. The kernel currently has no AVX2 path, so it runs
about 4x slower than the 8-bit AVX2 kernel. ... the AVX2 version is the next piece of work.
```

#### 7.77.2 而定位确认了它

```
pythonInterface.cpp:931   coptimizer_update_4bit_blockwise_cpu(...)      ← 导出的 4-bit 入口
cpu_ops.cpp:3341          static inline __m256i avx2_opt_quant_fast(...) ← 量化助手【已有】
cpu_ops.cpp:3361          optimizer_8bit_blockwise_avx2(...)             ← ★【8-bit 的 AVX2 版】
```

**⇒ ★【8-bit 有 AVX2 路径，4-bit 没有】—— 而这正是"比 8-bit 的 AVX2 内核慢约 4×"的来源。**

**⇒ 而这是今天那个形状的又一次出现：**
**一个标量路径，而旁边【同一个文件里】就有一条向量路径可以照抄。**

| 今天的实例 | 标量在哪 | 向量在哪 |
|---|---|---|
| 4-bit GEMV（AVX2） | — | ✓ 已填（我们做的） |
| INT8 GEMM | 全后端零整数点积指令 | ✗ 未写 |
| **4-bit 优化器** | `coptimizer_update_4bit_blockwise_cpu` | ★ **照 `optimizer_8bit_blockwise_avx2`** |

#### 7.77.3 实现规格

```
1. 读 cpu_ops.cpp:3361 的 optimizer_8bit_blockwise_avx2 —— 那是模板
   （它的结构、分块、absmax 处理、以及 kOptBlockSize=256 的用法）

2. 读 avx2_opt_quant_fast (L3341) —— 量化助手可能直接复用
   （8-bit 的量化是 8 个一块；4-bit 是【两个码一个字节】,所以要改打包/解包）

3. 写 optimizer_4bit_blockwise_avx2，与 8-bit 版同构,差别只在:
   · 状态打包:4-bit 两个码一字节,HIGH nibble = 偶数下标
     （这条契约 adamw4bit.py 的头注释里写了,且与 avx2_gemv_4bit.h 对齐）
   · absmax:每 kOptBlockSize=256 个元素一个 fp32
   · qmap:256 项,只读前 16

4. 接进 pythonInterface.cpp:931 的 coptimizer_update_4bit_blockwise_cpu:
   【运行时按 has_avx2_cpu() 分派】,标量版留作回退 ——
   ★【不要直接替换】:非 AVX2 机器(以及所有 ARM)还要用它

5. 正确性:必须与标量版【逐字节一致】
   （test_optimizer_4bit_cpu.py 已经是这么测的 ⇒ 把新路径挂进去重跑即可）

6. 性能:拿 bench_opt_4bit.py 对 8-bit AVX2 版比
   ⇒ 目标是从"慢 4×"变成【同量级】
```

#### 7.77.4 而它值多少，要说清楚

```
优化器占一步 【1~4%】（adamw4bit.py 头注释的估算,与 clean_step_timing 的 3.1% 一致）
⇒ 4× 的慢 = 步时的 0.27%~1.1% ⇒ 【即使修好,步时也只快 1% 左右】

★ 而真正的收益在【别的场景】:
   · 大规模微调（优化器占步时更高）
   · 以及"4-bit 优化器能不能当作纯内存工具用而不付速度代价"
⇒ ★【所以它是【工具完备性】的活,不是【训练加速】的活】
```

**⇒ 这与 7.76 的结论一致：4-bit 优化器在我们这个规模上是内存工具（省 1.3% 内存），
而它的速度短板【在真实训练里本来就不显著】—— 修好它是为了让它在大模型上不掉链子，
不是为了让我们这台机器更快。**

#### 7.77.5 状态

**规格已记，worktree 未动**（`D:\\work\\bnb-4bitopt`，分支 `perf/4bit-blockwise-optimizer`，
工作区干净）。**而本轮上下文已不足以完成"写 + 编 + 测"三步 —— 半写一个内核比没写更糟。**

'''

m = re.search(r'^### 7\.77 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.77'
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
