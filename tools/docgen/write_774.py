"""Append 7.74: the AVX2 4-bit fused path was already built, wired and tuned. 7.59 and 7.72 corrected.

I traced the call graph that 7.59 and 7.72 assumed rather than checked, and it does not have the shape
they describe. This is the fourth time today the answer was already in the repository, and the second
and third times I declared a gap without looking far enough.
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

SEC = '''### 7.74 ★ 更正 7.59 与 7.72：AVX2 的融合路径【早就建好、接好、调好了】

#### 7.74.1 调用图（实测，而非推断）

```
cpu_ops.cpp:
  L1867  gemv_4bit_inference<T, DATA_TYPE>(...)     ← AVX512-BF16 版,内部调 tinygemm_kernel
  L1955  template void gemv_4bit_inference<【bf16_t】, FP4>(...)   ← 实例化【只有 bf16】
  ★ 而它【在 _cpu_impl 里从来没有被调用】—— 它不是"回退",是【死代码】

  L2243  #if defined(__AVX2__) || ...
  L2246      if (has_avx2_cpu() && (K&1)==0 && K%blocksize==0 && (blocksize&7)==0) {
  L2278/2280     avx2_gemv_4bit_inference<T, FP4/NF4>(...)   ← ☑ AVX2 融合路径【是活的】
  L2284-2287     scalar_gemv_4bit_inference(...)             ← 真回退(标量)
```

#### 7.74.2 而 Python 层也全都接好了

```
ops.py:298  # AVX2 回退：用【所有 CPU 构建都导出】的 cgemv_4bit_inference_cpu_* 融合内核
ops.py:309      lib.cgemv_4bit_inference_cpu_fp32(...)
ops.py:317      lib.cgemv_4bit_inference_cpu_bf16(...)
ops.py:325      lib.cgemv_4bit_inference_cpu_fp16(...)
      ⇒ ★【三种 dtype 全部导出并接线】

ops.py:344  #     M=64  gemm_4bit 38.576 ms  vs 融合内核 30.382 ms  ⇒  1.27x
ops.py:345  #     M=256 gemm_4bit 75.532 ms  vs 融合内核 164.840 ms ⇒  0.46x  ← 融合反而慢
ops.py:360  # ⇒ 融合内核一直赢到 M=64,default 要到 M≈96 才反超
ops.py:361  # ★ 阈值原先写 16 ⇒ 【M=17~64 被错误地分给 default,白慢最多 3.4x】
ops.py:368  _GEMM_4BIT_FUSED_MAX_M = 64
```

#### 7.74.3 所以两条更正

| 版本 | 说法 | 更正 |
|---|---|---|
| **7.59** | "AVX2 机器静默回退到反量化+稠密 GEMM" | ✗ **`_cpu_impl` 从不调那个 AVX512 版 ⇒ 死代码,不是回退** |
| **7.72** | "M=2~4 缺口值 9.8~14.2×，值得填" | ✗ **已经填了：AVX2 融合路径覆盖 M≥1,阈值是 64 而不是 4** |
| 7.65/7.72 的 10.6× | 融合内核相对 default 的实测 | ✓ **测量是真的**，**而机制说错了** |

**⇒ 正确的机制是**：

```
不是"缺 C++ 实例" ⇒ 而是【Python 层的分派阈值把 M=17~64 分给了 default】
⇒ 而那个阈值【已经从 16 提到 64】,注释里写明了原因和实测数字（白慢最多 3.4x）
```

#### 7.74.4 ★ 而这是今天第四次"答案已经在仓库里"

| 次 | 资产 | 状态 |
|---|---|---|
| ① | 循环 Transformer | 挂载了但 `random=False` |
| ② | 4-bit 优化器 | 做好验证了但没接进训练 |
| ③ | MaskGIT 完整管线 | 实现完整但从未执行 |
| **④** | **AVX2 4-bit 融合路径 + 阈值修正** | ★ **已做、已测、已注释** |

**⇒ 而今天有两次（7.59、7.72）是【在没查全调用图的情况下宣布"缺什么"】。**

**⇒ 那和"跑过了 vs 跑对了"是同一类错误：**

> **看到了一个【形状】（只有 bf16 实例、BLOCK_M 覆盖 1~4），就以为它是一个【空缺】。**

**⇒ 而正确的第一步是【查调用图】：一个只对 bf16 特化的模板，如果它的调用者根本不在
AVX2 机器上被编译，那它既不缺也不回退 —— 它只是不在那里。**

#### 7.74.5 而库这条线的最终状态

```
✓ 4-bit GEMV 的 AVX2 融合内核         已建（avx2_gemv_4bit.h）
✓ 三种 dtype（fp32/bf16/fp16）导出     已接（ops.py L309/317/325）
✓ Python 分派 + 阈值                  已做（_GEMM_4BIT_FUSED_MAX_M = 64,从 16 提上来）
✓ 交叉点实测（M≤64 融合赢 · M≥96 default 反超）  已记录
✓ RMSNorm 两遍                        已做
✓ 4-bit 优化器                        已做已验证
⇒ ★【库侧我列的四条,查完调用图之后【一条都不缺】】
```

**⇒ 所以"9.8~14.2× 太香了"这件事的答案是：【那 9.8~14.2× 在这个仓库里已经拿到了】——
它是 `_GEMM_4BIT_FUSED_MAX_M` 那次从 16 改到 64 的收益（注释记的是"白慢最多 3.4x"，
而 7.72 独立测出同量级的差距）。**

'''

m = re.search(r'^### 7\.74 ', s, re.M)
if m:
    nxt = re.search(r'^### |^# ', s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + '\n' + s[end:]
    how = '替换 7.74'
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
