"""Append 7.105: an AVX2 int8 GEMM, verified, and where it wins.

The INT8 item has been listed as untested since 7.78: torch._int_mm falls back to scalar on AVX2,
measured at 0.5 to 3.2 GOPS against fp32's 180. This writes the kernel and finds that the answer to
"is it faster" is the same shape as the answer to "is the iGPU faster" -- it depends on M.
"""
from __future__ import annotations

import io
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

P = r"D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md"
s = io.open(P, encoding="utf-8").read()

SEC = '''### 7.105 ★★★ AVX2 INT8 GEMM：写完、逐位验证、而"快不快"取决于 M

#### 7.105.1 指令与两条硬约束

```
核心: _mm256_maddubs_epi16 + _mm256_madd_epi16
  ⇒ 每指令做【32 个 int8 乘加】—— 那才是 int8 值得做的原因（fp32 是 8 个）

约束一: maddubs 的第一个操作数必须【无符号】,第二个有符号
约束二: ★【它的两两和会【饱和到 int16】】
```

**⇒ 而约束二排除了常见的"加 128 偏移 + 修正项"做法：**

```
偏移之后 a 到 255 ⇒ 255 × 127 × 2 = 64770 > 32767 ⇒ ★【饱和,结果错】
而让 a 落在 [0,127] ⇒ 2 × 127 × 127 = 32258 < 32767 ⇒ ★【不会饱和】
⇒ ★【所以"A ≤ 127"不是风格,是【正确性条件】
   —— 而它恰好是【标准非对称 int8 量化】的值域】
```

**★ 而布局正好对：`C[m,n] = Σ_k A[m,k]·W[n,k]` = `A @ Wᵀ`，
`W[n][k]` 行主序 ⇒ 而那是 bnb / LLM 的权重布局 ⇒ 无需转置。**

#### 7.105.2 正确性：6 个形状【逐位相等】

| 形状 | 覆盖什么 | 元素 | 不符 |
|---|---|---|---|
| 1×1×32 | 最小（正好一条指令） | 1 | **0** |
| 1×1×31 | K 不足 32（纯尾巴） | 1 | **0** |
| 3×5×71 | 混合尾巴（71%32=7） | 15 | **0** |
| 8×16×512 | LLM 一小层 | 128 | **0** |
| 1×4096×4096 | 单 token 解码 | 4096 | **0** |
| 128×1024×512 | prefill 一档 | 131072 | **0** |

**⇒ ★【全部逐位相等 —— 而整数算术没有舍入,所以"不等"就是 bug,没有"接近"这一档。】**

#### 7.105.3 ★★ 而"快不快"：朴素对朴素是 21~72×

| 形状 | int8 | fp32 朴素循环 | 比值 |
|---|---|---|---|
| 1×4096×4096 | 1.05 ms | 22.20 ms | **21.09×** |
| 128×1024×512 | 1.15 ms | 83.00 ms | **72.02×** |

**⇒ ★【而这是【错的基线】:那个 fp32 朴素循环只跑到 1.6 GFLOPS,而 torch 跑到 200 以上。】

#### 7.105.4 ★★★ 而对调优 BLAS 比，答案取决于 M

| 形状 | int8 内核 | torch fp32 | int8 用时 / torch 用时 | 判定 |
|---|---|---|---|---|
| **1×4096×4096** | 32.0 GOPS | 10.5 GFLOPS | ★ **0.33×** | ★ **int8 快 3.05×** |
| **128×1024×512** | 116.7 GOPS | **275.7 GFLOPS** | 2.36× | ★ **int8 慢 2.36×** |

**⇒ ★★★ 而这不是"int8 没用",而是与 §7.97 完全同构的一件事：**

```
M=1  ⇒ 【访存受限】⇒ int8 赢 3.05×
        （因为它只有 1/4 的数据量,而访存受限时那正是唯一重要的东西）
M=128 ⇒ 【计算受限】⇒ torch 赢 2.36×
        （此时决定胜负的是【每周期做多少 FLOP】,而我的内核只用 8 个 int32 累加器、
          没做寄存器分块、没做 L2/L3 分块）
⇒ ★【所以"int8 快不快"和"核显快不快"是同一个问题的两种形式:
   答案取决于 M,因为 M 决定它是访存受限还是计算受限。】
```

#### 7.105.5 所以这一项的状态

| 项 | 状态 |
|---|---|
| int8×int8→int32 内核 | ★ **✓ 写完,6 形状逐位验证** |
| 饱和约束 | ★ **✓ 已定位并写进判据（A ≤ 127 是正确性条件）** |
| **M=1 解码** | ★ **✓ 赢 3.05×** |
| **M=128 prefill** | ✗ **输 2.36×（缺分块）** |
| 接入 torch（替 `_int_mm`） | ✗ **未做** |
| 反传 / "int8 训练" | ✗ **未做，而那是另一件事** |

**⇒ ★【所以准确的表述是:"int8 GEMM 内核【有了、对了、而在解码形状上已经赢 3 倍】；
而它在 prefill 形状上还输,原因是缺分块,而那是一个已知的、可做的优化。"】**

**⇒ ★★ 而"int8 GEMM【训练】内核"这个说法要拆开：**
```
① int8 前向 GEMM        ⇒ ✓ 本次
② int8 反传             ⇒ 未做（而工业界也很少做 —— LLM.int8() 是【推理】）
③ "全 int8 训练"        ⇒ 那是一个【混合精度系统】,不是内核
⇒ ★【我们做的是 ①。而 ②③ 从来没有被承诺过,也不该被记成"已完成"。】
```

'''

m = re.search(r"^### 7\.105 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.105"
else:
    a = re.search(r"^# 附录", s, re.M)
    if a:
        s = s[:a.start()] + SEC + "\n\n" + s[a.start():]
        how = "插在附录之前"
    else:
        s = s.rstrip() + "\n\n" + SEC
        how = "追加"
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("  %s；报告 %d 行" % (how, s.count("\n")))
