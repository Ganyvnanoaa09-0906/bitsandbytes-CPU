"""Append 7.106: register blocking does not help, and the reason is that the diagnosis was wrong.

7.105 said the prefill loss was 'missing blocking'. Blocking was implemented at 2x4, verified bit-exact,
and produced 0.91x -- slower. The diagnostic was wrong, and what it was wrong about is worth recording.
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

SEC = '''### 7.106 ★★ 分块没用，而"为什么没用"说明诊断错了

#### 7.106.1 做了什么

**§7.105 把 prefill 的 2.36× 差距诊断为"缺寄存器分块 + 缓存分块"。
于是实现了 2×4 的寄存器分块（2 个 A 向量 + 8 个累加器，10 个 YMM 在飞，AVX2 有 16 个）。**

#### 7.106.2 正确性：7 个形状全部逐位相等

| 形状 | 覆盖什么 | 对参照 | 对朴素 |
|---|---|---|---|
| 1×1×32 | 最小 | 0 | 0 ✓ |
| 1×1×31 | 纯尾巴 | 0 | 0 ✓ |
| 3×5×71 | 混合尾巴 | 0 | 0 ✓ |
| 2×4×32 | 正好一个寄存器块 | 0 | 0 ✓ |
| **7×13×100** | **块不足（7%2=1, 13%4=1）** | 0 | 0 ✓ |
| 128×1024×512 | prefill | 0 | 0 ✓ |
| 1×4096×4096 | 解码 | 0 | 0 ✓ |

**⇒ ★【与 int64 参照、与朴素版【都】逐位相等 ⇒ 优化没改语义。】**

#### 7.106.3 ★★★ 而吞吐没有改善

| 形状 | 朴素 | 分块 | 加速 |
|---|---|---|---|
| 128×1024×512 | 1.12 ms | 1.24 ms | ★ **0.91×（更慢）** |
| 1×4096×4096 | 0.99 ms | 0.95 ms | 1.05× |

**⇒ ★【所以"缺分块"这个诊断【是错的】。】**

#### 7.106.4 ★★★ 而真正的诊断：它是 ALU 受限，不是访存受限

```
朴素版: 128×1024×512 ⇒ 0.134 GOP / 1.12 ms = 【120 GOPS】
        1×4096×4096  ⇒ 0.034 GOP / 0.99 ms = 【 34 GOPS】← 3.5 倍慢

而理论上限: maddubs 每指令 32 MAC × 2.6 GHz × 6 核 ≈ 【499 GOPS】
⇒ 120 / 499 = 【24% 峰值】

★ 而两个形状的差别说明:
   M=1   ⇒ 权重 16 MB 不进 L3 ⇒ 【访存受限】
   M=128 ⇒ 工作集 1 MB 在 L2  ⇒ 【计算受限】

⇒ ★【分块只帮访存受限的内核。它是 ALU 受限 ⇒ 所以分块没用。
   ⇒ 而这与 7.105 那个诊断【正好相反】。】
```

**⇒ 而 ALU 受限的原因可以数:**

```
每 32 个 MAC 要做:
  1 × maddubs        ← 真正的乘法（32 个 MAC）
  1 × madd_epi16     ← ★ 只把 int16 扩成 int32,【不做乘法】
  1 × add_epi32      ← 累加
⇒ ★【3 条 ALU 指令换 32 个 MAC ⇒ 每条指令 10.7 个 MAC】

★ 而为什么省不掉 madd_epi16:
   maddubs 给 int16,而累加必须 int32
   而【不能在 int16 里多累加几次】: 两个 maddubs 结果相加就到 64516 > 32767 ⇒ 饱和
   ⇒ ★【所以"省不掉"是【饱和约束】的直接后果
      —— 而那条饱和约束来自 A ≤ 127 的取值域（§7.105）】
```

#### 7.106.5 所以三条串起来是一条因果链

```
A ≤ 127（正确性条件）
   ⇒ maddubs 的 int16 结果已用到 98.5% 的余量
   ⇒ 不能在 int16 里合并多个 maddubs
   ⇒ 每次都要 madd_epi16 展宽 + add 累加
   ⇒ 每 32 MAC 要 3 条 ALU 指令
   ⇒ 24% 的指令效率
   ⇒ ★【而"分块"帮的是访存,不是这个 —— 所以它没用】
```

**⇒ ★【而这条链的每一环都是有实测或数学依据的,不是推测。】

#### 7.106.6 而真正的下一步是什么

**⇒ 不是分块，是【减少每 MAC 的指令数】。而那条路有三条，各有代价：**

| 方案 | 做法 | 代价 |
|---|---|---|
| **预展宽 + vpmaddwd** | int8 → int16 后再算 | 展宽本身要指令 ⇒ 未必净赚，需实测 |
| **AVX-512 VNNI（VPDPBUSD）** | 一条指令 = 4×int8 点积累加进 int32 | ★ **这颗 CPU 没有**（Zen4 才有） |
| **改用量化方案** | 让 A ∈ [0,127] 但 K 更短 | 改变精度契约，不是纯优化 |

**⇒ ★【所以在这颗 CPU 上，int8 的 24% 指令效率【接近它的结构上限】——
不是写坏了，是 AVX2 的 int8 路径本来就要为"没有 VNNI"付这个代价。】

#### 7.106.7 而这一节的教训

```
§7.105: 看到"朴素内核输给调优 BLAS" ⇒ 诊断"缺分块"
§7.106: 实现分块 ⇒ 没有改善 ⇒ ★【诊断错了】

★ 而错的根源是【用"对手有什么"代替"我缺什么"】:
   对手有分块 ⇒ 我以为我缺分块
   ★ 而实际上我缺的是【少一条指令】,而那与分块无关
⇒ ★【所以"向对手看齐"是一个不可靠的诊断方法 —— 它只能告诉你【差多少】,不能告诉你【差在哪】。】
```

'''

m = re.search(r"^### 7\.106 ", s, re.M)
if m:
    nxt = re.search(r"^### |^# ", s[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(s) - m.end())
    s = s[:m.start()] + SEC + "\n" + s[end:]
    how = "替换 7.106"
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
