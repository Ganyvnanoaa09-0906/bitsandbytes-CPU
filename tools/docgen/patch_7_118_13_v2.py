"""Report 7.118.13: full-GPU v2 -- the block is GEMM-bound; two "obvious" optimisations mostly did not pay.

Tried in gt1030/full_gpu_step_v2.py:
  * vectorised AdamW (float4 masters/moments, 4 elements per work-item);
  * coalesced 2D transpose kernel instead of the 1-element version;
  * fused one-workgroup-per-row RMSNorm (this one produced NaNs in the first attempt and was reverted
    to the two-pass version; the fused variant is NOT in the numbers below).

Measured (2048 tokens/step, D=512, FF=1368):
  v1 block          55.6 ms
  v2 block          52.8-54.7 ms   (~5%, borderline noise)
  v2 attribution    11 GEMMs 41.5 ms = 623 GFLOPS | kernels + launches 13.2 ms
  phases            fwd 24.4 | bwd 32.0 | AdamW 6.3 ms (separate, hence the sum exceeds the pipelined 54.7)

Then tested the one structural idea that could matter: concatenating W1/W3 into one (2048x2736x512)
GEMM and da/db into one (512x2736x2048) dW GEMM. Result: 1.8% faster and 1.4% slower respectively --
i.e. nothing. The GEMM time is set by CLBlast's per-shape efficiency, not by launch count.

Consequence: the MLP+norm block is now GEMM-bound at ~623 GFLOPS effective, with 13 ms of kernel/
launch overhead. Ten blocks = 0.53-0.55 s; adding attention (+29% FLOPs at the same rate + overhead)
gives a full AR step of ~0.70-0.75 s. Against the 1030 pure-compute floor (0.594 s) that is ~0.80-0.85x;
against a *realistic* 1030 step (floor + its own launch/non-GEMM overhead, commonly 10-20% on such a
small GPU) it is ~1.0x. Both numbers must be reported; the honest target is a measured full step, so
the next work is the attention kernels, not more micro-optimisation of this block.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.13 诚实边界\n'
NEW = '''#### 7.118.13 full-GPU v2：块已经 GEMM 瓶颈，两个"显然"的优化都不怎么值钱

v2（`gt1030/full_gpu_step_v2.py`）试了三件事：

1. **AdamW 向量化**（float4 master/moment，4 元素/work-item）；
2. **二维 coalesced 转置 kernel**（替代 1 元素散写版）；
3. 单 workgroup 一行的**融合 RMSNorm**——第一版出 NaN，已回退两遍版，**不在下面的数字里**。

| 量 | 实测 |
|---|---|
| v1 单块 | 55.6 ms |
| **v2 单块** | **52.8~54.7 ms**（~5%，接近噪声） |
| v2 归因 | **11 个 GEMM = 41.5 ms（623 GFLOPS）** · kernel+launch = 13.2 ms |
| 相位（分开测） | fwd 24.4 · bwd 32.0 · AdamW 6.3 ms |

再试了唯一可能结构性地省钱的招：把 W1/W3 **拼成一个宽 GEMM**（2048×2736×512）、da/db 拼成
（512×2736×2048）的 dW GEMM：

| 对比 | 两个窄 GEMM | 一个宽 GEMM | 差 |
|---|---|---|---|
| 上投影 | 8.75 ms | 8.60 ms | **−1.8%（赚）** |
| dW | 7.94 ms | 8.06 ms | **+1.4%（亏）** |

**⇒ 结论：GEMM 时间由 CLBlast 在这类形状上的效率决定（实配 623 GFLOPS），不由 launch 数决定。
"拼接/合并调用"这类优化在这里基本无效。**

**整步推算**：10 块 MLP+norm = **0.53~0.55 s**；加 attention（+29% FLOPs，同效率 + 开销）
≈ **0.70~0.75 s**。两个口径：

- 对 1030 的**纯计算地板 0.594 s**：**0.80~0.85×**；
- 对**真实 1030 一步**（地板 + 它自己的 launch/非 GEMM 开销，这类小 GPU 通常 +10~20%）：
  **约 1.0×**。

**⇒ 所以下一步不是继续微调这个块，而是把 attention kernel 补上、测一条真正的完整 AR 整步**——
两个口径都给数，才配叫"等效判决"。

#### 7.118.14 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.13 full-GPU v2; renumber honesty to 7.118.14')]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-30s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
