"""Report 7.118.9: P1a first real number -- the AR step's GEMM phase on the iGPU.

Measured with gt1030/ar_step_gemm_schedule.py (173 GEMMs = the real forward+backward schedule of the
AR-30.5M step, fp16 CLBlast on the iGPU, per-shape medians):
  GEMM calls         173
  FLOPs              364.1 GFLOP  (vs the 375 GFLOP analytic figure of 7.12)
  GEMM phase time    0.515 s
  effective rate     707 GFLOPS
  GT 1030 reference  0.594 s  -> GEMM-only fraction 1.15x

Whole-step projection: GEMM 0.515 s + CPU non-GEMM phase 0.09-0.26 s (3-9% of the measured 2.854 s
CPU step, report 7.12) + phase-switch overhead => 0.61-0.78 s, i.e. 0.76-0.98x of the 1030 reference.
So the compute phase already beats the 1030 floor; the remaining gap is the CPU-side work and switch
overhead, which is exactly what P2 (fusion) and the executor have to remove.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.9 诚实边界\n'
NEW = '''#### 7.118.9 P1a 实测：AR 整步的 GEMM 相位 0.515 s = 1030 参考的 1.15×

`gt1030/ar_step_gemm_schedule.py` 把 AR-30.5M 一步训练的**真实前向+反向 GEMM 调度**（10 层 +
head，每层 6 个前向、11 个反向，共 **173 次 GEMM**）按真实形状在核显上跑了一遍（fp16 CLBlast，
每个形状取中位数并计入调度表）：

| 量 | 实测 |
|---|---|
| GEMM 调用数 | 173 次/步 |
| GEMM 计算量 | **364.1 GFLOP**（§7.12 解析值 375 GFLOP，差 3%） |
| **GEMM 相位总时间** | **0.515 s** |
| 有效吞吐 | **707 GFLOPS** |
| GT 1030 参考 | 0.594 s |
| **分数（仅 GEMM 相位）** | **1.15×** |

**整步投影**：0.515 s（核显 GEMM 相位）+ **0.09~0.26 s**（CPU 非 GEMM 相位：softmax/norm/GELU/
优化器，按 §7.12 的 3~9% 折算）+ 相位切换开销 ⇒ **0.61~0.78 s**，即 **0.76~0.98×**。

**⇒ ★ 结论：算力相位已经越过 1030 的地板（1.15×），整步差的那 0.1~0.2 s 全部在 CPU 非 GEMM
相位与切换开销上。** 这正是 P2（GEMM epilogue 融合 + 非 GEMM 算子核显化 + 减少切换）要吃的部分；
只要把 CPU 相位压到 0.08 s 以内、GEMM 效率提到 750 GFLOPS，整步就是 0.57 s ⇒ 分数 1.04×。

> 口径说明：这里比的是 1030 的 roofline 地板（纯计算 0.594 s），真实 1030 的每步还会额外付出
> 启动/非 GEMM 开销 —— 所以我们这个"整步含 CPU 相位"的投影对 1030 是偏严格的比较。

#### 7.118.10 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.9 P1a GEMM measurement; renumber honesty to 7.118.10')]


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
