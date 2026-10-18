"""Report 7.118.12: the full-GPU step skeleton works -- 0.526 s for 10 MLP+norm blocks, ~0.74 s projected.

What was built (gt1030/full_gpu_step.py): the MLP+norm block that carries ~71% of the AR layer's FLOPs,
entirely device-resident, with OpenCL kernels for RMSNorm (fwd/bwd), SiLU gate (fwd/bwd), residual add,
device transpose, AdamW; GEMMs via CLBlast Hgemm with the verified device-buffer conventions. No host
round trips inside the step.

Correctness (vs torch fp32 reference on the same fp16 inputs):
  forward  max abs err 5.2e-3 (relative 8.2e-3)
  dW2      max abs err 3.6e-3 (relative 1.5e-2)

Timing (median, 2048 tokens/step, D=512, FF=1368):
  block step      55.6 ms  (25.8 GFLOP GEMM -> 464 GFLOPS effective)
  phases          fwd 14.7 ms | bwd 32.0 ms | AdamW 5.8 ms
  10 blocks       0.526 s  (MLP+norm only)
  full-step est.  ~0.74 s  (attention/qkv adds ~29% of FLOPs at similar efficiency) => ~0.80x of the
                  1030 reference

vs the hybrid executor (7.05 s, floor 1.1-1.4 s), keeping everything on the device is already a 10x
improvement, and the gap to 0.594 s is now ~25% rather than an order of magnitude. The known next wins:
  * dW without the two transposes (find the accepted ldb convention or keep transposed copies as the
    forward kernels produce them): bwd is 2.2x fwd, partly transfers;
  * vectorise AdamW (2.1M elements in 5.8 ms is bandwidth-limited);
  * fuse the two-pass RMSNorm kernels (one workgroup reduction per row);
  * add attention kernels (QK, softmax, AV) and qkv/out GEMMs.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.12 诚实边界\n'
NEW = '''#### 7.118.12 full-GPU step 骨架：**10 层 MLP+norm = 0.526 s**，整步估算 ~0.74 s（≈0.80×）

`gt1030/full_gpu_step.py`：把 AR 单层里占 ~71% FLOPs 的 **MLP+norm 块**做成**全设备常驻**的一步——
RMSNorm（前/反向）、SiLU gate（前/反向）、残差加、设备内转置、AdamW 全部是 OpenCL kernel，
GEMM 全走 CLBlast Hgemm（设备缓冲，已验调用约定）；**步内零主机往返**。

**正确性**（同 fp16 输入，对 torch fp32 参考）：

| 检查 | 最大绝对误差 | 相对 |
|---|---|---|
| 前向输出 | 5.2e-3 | 8.2e-3 |
| dW2 | 3.6e-3 | 1.5e-2 |

**计时**（2048 token/步，D=512，FF=1368，中位数）：

| 量 | 实测 |
|---|---|
| 单块步（fwd+bwd+AdamW） | **55.6 ms**（25.8 GFLOP GEMM ⇒ 464 GFLOPS 有效） |
| 相位 | fwd **14.7 ms** · bwd **32.0 ms** · AdamW **5.8 ms** |
| 10 块（仅 MLP+norm） | **0.526 s** |
| 整步估算（加 attention/qkv ≈ +29% FLOPs） | **~0.74 s ⇒ 1030 分数 ~0.80×** |

**⇒ ★ 与混合执行器（7.05 s，地板 1.1~1.4 s）相比，全设备路线一步就把估算拉到 0.74 s——
离 0.594 s 只差 ~25%，不再是数量级。** 已量出的下一步收益点：

1. **dW 去掉两次转置**（找到 CLBlast 接受的 ldb 约定，或让前向 kernel 直接产出转置副本）；
   当前 bwd = 2.2× fwd，其中一部分是 2 次转置（h 2 MB + g 5.6 MB）；
2. **AdamW 向量化**（2.1M 元素 5.8 ms 是带宽受限，float4 处理可望减半）；
3. **RMSNorm 两遍合一**（每个 workgroup 一行做归约，省一趟 2 MB 读写）；
4. 补 attention kernel（QK/softmax/AV）与 qkv/out GEMM —— 完成整步。

> 口径：0.74 s 是"MLP+norm 实测 × 1/0.71"的外推，attention 的实际效率尚未测；它是当前最好的
> 整步估计，不是已实测的整步。

#### 7.118.13 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.12 full-GPU skeleton; renumber honesty to 7.118.13')]


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
