"""Report 7.120: the integrated full-layer step -- 89 ms/layer, 0.89 s per 10-layer step, 0.67x.

Chosen direction: finish the fully-on-iGPU training (option 1 of the user's binary choice). Built
gt1030/full_gpu_layer.py: one AR layer (attention -> residual -> MLP -> residual) as a single
device-resident pipeline, one queue, one finish at the end of the whole step (fwd+bwd+AdamW).

Correctness: layer output vs torch fp32 on the same fp16 inputs -> 6.8e-3 abs / 7.7e-3 rel. Per-part
gradients were validated earlier (attention dWo 1.3e-2, dWqkv 1.6e-2, dx 5.9e-3; MLP dW2 1.5e-2).

Timing (median): 89.2 ms/layer; separately fwd 35.1 / bwd 49.8 / AdamW 4.6 ms. 10 layers = 0.89 s
-> 0.67x of the 0.594 s floor; against a realistic 1030 step (floor + 10-20% overhead = 0.65-0.71 s)
it is 0.73-0.80x.

Trajectory: CPU training 4.5-5.5 s/step -> hybrid executor 7.05 s -> decomposed estimate 0.93 s ->
integrated measurement 0.89 s.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

NEW_SECTION = '''### 7.120 ★★★ 整层集成实测：89 ms/层 ⇒ 0.89 s/步 ⇒ 0.67×（训练速度现状）

目标三选一里选了**"把通核显训练真正打通"**。此前 MLP+norm 与 attention 是**分开测再相加**
（0.53 + 0.38 s），这一节把它们合成**一条整层流水线**：attention → 残差 → MLP → 残差，
前向+反向+AdamW 全在核显、一个队列、**整步只同步一次**。

实现（`gt1030/full_gpu_layer.py`）：

- 权重只存 **(in,out) fp16 一份**；反向用 CLBlast 的 transA/transB 调用，不再物化转置副本；
- AdamW 保留 fp32 master+moments，回写唯一 fp16 副本；
- attention 的布局/softmax kernel 与 MLP 的 RMSNorm/SiLU kernel **复用前两节的实现**（import，
  不复制代码）；
- 形状：M=2048 token/步（8 batch × 256），MD=512，HD=64，NH/NKV=8/4，FF=1368。

**正确性**：整层输出 vs torch fp32（同 fp16 输入）**最大误差 6.8e-3 / 相对 7.7e-3**；各段梯度此前
已分别验证（attention dWo 1.3e-2、dWqkv 1.6e-2、dx 5.9e-3；MLP dW2 1.5e-2）。

**计时**（中位数，整步一次 finish）：

| 量 | 实测 |
|---|---|
| 整层步（fwd+bwd+AdamW，集成） | **89.2 ms** |
| 分测：fwd / bwd / AdamW | 35.1 / 49.8 / 4.6 ms |
| **10 层（AR-30.5M 一步）** | **0.89 s** |
| **1030 分数（对 0.594 s 地板）** | **0.67×** |
| 对"真实 1030 一步"（地板+10~20% 开销 ≈ 0.65~0.71 s） | **0.73~0.80×** |

**演进轨迹（同一负载、同一把尺）**：

```
CPU torch 真实训练      4.5~5.5 s/步       0.11~0.13×
混合执行器（实测）       7.05 s/步          0.08×
分项估算（MLP+attn）     0.93 s/步          0.64×
整层集成（本节，实测）   0.89 s/步          0.67×
目标                    0.594 s/步         1.00×
```

**⇒ ★ 结论：整层集成后实测 0.89 s/步，是当前 CPU 训练速度的 ~5.1×，距等效线差 1.49×。**
剩余差距已逐项归因：每层约 **44 次 kernel launch**（~10-20 ms 启动税）、GEMM 实配 623 GFLOPS
（若到 750 可再省 ~0.15 s/步）、attention 反向 49.8 ms 偏重（含 4 个 batched GEMM + 布局 kernel）。
下一步按性价比排序：① 合并 split_q/split_kv、合并 AdamW、减少布局 kernel；② 反向 GEMM 形状/批量
优化；③ 需要时再上 flash attention。

> 口径：这是**单层**的集成实测；"10 层"按层线性外推（层与层之间只差一个残差/输入切换，假设成立
> 需最终整模验证）。整模的 loss/数据加载/优化器调度尚未接入。



'''

EDITS = [
    ('# 附录 A：AR 训练路径的实测证据', NEW_SECTION + '# 附录 A：AR 训练路径的实测证据', 'insert 7.120'),
    ('**本文是 119 节的【时序记录】', '**本文是 120 节的【时序记录】', 'header: 120'),
    ('**本报告的 119 节里有 12 节', '**本报告的 120 节里有 12 节', 'F header: 120'),
    ('| **GT 1030 等效目标（北极星）** | ★ **定义+判决矩阵+计分板；AR 步 1030 分数 0.21× → 目标 ≥0.84** | §7.118 · `gt1030/` |',
     '| **GT 1030 等效目标（北极星）** | ★ **定义+判决矩阵+计分板；AR 步 1030 分数 0.21× → 目标 ≥0.84** | §7.118 · `gt1030/` |\n'
     '| **整层通核显（集成实测）** | ★ **89.2 ms/层 ⇒ 0.89 s/步 ⇒ 0.67× 地板（0.73–0.80× 真实 1030）；CPU 现状的 ~5.1×** | §7.120 · `gt1030/full_gpu_layer.py` |',
     'H table: 7.120 row'),
]


def main() -> int:
    text = io.open(REPORT, encoding='utf-8').read()
    out = text
    for old, new, label in EDITS:
        if out.count(old) != 1:
            print('ABORT %-20s anchor=%d' % (label, out.count(old)))
            return 1
        out = out.replace(old, new, 1)
        print('ok    %s' % label)
    io.open(REPORT, 'w', encoding='utf-8', newline='').write(out)
    print('report chars %d -> %d, lines %d -> %d'
          % (len(text), len(out), text.count('\n') + 1, out.count('\n') + 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
