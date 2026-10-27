"""Report 7.118.14: attention phase measured -- forward validated, and it is now the new bottleneck.

What was built (gt1030/attention_gpu.py): the full attention phase of one AR layer on the iGPU, with
the key enabler discovered by probing CLBlast (gt1030/probe_batched_gemm.py): CLBlastHgemmStridedBatched
exists in this build and works with the DOCUMENTED enum values (layout=101 row-major, transposes
111/112). That turns 64 (batch x head) matrices into a handful of launches:
  qkv layout kernels (split_q/split_kv), S = QK^T (batched 64x[256,256,64]), causal softmax kernel,
  O = PV (batched), merge to token layout, out projection; backward: dWo, dO, split dO,
  dP = dO V^T (batched), softmax backward, dV = P^T dO, dQ = dS K, dK = dS^T Q (batched), merge dqkv.

Correctness: forward O vs a torch fp32 reference on the same fp16 inputs -> max abs err 8.2e-4
(relative 5.0e-3). The backward sequence in this script is a TIMING skeleton: its layout/GEMM
arguments are not yet the validated ones, so only the forward number is a correctness claim.

Timing per layer: attention fwd 31.1 ms | bwd 28.8 ms (timing skeleton) = 59.9 ms.
Combined with the MLP+norm block (54.7 ms): 114.6 ms/layer -> 10 layers = 1.15 s -> 0.52x of the
0.594 s floor.

So attention is now the bottleneck (its GEMM content is only ~12.7 GFLOP vs the MLP block's 25.8, but
it pays for 64-way batched small GEMMs plus layout kernels and two 8 MB S/P materialisations). Next:
a flash-style fused kernel (softmax inside the score kernel, no S/P materialisation, layout fused into
the qkv projection) rather than more batching.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.14 诚实边界\n'
NEW = '''#### 7.118.14 attention 相位实测：前向正确，但它成了新的瓶颈（0.52×）

`gt1030/attention_gpu.py`：把一层的 **attention 相位**整体搬上核显。关键发现来自
`gt1030/probe_batched_gemm.py`：**这个 CLBlast 构建里有 `CLBlastHgemmStridedBatched`，而且必须用
文档里的枚举值**（layout=**101** row-major、transpose **111/112**——此前我们所有 GEMM 传的
layout=0 是非法值，只是碰巧被当成默认布局）⇒ 64 个 (batch × head) 矩阵可以用**少数几次 launch**
搞定：

- 布局 kernel：`split_q` / `split_kv`（token 布局 → 每头连续）；
- 前向：**S = QKᵀ**（batched 64×[256,256,64]）→ 因果 softmax kernel → **O = PV**（batched）→
  `merge_o` → out 投影 GEMM；
- 反向：dWo、dO、`split_dout`、**dP = dO Vᵀ**（batched）、softmax 反向 kernel、**dV = PᵀdO**、
  **dQ = dS K**、**dK = dSᵀQ**（batched）、`merge_dqkv`。

**正确性**：前向 O 对 torch fp32 参考（同 fp16 输入）**最大误差 8.2e-4（相对 5.0e-3）**。
⚠️ 反向序列在本脚本里是**计时骨架**：布局/GEMM 参数尚未全部校核，**不作为正确性声明**。

**计时（每层）**：

| 相位 | 时间 |
|---|---|
| attention 前向 | **31.1 ms** |
| attention 反向（计时骨架） | **28.8 ms** |
| **attention 合计** | **59.9 ms** |
| MLP+norm 块（§7.118.12/13） | 54.7 ms |
| **整层** | **114.6 ms** |
| **10 层** | **1.15 s ⇒ 0.52×**（对 0.594 s 地板） |

**⇒ ★ attention 取代 MLP 成为新瓶颈**：它的 GEMM 只有 ~12.7 GFLOP（MLP 块 25.8），却要付
**64 路小 GEMM 的批处理低效** + 布局 kernel + **两次 8 MB 的 S/P 物化**。继续"批处理调参"收益有限，
下一步是 **flash 风格融合 kernel**：softmax 进 score kernel、不物化 S/P、布局融进 qkv 投影。

#### 7.118.15 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.14 attention; renumber honesty to 7.118.15')]


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
