"""Report addendum 2: attention backward is now correct and validated against torch autograd.

The 7.118.14 backward was a timing skeleton with wrong transposed GEMMs. Probed the row-major
transpose conventions (gt1030/probe_transpose_gemm.py: transA=YES and transB=YES both work, with
lda/ldb = the storage's row length), then rewrote the backward chain:

  dWo = O_tok^T dOut                 (transA)
  dO_tok = dOut Wo^T                 (transB)
  split_dout -> dO per (batch,head)
  dP = dO V^T                        (batched transB)
  softmax backward -> dS
  dV = P^T dO                        (batched transA), dQ = dS K (batched), dK = dS^T Q (batched transA)
  merge_dqkv -> dqkv_tok
  dWqkv = x^T dqkv                   (transA), dx = dqkv Wqkv^T (transB)

Validated end-to-end against torch autograd on the same fp16 inputs:
  dWo   max abs err 2.3e-4 (relative 1.3e-2)
  dWqkv max abs err 3.7e-4 (relative 1.6e-2)
  dx    max abs err 5.6e-5 (relative 5.9e-3)

Timing now: attention fwd 20.4 ms | bwd 17.6 ms = 38.0 ms; full AR layer estimate 92.7 ms;
10 layers = 0.93 s -> 0.64x of the 0.594 s floor. (Stage sum for fwd is 13.3 ms, so ~7 ms of the
measured forward is per-launch/pipeline overhead across the 8 launches -- next lever.)
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.15 诚实边界\n'
NEW = '''**（同日二）attention 反向从"计时骨架"变成"已验证"。** 先用 `probe_transpose_gemm.py` 确认
row-major 下 `transA=YES` / `transB=YES` 都可用（lda/ldb = 存储的行长），再重写整条反向链：

```
dWo = O_tokᵀ dOut (transA) · dO_tok = dOut Woᵀ (transB) · split_dout → dO
dP = dO Vᵀ (batched transB) · softmax 反向 → dS · dV = PᵀdO (batched transA)
dQ = dS K (batched) · dK = dSᵀQ (batched transA) · merge_dqkv → dqkv_tok
dWqkv = xᵀ dqkv (transA) · dx = dqkv Wqkvᵀ (transB)
```

**对 torch autograd（同 fp16 输入）端到端验证**：

| 梯度 | 最大绝对误差 | 相对 |
|---|---|---|
| dWo | 2.3e-4 | **1.3e-2** |
| dWqkv | 3.7e-4 | **1.6e-2** |
| dx | 5.6e-5 | **5.9e-3** |

**计时**：attention 前向 20.4 ms · 反向 17.6 ms = **38.0 ms**；整层估算
54.7（MLP+norm）+ 38.0 = **92.7 ms**；10 层 = **0.93 s ⇒ 0.64×**。

**⇒ 注意**：前向逐段之和只有 13.3 ms，而整链实测 20.4 ms ⇒ **~7 ms 是 8 次 launch/pipeline
开销**——这就是下一个杠杆（减少 kernel 数 / 合并启动），而不是算法。

#### 7.118.15 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'append attention-backward validation to 7.118.14')]


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
