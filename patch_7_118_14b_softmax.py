"""Report addendum: the attention softmax was the single biggest cost and is now 10x faster.

Attribution of the 30.3 ms attention forward (stage timings):
  qkv gemm 3.64 | split_q 0.77 | split_kv 1.21 | S batched 2.10 | softmax S->P 18.02 |
  O batched 1.69 | merge_o 0.97 | out gemm 1.82  (sum 30.22 = measured, pipeline had nothing hidden)

The softmax kernel processed an entire 256-key row in ONE work-item (3 serial passes over global
memory), i.e. 16384 work-items each latency-bound. Rewritten as one workgroup (256 lanes) per row with
local-memory max/sum reductions and parallel exp/write: 18.02 -> 1.71 ms. Two bugs were caught on the
way (a missing barrier between the max-read and the sum-reuse of the same local array, and a launch
grid of (T,BH) with local (256,1) which silently ran only ONE workgroup per head instead of one per
row -- the latter was found because row 0 was correct and every other row came out zero).

Effect:
  attention fwd 30.3 -> 16.0 ms, bwd 27.9 -> 18.3 ms (bwd is still a timing skeleton), total 58.3 ->
  34.3 ms; full AR layer = 54.7 (MLP+norm) + 34.3 = 88.9 ms; 10 layers = 0.89 s -> 0.67x of the
  0.594 s floor (was 0.52x).
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
NEW = '''**（同日）softmax 成了单一最大成本，已 10× 修掉。** 对 30.3 ms 的 attention 前向做逐段归因：

| 阶段 | ms |
|---|---|
| qkv gemm | 3.64 |
| split_q / split_kv | 0.77 / 1.21 |
| S batched | 2.10 |
| **softmax S→P** | **18.02**（60%） |
| O batched | 1.69 |
| merge_o / out gemm | 0.97 / 1.82 |

罪魁：softmax kernel **一个 work-item 串行处理整行 256 个 key**（3 趟全局内存往返），16384 个
work-item 全部 latency-bound。改成**一行一个 workgroup（256 lane）+ 本地内存 max/sum 归约**后：
**18.02 → 1.71 ms**。过程中抓了两个 bug：① 复用同一块 local 数组做 max→sum 时缺一个 barrier；
② 启动网格写成 `(T,BH) × (256,1)`，实际每头只跑了 **1 个** workgroup（只有 row 0 正确、其余全零，
才暴露出来）。

**效果**：attention 前向 30.3→**16.0 ms**、反向 27.9→**18.3 ms**（反向仍是计时骨架），合计
58.3→**34.3 ms**；整层 = 54.7（MLP+norm）+ 34.3 = **88.9 ms**；10 层 = **0.89 s ⇒ 0.67×**
（此前 0.52×）。

#### 7.118.15 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'append softmax optimisation to 7.118.14')]


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
