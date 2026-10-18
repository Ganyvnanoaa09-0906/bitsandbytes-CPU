"""Report 7.118.8: is bandwidth the current bottleneck? Measured answer: not alone, but under concurrency yes.

Measurements (gt1030/bench_bandwidth_contention.py, idle machine):
  CPU memory read bandwidth: 1 thread 15.52 GB/s, 6 threads 22.25 GB/s (copy r+w 12.93) -- the
  MLP-limited ceiling from 7.3 still holds, far below the 51.2 GB/s theoretical DDR4-3200.
  iGPU Hgemm 512x12288x4096: 888.5 GFLOPS, implied streamed bandwidth 2.02 GB/s -- an order of
  magnitude below the 22 GB/s ceiling, so a lone GEMM is compute/occupancy bound, not bandwidth bound.
  Same GEMM while a CPU thread streams memory: 614.1 GFLOPS, -44.7% -- this is the real bandwidth
  problem, and it is a concurrency problem (shared L3/memory controller), not a single-engine one.
  Arithmetic intensity: AR step 624 FLOP/byte, 8B NF4 step 3268 FLOP/byte, machine ridge ~58 => both
  workloads are compute-bound at the roofline level.

Design consequence: phase separation (handover), not concurrency: the iGPU owns GEMM phases, the CPU
owns GDN/attention/norm phases, and they do not overlap. This matches 7.97's "handover rather than
division" finding, now quantified at the bandwidth level.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.8 诚实边界\n'
NEW = '''#### 7.118.8 带宽到底是不是主矛盾？——单跑不是，并发才是

这个问题值得实测，因为两种答案导向两种完全不同的架构：若带宽是瓶颈，就要做数据复用/量化；
若计算是瓶颈，就要做内核效率与相位编排。

**（a）机器带宽（空闲，实测）：** CPU 读 1 线程 **15.52 GB/s**、6 线程 **22.25 GB/s**（copy r+w
12.93 GB/s）——仍受 outstanding-miss（MLP）限制（§7.3），远低于 DDR4-3200 理论 51.2 GB/s。

**（b）单跑核显 GEMM 的隐含带宽：** Qwen3-8B mlp-up 形状 512×12288×4096，**888.5 GFLOPS**，
按"每调用读一次 A/B/C"折合隐含带宽 **2.02 GB/s**——离 22 GB/s 天花板差一个数量级。
**⇒ 单跑时 GEMM 是计算/占用受限（888 / 1150 峰值 = 77%），不是带宽受限。**

**（c）但 CPU 一跑带宽活，核显就掉：** 同一个 GEMM，后台同时跑 CPU 内存拷贝流：
**888.5 → 614.1 GFLOPS（−44.7%）**。这正是 §7.97"共享 L3 是零和"的带宽维度量化。

**（d）workload 层面的算术强度：**

| workload | FLOP/byte | 对比机器拐点 ~58 |
|---|---|---|
| AR-30.5M 步 | **624** | 高 10 倍 ⇒ 计算受限 |
| Qwen3-8B NF4 步（仅权重流） | **3268** | 高 50 倍 ⇒ 计算受限 |

**⇒ ★ 三条结论：**
1. **对单个引擎，带宽不是当前主矛盾**——GEMM 隐含带宽只有 2 GB/s；
2. **真正的带宽问题是"并发"**：CPU 内存流让核显 GEMM 掉 45%，所以**架构必须相位分离（换手），
   不能追求"CPU+核显一起干"**——这给 §7.97 的结论补上了带宽侧的量化证据；
3. 带宽仍会咬人的三处：① 8B 每步流式读 5.14 GB NF4 权重（22 GB/s ⇒ **0.23 s**，相对 20 s
   计算可忽略）；② 若把 NF4 物化成 16 GB fp16 staging（**0.73 s**，仍可接受但不该做）；
   ③ CPU 侧 GDN/attention/norm 阶段自身受 15~22 GB/s 限制 —— 这些留在 CPU 是对的。

**⇒ 因此 P1 的编排原则定为：GEMM 相位核显独占；GDN/注意力/norm 相位 CPU 独占；两相位不重叠，
切换点按层块粒度。** 这既避开 45% 的并发惩罚，又让每个引擎各自跑在计算受限区。

#### 7.118.9 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.8 bandwidth; renumber honesty to 7.118.9')]


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
