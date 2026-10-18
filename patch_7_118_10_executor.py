"""Report 7.118.10: the P1a executor prototype -- semantics correct, naive plumbing 10x too slow.

Measured on the full-size executor (gt1030/ar_step_executor.py): d512, 10 layers, 8/4 GQA, SwiGLU,
vocab 1024, 2048 tokens/step, 8 steps, AdamW; linears on the iGPU, everything else on the CPU.

  CPU reference step            3.683 s
  iGPU offload step (median)    6.854 s   -> naive offload is SLOWER than the CPU (0.54x)
  breakdown per step            H2D 588 ms | GEMM 2403 ms | D2H 274 ms (153 GEMM calls)
  loss parity (8 steps)         max |diff| 4.6e-2 (fp16 accumulation drift, same trajectory)

Why GEMM is 2.4 s when the schedule benchmark said 0.515 s: per-call overhead. An isolated gemm_nn at
the same shapes costs 9-11 ms = 235-258 GFLOPS, against 3-6 ms / 700-830 GFLOPS for the pre-allocated
buffers of the schedule benchmark. The difference is buffer creation, pageable COPY_HOST_PTR copies
(~1 GB/s effective) and a queue.finish per call.

So the executor is semantically right (loss matches) but architecturally naive. The compute floor
(0.515 s) is unchanged; the next iteration must remove the plumbing instead of the maths:
  1) weight buffers persisted on the device (upload once per step, ~61 MB at whatever the copy
     engine really gives);
  2) a scratch pool keyed by shape (no per-call cl.Buffer creation);
  3) pinned/mapped host memory for activations, or better: keep activations GPU-resident and move
     the elementwise CPU ops (SiLU, RMSNorm, softmax) to small OpenCL kernels so a phase becomes
     "upload once, many GEMMs, download once";
  4) one queue.finish per phase instead of per GEMM.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.10 诚实边界\n'
NEW = '''#### 7.118.10 P1a 执行器原型：语义对了，但朴素实现 6.85 s——瓶颈是每次调用的搬运

`gt1030/ar_step_executor.py` 把 GEMM 相位做成了**真实训练步**：形状与 AR 模型一致（d512、10 层、
8/4 GQA、SwiGLU、vocab 1024、2048 token/步），线性层走核显 fp16 CLBlast，其余（RMSNorm、SiLU、
softmax、loss、AdamW）留在 CPU；两种实现（纯 CPU 参考 vs 核显卸载）从**同一初始权重**、同数据
起跑，对比 loss 曲线。

| 量 | 实测 |
|---|---|
| CPU 参考整步 | 3.683 s |
| **核显卸载整步** | **6.854 s**（比 CPU 还慢，0.54×） |
| 分项/步 | H2D 588 ms · **GEMM 2403 ms** · D2H 274 ms（153 次 GEMM） |
| loss 一致性（8 步） | 最大差 **4.6e-2**（fp16 累加漂移，轨迹一致） |
| 小配置（2 层）loss 一致性 | 最大差 **1.2e-4** |

**语义是对的**（小配置 1.2e-4，全配置 8 步 4.6e-2，轨迹一致）；**但工程实现是朴素的**。GEMM 相位
2.4 s vs 调度基准 0.515 s —— 差 4.7×。单独测同一个 `gemm_nn`：**9~11 ms/次 = 235~258 GFLOPS**，
而调度基准（预分配缓冲）是 **3~6 ms / 700~830 GFLOPS**。差额全部来自每次调用的开销：
`cl.Buffer` 现场创建、pageable 的 `COPY_HOST_PTR` 拷贝（实测有效带宽只有 ~1 GB/s）、以及每次
GEMM 一次 `queue.finish`。

**⇒ ★ 这不是"核显不行"，是"搬运方式不对"：计算地板仍是 0.515 s，接下来要消灭的是管线开销：**
1. **权重缓冲常驻**（每步只传一次，~61 MB；先测清真实拷贝带宽再决定 pinned 还是分块）；
2. **按形状建 scratch 池**，不再每次 `cl.Buffer`；
3. **激活用 pinned/mapped host 内存**，或更好的方向：**激活全程留在核显**，把 SiLU/RMSNorm/
   softmax 做成小 OpenCL kernel，让一个相位变成"上传一次 → 多次 GEMM → 下载一次"；
4. **每相位一次 finish**，而不是每次 GEMM 一次。

#### 7.118.11 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.10 executor prototype; renumber honesty to 7.118.11')]


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
