"""Report 7.118.11: the optimisation attempt -- two walls, and why the hybrid executor cannot reach 0.59 s.

What was tried:
  * pool the device buffers by (role, nbytes) and drop the three queue.finish calls per GEMM; the only
    synchronisation is the blocking device->host read of the result;
  * keep the rest of the hybrid executor unchanged.

Result: whole step 7.051 s vs 6.854 s before -- no improvement. Per-GEMM attribution:
  shape            A copy   B copy   kernel   D2H    total
  2048x1024x512    0.62     0.25     3.13     1.38   5.38 ms
  2048x1368x512    0.66     0.33     4.78     2.21   7.98 ms
  512x1024x2048    0.72     1.28     2.85     0.33   5.18 ms
  2048x512x1368    1.77     0.44     4.74     0.85   7.80 ms

So per GEMM the hybrid pays 2-4 ms of movement/readback on top of a 3-4.8 ms kernel. Over 153 GEMMs
that is 0.3-0.6 s of pure plumbing per step, before the CPU-side phase even starts. Transfer bench:
pageable host memory 2.4-4.0 GB/s, USE_HOST_PTR ("pinned") 7-9 GB/s.

The second wall was hidden in plain sight: the executor's CPU-side phase (naive PyTorch RMSNorm/SiLU/
softmax/autograd) costs ~1.9 s per step, against the real model's fused-kernel figure (non-GEMM ops
were 3-9% of the 2.854 s step in 7.12). The naive prototype is not representative there.

Consequence: even with pinned memory and fused CPU kernels, the hybrid's floor is roughly
  0.515 s (kernels) + 0.3-0.6 s (movement) + 0.1-0.3 s (fused CPU phase) + overhead ~= 1.1-1.4 s
i.e. 0.4-0.5x of the 1030 reference. The hybrid route is therefore closed for the equivalence goal,
and the next architecture is a full-GPU step: activations GPU-resident, elementwise ops as OpenCL
kernels (fwd+bwd for RMSNorm/SiLU/softmax), dW and the optimiser on the device, CPU only orchestrates.
That is the only shape that can approach 0.59 s.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.11 诚实边界\n'
NEW = '''#### 7.118.11 优化尝试：两堵墙浮出水面——混合执行器的地板 ~1.1-1.4 s，够不到 0.59 s

按 §7.118.10 的清单先做最便宜的两刀：**设备缓冲池化**（按 (角色, 字节) 复用）**+ 去掉每次 GEMM
的三次 `queue.finish`**（只留阻塞式 D2H 读结果）。结果：整步 **7.051 s**，对比朴素版 6.854 s ——
**没有改善**。逐项归因（每 GEMM）：

| 形状 | A 上传 | B 上传 | kernel | D2H | 合计 |
|---|---|---|---|---|---|
| 2048×1024×512 | 0.62 ms | 0.25 ms | 3.13 ms | 1.38 ms | **5.38 ms** |
| 2048×1368×512 | 0.66 | 0.33 | 4.78 | 2.21 | **7.98 ms** |
| 512×1024×2048 | 0.72 | 1.28 | 2.85 | 0.33 | **5.18 ms** |
| 2048×512×1368 | 1.77 | 0.44 | 4.74 | 0.85 | **7.80 ms** |

**⇒ 第一堵墙（显式）：** 每个 GEMM 在 3~4.8 ms 的 kernel 之外，还要付 **2~4 ms 的搬运/读回**；
153 个 GEMM ⇒ **每步 0.3~0.6 s 纯管线费**，这还没算 CPU 相位。传输实测：pageable 2.4~4.0 GB/s，
USE_HOST_PTR（"pinned"）7~9 GB/s——pinned 只值 2×，补不上这个坑。

**⇒ 第二堵墙（隐蔽）：** 执行器的 CPU 相位是**朴素 PyTorch 实现**（RMSNorm/SiLU/softmax/autograd），
实测 **~1.9 s/步**；而真实模型的融合内核版非 GEMM 只占 2.854 s 的 **3~9%**（§7.12）。原型在
这一侧不具代表性——但即使换成融合内核（0.1~0.3 s），墙仍在：

```
0.515 s (kernel) + 0.3~0.6 s (搬运) + 0.1~0.3 s (CPU 相位) + 开销  ≈ 1.1~1.4 s   ⇒ 0.4~0.5×
```

**⇒ ★ 结论：混合执行器路线对"等效 1030"是封死的**（它注定要为 153 次 CPU↔GPU 往返付费）。
下一种架构必须是 **full-GPU step**：激活全程常驻核显、RMSNorm/SiLU/softmax 的前向+反向写成
OpenCL kernel、dW 与优化器也在设备侧，CPU 只做编排。只有这个形态才可能接近 0.59 s。

> 方法学记录：上一节把"每次 cl.Buffer + finish"当成主因，本节证明**池化+去 finish 无效**，
> 真正的主因是**每次调用的数据往返**（以及 CPU 侧实现的朴素度）。假设被自家数据推翻，照实记录。

#### 7.118.12 诚实边界
'''

EDITS = [(ANCHOR, NEW, 'insert 7.118.11 hybrid walls; renumber honesty to 7.118.12')]


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
