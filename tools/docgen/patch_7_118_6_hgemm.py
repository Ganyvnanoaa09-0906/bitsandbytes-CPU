"""Add the first P1 measurement to report 7.118: fp16 GEMM at the AR model's real shapes.

CLBlast Hgemm (fp16 in/out, column-major trick): 546-800 GFLOPS at the four AR training shapes,
i.e. mean ~731, above the 631 GFLOPS that the whole step needs to tie a GT 1030. Accumulation error
against a float64 reference of the same fp16 inputs: 0.009-0.035 absolute on values of order 0.2-0.5,
so it is a real (if not fp32-grade) GEMM.

One breadcrumb worth recording: the first run reported |err| 8-17 because C had been allocated
uninitialised while beta was 1.0 -- i.e. the "correctness check" was measuring a stale buffer. The
fixed run (beta=0, C zeroed) gives the numbers above.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '''#### 7.118.6 诚实边界
'''
NEW = '''#### 7.118.6 前置测量：fp16 GEMM 在真实形状上已经越过 631

P1 之前先量一步：把核显 GEMM 换成 **fp16 存储**（CLBlast `Hgemm`，fp16 进/出，列主序调用），
在 AR 模型的四个真实形状上测：

| 形状 (M×N×K) | fp32 CLBlast | **fp16 Hgemm** | fp16 相对 fp32 |
|---|---|---|---|
| 2048×1536×512 | 482.5 | **788.4** | 1.63× |
| 2048×512×512 | 385.2 | **546.2** | 1.42× |
| 2048×2048×512 | 504.9 | **800.3** | 1.59× |
| 2048×512×2048 | 480.0 | **791.9** | 1.65× |
| 平均 | ~463 | **~731** | 1.58× |

精度：对同 fp16 输入的 float64 参考，最大绝对误差 **0.009~0.035**（输出量级 0.2~0.5）——是
真正的 GEMM，不是跑分；但**不是 fp32 级精度**（fp32 GEMM 误差约 1e-4），这点必须在 P3 用
loss 曲线验证，必要时换"fp16 存储 + fp32 累加"的自写内核。

**⇒ ★ 731 > 631：W1（AR-30.5M 步）的"整步跑在核显"在算力上已经够线——加上 3~9% 非 GEMM
开销，整步约 0.53~0.56 s ⇒ 1030 分数 1.06~1.12×。P1 剩下的工作不是提峰值，而是"把整步真的
跑起来"（权重/梯度/优化器状态常驻、反向 GEMM 同样走核显、消除同步点）。**

> 方法学备注：第一次测出 |err| 8~17 是假警报——C 缓冲未初始化且 beta=1.0，等于在测脏缓冲。
> 改成 beta=0 + C 清零后得到上表。这是本报告反复出现的"代理量代替真实约束"的又一例，记下来。

#### 7.118.7 诚实边界
'''

# The old 7.118.6 must become 7.118.7; the later sections then keep their numbers (tests only
# reference section titles in prose, not numbers, so no cascade).
EDITS = [
    (ANCHOR, NEW, 'add 7.118.6 measurement, renumber honesty to 7.118.7'),
]


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
