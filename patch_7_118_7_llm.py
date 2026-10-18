"""Report 7.118.8: generalise the GT 1030 equivalence question beyond the AR model.

Facts used:
  * capacity: a GT 1030 has 2 GB VRAM. AR-30.5M fp32 = 0.12 GB fits; Qwen3-1.7B fp32 = 6.8 GB and
    Qwen3-8B NF4 = 5.14 GB do NOT fit. So for those two, wall-clock equivalence cannot even be
    defined on a real 1030 -- the only comparable quantity is per-token throughput.
  * rate: frozen-base LoRA step ~= 4*N*tokens FLOPs (forward 2N + input-gradient 2N). Measured steps:
    8B NF4 37.5-38.1 s (report 8.3), 1.7B fp32 7.0 s. The tokens/step of those runs were not recorded
    in the text, so the table is parameterised by tokens.
  * iGPU fp16 GEMM at Qwen layer shapes (measured today): 760-903 GFLOPS, mean ~830, vs the 631
    GFLOPS the 1030's 55-60% efficiency implies.
"""
from __future__ import annotations

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

REPORT = r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md'

ANCHOR = '#### 7.118.6 前置测量：fp16 GEMM 在真实形状上已经越过 631\n'
NEW = '''#### 7.118.6 前置测量：fp16 GEMM 在真实形状上已经越过 631

先补一句**范围**：本节及 7.118.8 讨论的是"能不能等效"，而 GT 1030 的 2 GB 显存先划掉了一大半
workload——见 7.118.8 的容量表。AR 是"能装下"那一类里最容易测的样本，不是唯一目标。

'''
# insert the AR numbers after the scope note (the old table follows the original heading; we only
# prepend a paragraph, so the rest of the section stays intact).

ANCHOR2 = '#### 7.118.7 诚实边界\n'
NEW2 = '''#### 7.118.7 不止 AR：8B 的口径不同——容量先出局，速率未必输

**（a）容量判据：** GT 1030 只有 2 GB 显存。

| workload | 权重体积 | 1030 装得下？ |
|---|---|---|
| AR-30.5M（fp32） | 0.12 GB | ✅ |
| Tiny-SD（26M） | 0.10 GB | ✅ |
| Qwen3-1.7B LoRA（fp32 冻结底座） | **6.8 GB** | ❌ |
| Qwen3-8B NF4 LoRA | **5.14 GB**（+激活/工作区） | ❌ |

**⇒ 对 8B/1.7B，"速度等效 1030"在字面上不成立——1030 不是慢，是根本放不下。** 正确的比较
只剩"每 token 的等效算力"：把墙钟时间除以 4·N·tokens（冻结底座 LoRA：前向 2N + 输入梯度 2N，
适配器项可忽略），得到有效 GFLOPS，再与 1030 的 631 GFLOPS 比。

**（b）速率判据（我们实测 vs 1030 的 631）：**

| workload | 实测步时 | 512-token 口径 | 1024-token 口径 |
|---|---|---|---|
| Qwen3-8B NF4 LoRA（CPU） | 37.8 s | **0.70×**（443 GFLOPS） | **1.41×**（888 GFLOPS） |
| Qwen3-1.7B LoRA（CPU） | 7.0 s | **0.79×**（497 GFLOPS） | 1.58× |

> 说明：8B/1.7B 两次实测的 tokens/step 没有写进旧文，表中按 token 数参数化，不假装知道其中一个。

**（c）核显在 LLM 形状上的实测 fp16 GEMM（今日，CLBlast Hgemm，fp16 进出）：**

| 形状 | GFLOPS | 形状 | GFLOPS |
|---|---|---|---|
| 1.7B qkv/o 512×2048×2048 | **759.8** | 8B qkv/o 512×4096×4096 | **801.1** |
| 1.7B mlp-up 512×6144×2048 | **902.8** | 8B mlp-up 512×12288×4096 | **897.6** |
| 1.7B mlp-down 512×2048×6144 | **762.7** | 8B mlp-down 512×4096×12288 | **853.0** |

**均值 ~830 GFLOPS，全部高于 631。** 折算：8B 512-token 步的纯 GEMM 下限 ≈16.8 TFLOP/830
≈ **20.2 s**，加 10~15% 非 GEMM/反量化开销 ≈ **22~23 s/步** ⇒ 1030 分数 **1.1~1.2×**；1.7B
512-token ≈ **4.4 s**（分数 ~1.27×，未计开销）。

**⇒ ★ 结论：不是只有 AR 能到。**
1. **1030 装得下的（AR、Tiny-SD）** ⇒ 同速等效可达（AR 的算力已经越过 631）；
2. **装不下的（8B/1.7B）** ⇒ "等效"只能退化为"每 token 算力当量"，而核显在 LLM 形状上
   760~903 GFLOPS 也已越过 631 ⇒ **速率目标可达**；
3. **真正的工程债**不在峰值，而在：① NF4 反量化要融进核显 GEMM（不然每步要物化 16 GB fp16）；
   ② GDN/注意力反向留在 CPU（Qwen3.5 的线性注意力不是 GEMM 主导）；③ 消除每步同步点。
4. 精度警告：Hgemm 的累加是半精度（误差 0.04~0.2 绝对量级），P3 必须用 loss 曲线验收；
   必要时换"fp16 存储 + fp32 累加"内核。

#### 7.118.8 诚实边界
'''

EDITS = [
    (ANCHOR, NEW, 'prepend scope note to 7.118.6'),
    (ANCHOR2, NEW2, 'insert 7.118.7 LLM workloads; renumber honesty to 7.118.8'),
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
