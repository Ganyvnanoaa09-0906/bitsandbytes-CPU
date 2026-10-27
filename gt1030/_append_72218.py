# -*- coding: utf-8 -*-
"""追加 §7.222.18：小 kernel 向量化/融合（item 1）+ 层间 dev→dev（item 3）的结果。"""
import io

TEXT = r"""

#### 7.222.18 小 kernel 向量化 + 层间 device→device：kernel 级 **2.7~2.9×**，单步 1.66 → 1.41 s

按 §7.222.16 的清单做前两项。**先说测量口径**：这台机器上我自己的 agent 进程占了 40~70% CPU，
所以"整步绝对时间"在不同时刻不可比；因此**主要证据用同进程背靠背的 A/B**（旧 kernel 仍编译在 program 里，
可以直接对拍），单步时间只作参考。

**① 小 kernel 向量化 + 融合（item 1）**

诊断：`rms_apply / split_silu / add_vec / scale_vec / split_q / merge_o / qk_norm` 全都是
"grid = 元素个数、每线程 1 个元素"，1M 线程 / 384 lane = 2730 波，每波只有 2 load + 1 store，
纯等内存延时。实测每个 0.4~2.5 ms，而它们的真实数据量只值 0.05 ms（40 GB/s）——**差 10 倍以上**。
`qk_norm` 更极端：grid = 行数，每线程串行扫 64 个元素。

做法：每线程 8 个元素（`half8` 向量读写）+ 把成对的融成一个：

| 组 | 旧 | 新 | 加速 |
|---|---|---|---|
| `rms_stats`+`rms_apply` → `rms_fwd_fused` | 2.19 ms | **0.34 ms** | **6.4×** |
| `rms_bwd_stats`+`rms_bwd_apply` → `rms_bwd_fused` | 5.01 ms | **0.60 ms** | **8.3×** |
| `scale_vec`+`add_vec` → `scale_add_v` | 0.60 ms | **0.07 ms** | **8.4×** |
| `split_silu` → `split_silu_v` | 1.90 ms | 1.03 ms | 1.8× |
| `split_silu_bwd` → `split_silu_bwd_v` | 2.23 ms | 1.61 ms | 1.4× |
| `split_q` → `split_q_v` | 1.27 ms | **0.10 ms** | **12.6×** |
| `split_kv` → `split_kv_v` | 1.71 ms | 0.79 ms | 2.2× |
| `merge_o` → `merge_o_v` | 1.19 ms | 0.41 ms | 2.9× |
| `split_dout` → `split_dout_v` | 0.98 ms | 0.66 ms | 1.5× |
| `qk_norm_fwd` → `_v` | 2.53 ms | 0.81 ms | 3.1× |
| `qk_norm_dx` → `_v` | 4.44 ms | 1.63 ms | 2.7× |
| `merge_dqkv` → `merge_dqkv_v`+`merge_dkv_v` | 2.69 ms | 1.15 ms | 2.3× |

**合计每层 27.5 ms → 9.9 ms（2.77×），即每步省约 176 ms。**

**② 层间 x 走 device→device（item 3）**

实测本机：`down` 2MB（必须同步）**1.909 ms**、`up` 2MB 0.652 ms、**dev→dev 2MB 仅 0.277 ms**。
原来每层前后各一次 down+up（~2.16 ms/层）外加两次 `d.finish()`（在负载下每次同步都可能几毫秒）。
改成：embedding 之后上传一次、层与层之间 dev→dev、最后同步一次（前向/反向各一次）。

效果（同一台机器、前后间隔 40 分钟、负载相近的两次 40 步实测）：

| 版本 | min | **median** | mean |
|---|---|---|---|
| item 1（A+B，向量化+融合） | 1.388 s | 1.673 s | 1.658 s |
| +item 3（层间 dev→dev） | 1.335 s | **1.407 s** | 1.469 s |

⇒ **中位数降 16%**（1.66 → 1.41 s/step），比按"省 40ms"估的多——因为每步去掉了 **20 次 `d.finish()`**，
在 CPU 被抢占时每次同步都要几毫秒。

**③ 正确性**：10 步轨迹复测 **1.94e-3**（验收 0.02），1 层对拍 loss 差 1.84e-4、
dW3 3.256e-2 / dWqkv 2.828e-2 / dqw 2.546e-2 与改造前**逐位相同** ⇒ 只换了实现，没动语义。

**待做（item 4）**：attention 的 softmax 与 S 融合（flash 风格）、bwd 用重算代替存 S/P/dP/dS，
预计再省 ~200 ms；之后软件层面的下限约 **1.1 s/步（1030 分数 ~0.54）**。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s += """
## 优化进展（2026-10-25 晚）

| 项 | 结果 |
|---|---|
| 小 kernel 向量化（half8）+ 融合 | 每层 27.5 → 9.9 ms（**2.77×**，每步省 ~176 ms） |
| 层间 x 改 device→device | 中位步时 1.66 → **1.41 s**（省 20 次 finish + 18 次 host 往返） |
| 10 步轨迹 | 1.94e-3（验收 0.02），梯度与改造前逐位相同 |
| 下一步 | attention 融合（flash 风格），目标 ~1.1 s/步 |
"""
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
