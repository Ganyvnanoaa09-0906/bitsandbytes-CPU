# -*- coding: utf-8 -*-
"""把「小向量梯度已补齐」的结果写进 plan 和报告。"""
import io

# ---------- 1) plan ----------
p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace("""| M4 | 10 层 + host emb/pos/head/CE + device AdamW（复用现有 trainer 骨架） | 10 步 loss 与 CPU 参考差 <0.02 | ✅ **10 步最大偏差 4.83e-3**（均值 2.27e-3）；loss 首步 4.910352 vs torch 4.910697（3.5e-4）（`ar_v2_multi.py` + `ar_v2_multi_selftest.py`） |""",
"""| M4 | 10 层 + host emb/pos/head/CE + device AdamW（复用现有 trainer 骨架） | 10 步 loss 与 CPU 参考差 <0.02 | ✅ **10 步最大偏差 1.85e-3**（均值 9.1e-4；补齐 ln/qk-norm/ls 小向量梯度后的复测值，之前只有大权重可训时是 4.83e-3）；首步 loss 4.910352 vs torch 4.910697（3.5e-4） |""")
s = s.replace("""**但比 CPU 的 3.1s/step（569 tok/s）慢 30%** ⇒ 要真用起来必须先优化 attention（见下） |""",
"""**但比 CPU 的 3.1s/step（569 tok/s）慢 30~40%** ⇒ 要真用起来必须先优化 attention（见下）。已补齐 ln/qk-norm/ls 的小向量梯度+AdamW（`rms_dw_from_ss`/`col_sum_prod` + host AdamW），所以现在 10 层里**所有**参数都可训，与 CPU trainer 语义一致 |""")
s = s.replace("""另外 ln/qk-norm/ls 这些小向量在 device 侧仍是**冻结**的（它们的梯度需要额外的列归约 kernel），
CPU 那边它们是可训的（实测 ckpt@12000 里 ls 已从 1e-2 长到 0.03~0.07），所以现在的 iGPU 训练
不能直接接管 CPU 的长期训练，必须先补上这 6 类小向量的梯度与 AdamW。""",
"""~~ln/qk-norm/ls 这些小向量在 device 侧冻结~~ —— 已补齐：`rms_dw_from_ss`（dw = Σ dh·x·rsqrt(ss)）
与 `col_sum_prod`（ls 的 Σ a·b），host 侧 float32 AdamW 后再传回 device。对拍结果：dls2 2.7e-2、
dls1 4~7e-2、dln 7~19e-2、dqw/dkw 7~20e-2（后者是 fp16 上 16384 行串行累加的抵消误差，
AdamW 会归一化，影响可接受但记在这里）。""")
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')

# ---------- 2) report 追加 ----------
TEXT = r"""

#### 7.222.5 补齐：ln / qk-norm / ls 小向量的梯度（§7.222.4 caveat 的修正）

上一版把 `ln1.w / ln2.w / q_norm.w / k_norm.w / ls1 / ls2` 留在 device 侧冻结，这是不能接管
CPU 续训的（CPU 那边它们可训，ckpt@12000 里 `ls` 已从 1e-2 长到 0.03~0.07、`ln.w` 1.027、`q_norm.w` 1.045）。
补法只用两个 kernel（都放在 `ar_v2_multi.py`）：

| kernel | 公式 | 用途 |
|---|---|---|
| `rms_dw_from_ss` | `dw[i] = Σ_r dh[r,i]·x[r,i]·rsqrt(ss[r]+1e-6)` | RMSNorm（含 per-head qk-norm）的 w 梯度；`ss` 复用 `rms_bwd_stats` 刚算好的每行均方 |
| `col_sum_prod` | `out[i] = Σ_r a[r,i]·b[r,i]` | `ls1/ls2` 的向量梯度 |

小向量共 ~22k 个数，走 **host AdamW**（float32 主副本 + m/v），每步批量下传/上传。
另外 `ls2` 不能原地缩放 `y`（要留未缩放的 `y` 才算得出 `dls2`），已改成写 `ys` 再相加。

**对拍（L0/L5/L9）**：`dls1` 4.3~6.9e-2、`dls2` 2.7~4.2e-2、`dln1` 6.4~19e-2、`dln2` 7.2~9.6e-2、
`dqw/dkw` 7.2~20.5e-2。后面这几项的误差是 fp16 上「2048/16384 行串行累加 + 抵消」造成的，
不是公式错：同一条链上的 `dW3/dW12/dWo` 都在 2.5~4.7e-2；而且误差随层数加深而变大（L0 6% → L9 19%），
说明是上游梯度误差的累积。AdamW 会把尺度归一化，对更新方向影响可接受，但**这里仍是个已知的精度短板**。

**复测 10 步轨迹（这次所有参数都可训，与 CPU trainer 语义一致）**：

| step | device | torch | 差 |
|---|---|---|---|
| 1 | 4.910352 | 4.910697 | −3.46e-4 |
| 2 | 4.857765 | 4.859503 | −1.74e-3 |
| 5 | 4.729832 | 4.730330 | −4.98e-4 |
| 10 | 4.546412 | 4.546566 | −1.55e-4 |

**最大偏差 1.85e-3、均值 9.1e-4**（验收线 0.02）—— 比"小向量冻结"那版的 4.83e-3 还更准，
因为更新量现在逐参数与 torch 对齐了。

**代价**：每步 +0.35s（反向多 6 个 kernel/层、优化器多 60 次小传输），一步 **4.47s / 458 tok/s**
（CPU 3.1s / 569 tok/s）。这个代价本身可以再压（`rms_dw_from_ss` 现在是串行累加，改成两级归约能省掉
大半），但**在 attention 没重写之前，省这 0.3s 没有意义**。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('report appended %d chars' % len(TEXT))
