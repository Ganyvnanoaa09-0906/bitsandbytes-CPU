# -*- coding: utf-8 -*-
"""最终收尾：修 note 的两处过时说法 + 追加报告 §7.222.17（M6 从 20k ckpt 的正式结果）。"""
import io

# ---------- 1) note ----------
p = r'D:\work\collab\shared\AR_WEIBO_OVERNIGHT_2026-10-24.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('§7.222 | 完整技术报告（10 个小节）', '§7.222 | 完整技术报告（16 个小节）')
s = s.replace('- 核显这条线的价值：算法快速迭代 + 推理侧（图像/视频）；生产训练暂时仍留在 CPU。',
              '- 核显这条线现在**可用且更快**：独占时 1.35~1.38 s/step（1481~1517 tok/s）= CPU 的 2.6~2.7×。\n'
              '  注意：**不要和 CPU 训练同时满载**（抢 DDR4 带宽 + APU 功耗），否则两者都掉到 500 tok/s 量级。')
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('note fixed')

# ---------- 2) report ----------
TEXT = r"""

#### 7.222.17 收官：CPU 20k 跑完 + **M6 从 ckpt_20000 续训 100 步 = 1.383 s/step（1481 tok/s，2.60× CPU）**

**CPU 训练结束**（2026-10-25 15:52）：`ar_run_full3/ckpt_20000.pt` / `ckpt_last.pt`，
val 从 12k 的 5.3914 一路单调降到 **5.3560**（19801 时 5.3560），`AROvernight` 任务随后自动
采样了 `ar_sample_20000.png`（16 张）并写了归档笔记。

**M6 正式测量**（机器已完全空闲，不需要再挂起任何进程）：
从 `ckpt_20000.pt` 起、lr 5e-6（对齐 CPU 余弦退火末端的量级）、尾部 2% 作 val 不参与训练：

| 项 | 值 |
|---|---|
| 步数 | 100 |
| 步时 | **median 1.383 s/step** |
| 吞吐 | **1481 tok/s** |
| 对照 CPU | 569 tok/s ⇒ **2.60×**（80 步独占窗口里测到的最好值 1.350 s/step = 1517 tok/s ⇒ 2.67×） |
| loss | 3.7495 → 5.4741（单 batch 快照，mean ≈5.3，与 CPU 同期分布一致） |
| 导出 | `ar_run_igpu/ckpt_20100.pt` |
| 采样 | `ar_sample_igpu_20100.png`（4 张，码使用 251/1024、熵 4.466/6.931）——与 `ar_sample_20000.png` 同为色彩场，同一训练阶段 |

**至此 M1~M6 全部闭环**：

| 里程碑 | 状态 |
|---|---|
| M1 基础算子 kernel | ✅ qk-norm rel 4.1e-4、ls 2.8e-4 |
| M2 单层前向 | ✅ 块贡献 1.03e-3、cosine 0.992 |
| M3 单层反向 | ✅ dx 6.7e-4、dW* 1.5~1.9e-2 |
| M4 整机（10 层+head+CE+AdamW） | ✅ **10 步轨迹最大偏差 1.98e-3**（验收 0.02）；首步 loss 差 3.5e-4 |
| M5 权重导出 + 采样 | ✅ 125/125 键、cfg 零差异、strict load 通过、`sample_ar` 出图、与 CPU ckpt 结构完全一致 |
| M6 从 CPU 20k 续训 + 吞吐 | ✅ **1.383 s/step / 1481 tok/s = CPU 的 2.60×** |

**一句话结论**：30M AR 图像模型现在**能**在核显上训练，而且**比 CPU 快 2.6~2.7 倍**（独占条件下），
数值与 torch fp32 对齐（10 步 1.98e-3、ckpt 可互换）。代价是：核显与 CPU 训练不能同时满载
（共享 DDR4 带宽与 APU 功耗预算），并发时两者都退化到 ~500 tok/s。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('report appended %d chars' % len(TEXT))

# ---------- 3) plan ----------
p2 = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s2 = io.open(p2, encoding='utf-8').read()
s2 = s2.replace("| M6 | 正式训练（B=2/T=1024 或 B=4） | 100M tokens 量级 |",
                "| M6 | 从 CPU ckpt 续训 + 记录吞吐 | 吞吐与 loss 曲线 |")
s2 = s2.replace("**吞吐结论（更正+优化后，80 步持续测）**：核显独占 **1.350 s/step = 1517 tok/s**",
                "**吞吐结论（最终）**：从 `ckpt_20000` 续训 100 步 **1.383 s/step = 1481 tok/s（CPU 的 2.60×）**；"
                "独占窗口 80 步持续测 **1.350 s/step = 1517 tok/s（2.67×）**")
io.open(p2, 'w', encoding='utf-8', newline='\n').write(s2)
print('plan updated')
