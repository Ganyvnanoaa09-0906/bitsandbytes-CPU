# -*- coding: utf-8 -*-
"""追加 §7.222.23：长跑稳定性事故与修复（run A NaN / run B 发散 / opt-state / 监督回滚）。"""
import io

TEXT = r"""

#### 7.222.23 长跑稳定性：两次事故的根因与修复（含"指标 vs val"打脸的教训）

**事故 1（run A，10-25 夜 → 10-26 05:xx）**：从 CPU `ckpt_20000` 续训 4 万步，**第 13335 步（global 33335）变 NaN**。
根因链（日志与权重体检共同确认）：

1. **lr 调度只写进了日志、没真的生效** —— `opt_step(args.lr)` 收的是常数 3e-5，整轮都在用峰值 lr 硬跑
   （`lr_used` 只用于打印）。这是最贵的一个 bug：它让"warmup + cosine"这句配置完全没起作用。
2. 某步出现 **inf 梯度** → 全局 `grad_norm = inf` → 裁剪系数 `f = 1/inf = 0` → **`grad *= 0` 作用在 inf 上 = NaN**
   → 污染 `token_emb/row_emb/col_emb` 三张 host 侧表（正好是乘 f 的地方）。
   证据：step 13334 `loss=4.7068` 但 **`gnorm=inf`**，13335 起全 nan；`ckpt_35000` 起权重 `non-finite`。

**事故 2（run B，10-26 07:56 → 14:31）**：从 run A 最后一个干净点 `ckpt_30000`（val 5.1963）以 lr 2e-5 续跑 2 万步，
**从第一步就退化**：loss 3.4724 → 5.11(2k) → 7.31(6k) → 9.13(10k) → 9.53(20k)，gnorm 3.6→61→197→486→762→inf，
val 逐 ckpt 崩：32500 = 5.4676、35000 = 7.1810、40000 = 8.9036、45000 = 9.5466、50000 = 10.0224。
根因有两条，第二条是**结构性的**：

1. `ckpt_30000` 本身已经在发散边缘 —— run A 在 global 30000 时 gnorm 已 **3.85**、32000 时 **42.5**、33335 变 NaN。
   从"尖锐点"续跑本来就危险。
2. **resume 时 AdamW 的 m/v 是清零的**（ckpt 只存权重）。冷启动第一步的更新量 ≈ `lr·(1-b1)/sqrt((1-b2)g²)` ≈ `lr·3.16·sign(g)`，
   比热状态大一个量级 ⇒ 直接把模型推下去。这不是"再调小 lr"能根本解决的。

**修复（都已落地）**：

| 问题 | 修复 |
|---|---|
| lr 调度不生效 | `opt_step(lr_used, ...)`，日志与实参一致 |
| `inf × 0 = NaN` 污染权重 | `opt_step` 里 `grad_norm` 非有限 → **整步跳过**；`clip_grads` 的 f 非有限 → 直接 return；trainer 里 loss 非有限 → 跳过该 batch |
| 冷启动丢优化器状态 | `ar_v2_optstate.py`：每个 ckpt 旁存 `<ckpt>.opt.npz`（master/mom/vel + embedding 的 m/v，**341 MB**），`--opt-state 1` 自动存+自动恢复；已验证 load 路径（恢复到 129 个数组） |
| 梯度已经很大但还有限 | `--spike 100`：grad_norm>100 直接跳过该步（比 NaN 守卫更早） |
| 无人看守 | `_runc_supervisor.py`：连续 3 步非有限 **或** 最近 200 步 spike≥5 → taskkill + 回滚到最近 `health=ok` 的 ckpt（lr×0.5、warmup 50）；另有"连续 2 个 ckpt val 不破纪录就自动停"省时间；`<ckpt>.health.json` 记录 val/健康标记；`.opt.npz` 只留最近 2 个 |

**一个必须记住的测量教训**：token 级指标在小样本下噪声**大于效应**。同一 ckpt `ckpt_30000`：
4 张图 → 边际熵 7.28 / 码 140 / 邻接 h 15.5%；8 张图 → 6.90 / 125 / h 22.4%；16 张图 → 6.85 / 104.8 / h 27.5%。
而且 16 张口径下 **25k 的指标比 30k 好看**（熵 7.59 vs 6.85、KL 1.44 vs 2.04），**但 val 是 30k 更好**（5.1963 vs 5.2308）。
⇒ **结论只报 val（382 张固定划分）；token 指标只当"塌成单码"的报警器**（run B 那种熵≈0、邻接 100% 才是可信信号）。
这也是对 §7.222.22 里"抄左边行为被打断"那句话的撤回。

**核显预训练的实测收益（唯一可信口径 = val）**：

| ckpt | 来源 | val（382 张） |
|---|---|---|
| ckpt_20000 | CPU 20k（干净基线） | 5.3551 |
| ckpt_25000 | 核显 +5k 步 | 5.2308（−0.124） |
| **ckpt_30000** | 核显 +10k 步 | **5.1963（−0.159，−3.0%）** |
| ckpt_32500+ | run B（发散） | 5.47 → 10.02（废） |

⇒ 核显跑 10 小时换来 val −0.159，这是**真实收益**；之后两次续跑失败是"尖锐点 + 冷启动 + 缺守卫"，
与"核显能不能训"无关（run C 从 25000 稳定段起跑后 gnorm 稳在 0.88~1.13）。
"""
with io.open(r'D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md', 'a', encoding='utf-8') as f:
    f.write(TEXT)
print('appended %d chars' % len(TEXT))

p = r'D:\work\collab\shared\IGPU_AR_V2_PORT_PLAN.md'
s = io.open(p, encoding='utf-8').read()
s += """
## 长跑稳定性（10-26）

- run A：global 33335 变 NaN。根因：lr 调度没真正生效（一直常数 3e-5）+ `inf×0=NaN` 污染 embedding
- run B：从 `ckpt_30000`（已在发散边缘，gnorm 3.85→42.5）冷启动 Adam 续跑 → val 5.47→10.02 逐 ckpt 崩
- 修复：`ar_v2_optstate.py`（每个 ckpt 存 AdamW 状态 341MB）、`--spike 100`、非有限跳过、`_runc_supervisor.py`（自动回滚 + 不破纪录自动停 + health.json + opt 状态只留 2 个）
- **实测收益（val，382 张）**：20k 5.3551 → 25k 5.2308 → **30k 5.1963**（−3.0%）
- **测量教训**：token 指标在 4~16 张样本下噪声 > 效应（同 ckpt 差到 0.4 bit），结论只报 val
"""
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('plan updated')
