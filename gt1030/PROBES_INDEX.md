# gt1030/ 目录索引：交付物 / 补丁脚本 / 探针

> 目的：让复核者一眼分清"要保留可运行的东西"和"一次性的改动历史与死探针"。
> 目录里约 190 个 `.py`，绝大多数是**一次性的补丁脚本**或**已经跑完的探针**，
> 真正的交付物只有下面第一张表。
> 报告里引用的小节号见 `D:\work\research\CPU_FORGE_TECHNICAL_REPORT.md` §7.222.x。

## 一、交付物（保留、可运行、报告引用）

| 文件 | 作用 | 报告引用 |
|---|---|---|
| `ar_v2_multi.py` | **整机实现**：10 层 + head，单 context + 3GB arena，全部 kernel 与调用 | §7.222.5–.18 |
| `ar_v2_multi_selftest.py` | M4 验收：per-tensor 梯度对拍 + 10 步轨迹（判据 <0.02，实测 1.94e-3） | §7.222.10 |
| `ar_v2_train.py` | M6 训练器：`--ckpt --steps --lr --lr-min --warmup --spike --opt-state --save-every --val-frac` | §7.222.14/23 |
| `ar_v2_export.py` | 设备权重 → CPU trainer 同格式 ckpt（M5，可被 `sample_ar.py` 直接用） | §7.222.11 |
| `ar_v2_optstate.py` | AdamW 状态存取（`<ckpt>.opt.npz`，341MB）：解决 resume 冷启动发散 | §7.222.23 |
| `ar_v2_eval.py` | 单 ckpt 体检：NaN/absmax + val（382 张，带 SEM） | §7.222.23/24 |
| `ar_v2_valdump.py` | 导出逐 batch val loss（配对比较的输入） | §7.222.24 |
| `ar_val_compare.py` | **配对比较**：Δmean ± SEM(paired) + t 值（±0.001~0.003，比非配对小 20~50×） | §7.222.24 |
| `ar_token_analysis.py` | token 级指标（边际熵/码数/KL/邻接一致），**只当"塌成单码"报警器** | §7.222.22/24 |
| `ar_v2_val.py` | 早期 val 脚本（已被 `ar_v2_valdump.py` 取代，保留作对照） | — |
| `_runc_supervisor.py` | 监督者：spike/skip 触发回滚 + plateau 自动停 + health.json + opt 状态只留 2 个 | §7.222.23 |
| `_watch_runb.py` | 看护：每 N 步采样 + 指标（支持 `ARW_*` 环境变量复用） | §7.222.22 |
| `_igpu_overnight_40k.py` | 4 万步通宵跑的驱动器（训练→采样→写笔记） | §7.222.22 |
| `_igpu_runb.py` | run B 驱动器（同上模板） | §7.222.23 |
| `gemm_half.py` / `gemm_half_v2.py` / `gemm_half_attn*.py` / `gemm_half_nn_batched.py` / `gemm_half_sweep.py` | fp16 GEMM 自写实现与参数扫描（与 CLBlast 对拍的证据） | §7.222.16/18/20 |
| `flash_attn.py` | **真 flash attention**（Negative result：比现有三 kernel 慢 1.7×，见 §7.222.21） | §7.222.21 |

## 二、补丁脚本（一次性，保留作"改动历史"）

| 模式 | 数量 | 说明 |
|---|---|---|
| `_append_7xxxx.py` | 18 | 把 §7.222.x 小节追加进报告（文本证据） |
| `_fix_*.py` | ~60 | 每修一个 bug 一份：arena 泄漏、CE 缩放、grad_norm、rms_dw、causal 掩码、NaN 守卫、lr 调度、watch 语法… |
| `_vec_stage_*.py` | 4 | Stage A/B 向量化改造的落地脚本（含 A/B 对照） |
| `_integ_*.py` / `_optstep_batch.py` / `_xchain.py` / `_head_optdev.py` | 6 | 自写 GEMM 接入、opt_step 合并同步、层间 device→device |
| `_mk_*.py` / `_sup_env.py` / `_fix_lr_args.py` / `_fix_stop_rule.py` | 4 | 生成变体脚本、环境变量化、调度参数/停止规则 |

## 三、探针（`_t_*.py`，跑完即弃，但结论都进了报告）

| 前缀 | 用途 | 保留理由 |
|---|---|---|
| `_t_fair_*.py`、`_t_prof_now.py`、`_t_opt_prof.py`、`_t_launch_prof.py`、`_t_bwd_pieces.py` | 独占口径 profiling（CPU 挂起后测 fwd/bwd/opt 分解） | 报告 §7.222.17/19 的数字来源 |
| `_t_vec_ab*.py`、`_t_vec_check.py`、`_t_softmax_ab.py` | 新老 kernel 同进程 A/B + numpy 对拍 | §7.222.18/19 的加速比来源 |
| `_t_xfer.py`、`_t_devcopy.py`、`_t_subbuf*.py` | 传输成本（down/up/dev→dev） | §7.222.18 item3 的依据 |
| `_t_igpu_clock.py`、`_t_sgemm*.py`、`_t_attn_gemm2.py` | 核显 GEMM 天花板与参数扫描 | §7.222.16/20 |
| `_t_mem_probe*.py`、`_t_buflimit.py`、`_t_manyctx.py`、`_t_crossctx.py`、`_t_gridprobe.py` | 驱动缺陷定位（≥130 cl_mem / ≥3 context 会让队列失效） | §7.222.4 的证据 |

## 四、可直接删（死脚本 / 被取代）

- `_t_mem_probe{2..18}.py`（除 `_t_mem_probe7`，驱动缺陷证据）—— 一次性定位过程
- `_t_m4_dbg{2..19}.py`、`_t_sgemm{1..5}.py`、`_t_5l*.py`、`_peek_opt*.py`、`_t_m2*.py`、`_t_n1_dbg.py`、`_t_rms_dbg.py`
- `gemm_half_v2b.py`（v2 的 BK 扫描变体，结论已写进 §7.222.20）
- `gemm_half_db.py`（双缓冲尝试，**Negative**：`async_work_group_strided_copy` 只能 1-D 跨步，做不到 2-D tile）

> 建议：真要清理，把第三、四类整体移到 `gt1030/probes/` 并在报告里改路径；
> 第一类（交付物）**不要动**，它们被报告和文档按当前路径引用。
