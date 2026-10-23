# 15.4GB CPU 训练内存预算曲线（2026-10-23 实测）

环境：CPU 6线程，Windows 15.37GB 物理内存；模型 fp32，bnb AdamW8bit（除注明）；
`gt1030/igpu_train.py --backend cpu`；峰值用进程 WorkingSet 300ms 采样。**每个配置单进程、
跑完即退出**；free RAM 低于安全线会杀进程并标注。

## 1. 实测表

| 模型 | 模式 | seq | batch | 优化器 | 峰值 RSS | 步时 | 结果 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B（752M） | full | 64 | 1 | AdamW8bit | **9.31GB** | 9.37s | PASS |
| Qwen3.5-0.8B | full | 128 | 1 | AdamW8bit | **9.09GB** | 13.24s | PASS |
| Qwen3.5-0.8B | full | 256 | 1 | AdamW8bit | **9.55GB** | 18.60s | PASS |
| Qwen3.5-0.8B | LoRA r8 | 64 | 1 | AdamW8bit | 5.88GB（既有） | 2.65s | PASS |
| deepseek-coder-1.3B（1346M） | full | 64 | 1 | AdamW8bit | **10.30–11.41GB** | 27.8s | PASS |
| deepseek-coder-1.3B | full | 64 | 1 | **AdamW4bit** | 11.39GB | 20.6s | PASS |
| deepseek-coder-1.3B | full + grad-ckpt | 64 | 1 | AdamW8bit | 11.25GB | 22.0s | PASS |
| deepseek-coder-1.3B | **LoRA r8** | 64 | 1 | AdamW8bit | **7.62GB** | **4.43s** | PASS |
| Qwen3-1.7B（1720M） | full | 64 | 1 | AdamW8bit | **爆点**（加载中 RSS6.63，free2.06→杀） | — | **FAIL：未到 forward** |
| Qwen3-1.7B | LoRA r8 | 64 | 1 | AdamW8bit | **爆点**（加载中 RSS7.76，free0.55→杀） | — | **FAIL：未到 step** |
| 1.3B full | seq128/256、batch2 | — | — | — | 未测（时间盒/风险） | — | — |
| 1.7B any | 其他 seq/batch | — | — | — | 未测 | — | — |

**15.4GB 能放下什么（结论）**：
- **舒适**：0.8B full（任何 seq64–256，≤9.6GB）、1.3B LoRA（7.6GB）、0.8B LoRA（5.9GB）。
- **可跑但紧**：1.3B full（10.3–11.4GB，需干净环境；步时27.8s）。
- **放不下**：**1.7B full**（权重+梯度+8bit状态本身>15.4；加载阶段即爆）。
- **1.7B LoRA 理论上 7.9GB 应能放**（estimator），但本机当时 free 已低（加载中 free 0.55–2GB），
  **未完成 step，不能算 FAIL 也不能算 PASS**；**理论需求 7.86GB 可放，需干净环境（重启/清 page cache/
  无 page cache 压力）复测，别再硬跑**。

## 2. planner 校准（`estimate_training_memory_gb`）

默认 `activation_gb=1.0, runtime=0.4, AdamW8bit=2.0156 B/param`：

| 配置 | 预测 total | 实测 | 误差（实测−预测） |
|---|---|---|---|
| 0.8B full | 8.42GB | 9.31GB | **+0.89**（低估） |
| 0.8B LoRA(5.41M) | 4.23GB | 5.88GB | **+1.65**（低估） |
| 1.3B full | 13.96GB | 10.30–11.41GB | **−2.5~−3.7**（高估，实际 opt/grad 未全驻留） |
| 1.3B LoRA(7.77M) | 6.46GB | 7.62GB | **+1.16**（低估） |
| 1.7B full | 17.44GB | 加载即爆 | 预测>15.4，一致 |
| 1.7B LoRA | 7.86GB | 未完成 | 理论可放，实测受 free 限制 |

**校准建议**：
- LoRA/小模型：`activation_gb` 实际约 **1.8–2.5GB**（seq64），默认1.0偏小；用
  `activation_gb≈2.0` 可把 0.8B/1.3B LoRA 误差压到 ≤0.4GB。
- 1.3B full：默认公式**高估约 2.5–3.7GB**（本机峰值未含全部 optimizer 常驻），计划并发时应
  按“实测优先”；保守规划可用预测，但会少放。
- 规划判断：**先算 weights+grads+opt，>14GB 直接判放不下**；再以实测 activation 加回。
- 单进程、干净环境、free≥4GB 才启动 full1.3B；1.7B full 不建议在本机尝试。

## 3. 复现命令

```powershell
python gt1030/igpu_train.py --model <MODEL> --backend cpu --mode full \
  --optimizer adamw8bit --lr 1e-4 --steps 1 --tokens 64 --batch 1 --log-every 1
# LoRA: --mode lora --lora-r 8
```
峰值用外部 300ms 采样 WorkingSet；free<0.7GB 自动终止（避免宿主崩溃）。
## 0. OOM 事故（2026-10-23 20:53）与防御

Windows 事件 2004：python.exe 虚拟内存 **22.2GB**，33s 后 DSH host 崩（Wan T5+DiT+VAE+CFG
同时驻留）。现所有重内存任务要求：
- `gt1030/memwatch.py`：available<2.5GB 或 commit>85% 立即退出；trainer start/每步调用；
- staged loading（T5 encode→释放→DiT）；单任务；
- 预判 >12GB 的配置直接标“放不下”，不真爆。
