# 真实模型 CPU 训练步剖面：Qwen3.5-0.8B LoRA（R5-4500U）

机器：R5-4500U（6C/6T）+ 15.4GB，Windows，torch 2.13.0+cpu，fp32。
模型：Qwen3.5-0.8B（混合线性注意力 GDN + 全注意力），LoRA r=8 包 96 个线性层，
可训 3.19M；AdamW8bit。脚本：`profile_real_step.py`、`profile_step_extra.py`。

## 1. 一步的时间构成（median）

| seq | forward | backward | optimizer | total |
|---|---|---|---|---|
| 64 | 1418 ms | 1405 ms | **49.7 ms** | 2873 ms |
| 256 | 2970 ms | 3504 ms | 52.3 ms | 6527 ms |

LoRA 真可训只有 3.19M，优化器只占 **约 1.7%**；对比全参微调场景（见下）才显著。

## 2. 全参微调时优化器是大头（v1 对照，548M 可训）

| 场景 | optimizer 时间 | 有效带宽 |
|---|---|---|
| 548M 全参 AdamW8bit | **1255 ms** | ~7.0 GB/s（548M×16B / 1.255s） |

和 `bench_optimizer.py` 的带宽模型完全对上——全参微调时优化器是第二大头。

## 3. 算子占比（torch profiler，self CPU）

| 算子 | seq64 | seq256 |
|---|---|---|
| `aten::mm`（GEMM） | 22.4% | 23.0% |
| `aten::mul` | 3.4% | 4.9% |
| `aten::copy_` | 3.0% | 3.4% |
| `aten::bmm` | 1.2% | 2.3% |
| `aten::fill_` | 1.2% | 1.2% |
| `aten::sum` / `add_` / `add` | ~2.7% | 类似 |

GEMM 占绝对主导；`mul/copy_/fill_` 等逐元素合计约 12–17%，是"缓存/带宽压榨"的主要对象。

## 4. GDN 融合内核到底值多少

- 每步 18 次 fused GDN 调用；seq64 时 GDN 总耗时 61–69 ms（**2.9–3.1%**），seq256 时
  200 ms（3.3%）。所以短序列下打不打开 GDN patch 几乎无感（1.00x）。
- 但 seq512 时：未 patch 15427 ms vs patch 后 14151 ms = **1.09x**，且 GDN 的 recurrent
  成本随 T 线性增长。长序列（2k+）训练时 GDN 内核才是值得压的对象。

## 5. 真实模型的线程数：4 比 6 快 7–10%

| 线程数 | seq64 | seq256 |
|---|---|---|
| 2 | 3002 ms | 9097 ms |
| 3 | 2537 ms | 7444 ms |
| **4** | **2034 ms** | **6107 ms** |
| 5 | 2258 ms | 6152 ms |
| 6 | 2188 ms | 6797 ms |

静态建议"物理核数=6"在这个混合注意力模型上并不是最优：小算子和逐元素操作多，
线程同步/调度开销吃掉了多出来的两个线程。→ 已给 `torch_cpu_kit.tune_threads()`
在真实 step 上自动标定线程数（测 2..6，取中位最优并设回 torch）。

## 6. 下一步优化候选（按收益/风险排序）

1. **长序列 GDN 内核**（seq 2k+ 才划算）：先 profile 一个 2k/4k step，确认 GDN 占比再优化；
2. **逐元素算子融合**：`mul/copy_/fill_` 合计 12–17%，`torch.compile`(inductor) 在 Linux 上
   可试（Windows 需要 MSVC）；成功可再省 5–15%；
3. **线程数/批大小联合标定**：`tune_threads` + 打包实验（多进程单线程）组合；
4. GEMM 本身由 torch/oneDNN 控制，能改的只有线程与算子选择（SDPA 等），空间有限。
