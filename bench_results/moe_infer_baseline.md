# CPU MoE 推理基线：fp32 vs 3-bit 物化 vs 3-bit fused（2026-10-19）

脚本：`moe_infer_bench.py`。scaled Mixtral（hidden 512 / 4 层 / 8 专家 / top-2，53.5M 参数，
随机权重，eval 模式），单次前向含全部层 + lm_head。这是与 KTransformers 这类推理期专家
卸载方案对照用的**可复现本机基线**，不是同台对比数字。

## 结果

| 专家实现 | 权重内存 | prefill seq=64 | decode seq=1 | RSS |
|---|---|---|---|---|
| fp32（原始 F.linear） | 0.205 GB | **43.1 ms（1485 tok/s）** | **11.96 ms** | 0.57 GB |
| 3-bit 物化 + F.linear | **0.038 GB（-81%）** | 74.1 ms（864 tok/s，+72%） | 18.92 ms（+58%） | 0.42 GB |
| 3-bit fused GEMM（1 线程） | 0.038 GB | 156.1 ms（410 tok/s） | 22.54 ms | 0.40 GB |
| 3-bit fused GEMM（4 线程） | 0.038 GB | 96.0 ms（667 tok/s） | 24.10 ms | 0.49 GB |

## 结论

1. **小专家矩阵上，3-bit 物化 + oneDNN matmul 比 fused GEMM 更快**：每个命中专家
   (M≈16–64, N=2048/512, K=512/1024) 太小，fused 内核的线程启动 + 手写 dot 效率
   打不过"一次快速解包 + oneDNN 批量 matmul"；fused 内核的优势在 lm_head 那种大 N
   （§7.143）。
2. 3-bit 专家在推理侧的本机代价：**内存 -81%，prefill -42% 吞吐，decode +58% 延迟**。
   是否值得取决于内存预算——15.4GB 机器上 8 专家 53M 模型无所谓，真实 14B MoE 上
   0.9GB→0.17GB 量级才有意义。
3. **KTransformers 对照的正确维度**：它是"专家卸载到 CPU/磁盘 + 热专家缓存"，我们是
   "专家常驻 3-bit 量化"；同模型/同内存预算下应比较**单位内存能放多少专家 + token/s**。
   本表给出了本机的 3-bit resident 基线，后续可在同样内存预算下做 offload 实验。
4. 优化空间：expert-wise fused GEMM 要赢需要小 M 专用内核（单线程、行向量化、批量多专家
   一起算），或把解包融进 oneDNN 前的 batch；当前物化路径已足够快。

## 复现

```bash
python moe_infer_bench.py
# 依赖：transformers>=5.16（Mixtral 融合专家）、torch CPU、q3bit_kernel.dll / q3bit_gemm.dll
```
