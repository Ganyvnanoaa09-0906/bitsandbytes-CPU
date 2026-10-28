# CPU MoE 推理：内存预算 vs 吞吐曲线（2026-10-19）

脚本：`scripts/misc/moe_infer_cache_sweep.py`。这是 KTransformers「专家卸载 + 热缓存」思路的**resident
量化对照基线**：专家统一存 3-bit（~11% fp32 专家内存），每层按 router 使用频率把 top-C
个专家解包成 fp32 常驻缓存，其余专家每次调用按需解包（矩阵很快，oneDNN 跟着算）。

模型：scaled Mixtral 53.5M 参数（hidden 512 / 4 层 / 8 专家 / top-2），随机权重，eval。
fp32 基线：0.205 GB，prefill seq64 43.1 ms（1485 tok/s），decode seq1 11.96 ms。

| 每层热缓存 C | 权重内存 | 占 fp32 | prefill seq64 | tok/s | decode seq1 | RSS |
|---|---|---|---|---|---|---|
| 0（全 3-bit） | 0.038 GB | 19% | 80.3 ms | 797 | 21.88 ms | 0.43 GB |
| 2 | 0.080 GB | 39% | 76.0 ms | 843 | 18.84 ms | 0.48 GB |
| 4 | 0.121 GB | 59% | 70.3 ms | 911 | 14.77 ms | 0.62 GB |
| 6 | 0.163 GB | 80% | 56.0 ms | 1142 | 12.65 ms | 0.79 GB |
| 8（全 hot） | 0.205 GB | 100% | 44.4 ms | 1441 | 12.00 ms | 0.60 GB |

结论：

1. **曲线单调、可调**：内存 19%→100% 对应 prefill 797→1441 tok/s（约 1.8×）、decode
   21.9→12.0 ms（约 1.8×）。每个"热专家"约值 0.021 GB / ~80 tok/s（prefill）。
2. 与 KTransformers 的正确对照维度就是这条曲线：**同内存预算下能保持多少 tok/s**。
   KTransformers 用"冷专家卸载到 DRAM/磁盘 + 热专家驻留"，本曲线是"冷专家 3-bit 驻留 +
   热专家 fp32 缓存"，两者可放在同一坐标里比（需要其运行时才能实测，本机不伪造数字）。
3. 实用建议（15.4GB 机器）：真实 14B MoE fp32 专家约 50GB 放不下；3-bit 后 ~5.6GB，
   每层热缓存 2/8 约 +2GB → 能跑且热路径不慢；若内存更紧，C=0 也能用（预填 -46% 吞吐）。
4. 与 §7.151 一致：冷专家走"解包 + oneDNN matmul"比 fused GEMM 快，这里沿用了该路径。
