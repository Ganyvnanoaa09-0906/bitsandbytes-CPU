# 雷霆-2bit：Qwen3-1.7B 推理四精度极限实验（2026-10-23）

目标：验证“2bit 权重流量减半、有效带宽翻倍”是否在 CPU 推理兑现。
模型 `D:\work\llm_env\Qwen3-1.7B`（fp16 3.87GB），block64 absmax group-wise；单进程/精度，
`gt1030/igpu_train.py` 无关；prefill M=64/128、decode M=1；峰值用 WorkingSet 采样。

## 结果表

| 精度 | 权重字节（模块内省） | RSS | prefill64 | prefill128 | decode M=1 | decode tok/s | 有效权重带宽* | PPL | top-1 vs fp16 |
|---|---|---|---|---|---|---|---|---|---|
| **fp16** | **3.441GB** | 4.06GB | 5988.8ms | 12602.9ms | **239.0ms** | **4.18** | **14.40GB/s** | **1.538** | — |
| **NF4**（197 Linear） | 0.457GB | 2.34GB | 4461.6ms | 5297.5ms | **1482.7ms** | 0.67 | **0.308GB/s** | 1.564 | 14.8% |
| **q3bit**（196 Linear） | 1.243GB | 2.03GB | 8817.5ms | 16366.3ms | **812.3ms** | 1.23 | **1.531GB/s** | **14.579（崩）** | 7.0% |
| **q2bit** | 理论≈0.5GB | 未完成 | — | — | **FAIL**（转换/forward 中 peak9.22GB、free0.79GB 被杀） | — | **无有效值** | — | — |

\* 有效带宽 = 权重字节 / decode 时间；不是实际 DRAM 流量（还含 dequant/计算/激活）。

## 结论

1. **“2bit 倒欠一半带宽”不成立**：decode M=1 的瓶颈是**逐层解包/反量化 + CPU GEMM**，不是权重字节。
   NF4 权重只有 fp16 的 13%，decode 反而慢 **6.2×**（239→1483ms），有效带宽 0.31GB/s（fp16 14.4）。
   q3bit 也慢 3.4×，有效带宽 1.53GB/s。
2. **q2bit 当前实现不可用于推理**：`Linear2bit` 每次 forward 反量化整层，加上模型驻留，
   转换/推理阶段即 OOM（peak9.22GB，free0.79GB）；没有 fused 2bit dequant+GEMM，**无法测带宽**。
   要兑现 2bit 需要专用 AVX2 LUT fused kernel（4-entry，16 权重/字节），当前无此内核。
3. **质量**：NF4 PPL 1.564 接近 fp16 1.538（可接受）；q3bit PPL **14.58 崩溃**（当前实现/权重
   映射有 bug，不能用于质量结论）；q2bit 未完成。top-1 对随机 token 仅14.8%/7.0%，不能当语义质量。
4. **哪里值得**：NF4 只值得作为“内存/放得下”手段（RSS 4.06→2.34GB），**不值得为提速**；
   q2bit 在现有 CPU 解包实现下**不值得**，除非先写 fused LUT 内核并验证小 M 吞吐。

## 复现

```powershell
python gt1030/thunder2bit_bench.py --precision fp16
python gt1030/thunder2bit_bench.py --precision nf4
python gt1030/thunder2bit_bench.py --precision q3bit
python gt1030/thunder2bit_bench.py --precision q2bit   # 当前 OOM，预期失败
```
日志/JSON：`research/bench_results/thunder2bit*`、`research/bench_results/thunder2bit/`。
