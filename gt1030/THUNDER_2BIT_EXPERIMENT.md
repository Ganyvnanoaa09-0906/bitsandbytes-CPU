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
## 4. NF4 fused AVX2 GEMV（M=1）实测（2026-10-23）

现成 `bnb.functional.gemv_4bit`（AVX2 LUT）单层，vs 解包物化 F.linear：

| shape | N×K | 权重字节 | fused | material | speedup | fused 有效带宽 | relL2 |
|---|---|---|---|---|---|---|---|
| qkv | 4096×2048 | 4.72MB | **0.899ms** | 8.72ms | **9.7×** | 5.25GB/s | 0.0017 |
| o | 2048×2048 | 2.36MB | 0.522ms | 4.55ms | 8.7× | 4.52GB/s | 0.0017 |
| gate | 6144×2048 | 7.08MB | 1.531ms | 13.60ms | 8.9× | 4.62GB/s | 0.0017 |
| up | 6144×2048 | 7.08MB | 1.395ms | 13.40ms | 9.6× | 5.07GB/s | 0.0017 |
| down | 2048×6144 | 7.08MB | 1.472ms | 13.12ms | 8.9× | 4.81GB/s | 0.0017 |
| lm_head | 151936×2048 | 175MB | **28.2ms** | 533ms | **18.9×** | 6.20GB/s | 0.0017 |

**整 token 估算**：28 层 × (qkv+o+gate+up+down=5.82ms) =162.9ms + lm_head28.2ms = **~191ms**；
对比 fp16 decode **239ms** ⇒ **约0.8×，仍不更快**；有效权重带宽约 **5.1GB/s**（fp16 14.4，
memcpy上限42.7）。**未达 <100ms/≥5–10GB/s 的“雷霆”门槛**。

**结论**：fused NF4 把“物化路径”的 1483ms 救到 ~191ms（7.8×），但**仍慢于 fp16 GEMV**；
2bit 再减半权重也救不了 dequant 计算/带宽效率，**不做 2bit fused 内核**；2bit 只在“放得下”
维度有意义（且当前 Linear2bit 仍 OOM）。

**正收益（可交付）**：`bnb.functional.gemv_4bit` fused AVX2 相对物化路径 **9–19×**
（lm_head 18.9×），relL2 0.0017；NF4 decode 1.48s→~0.19s，权重仅 fp16 的13%。虽仍0.8×于fp16，
但这是“放得下档”的实用加速——任何 NF4 推理应默认用 `gemv_4bit`，不物化。

## 5. 2bit fused LUT：本机无法构建（blocker）+ 最终结论

- 现有 native 库只有 **3bit** fused（`q3bit_gemm.dll`）；`q3bit_kernel.dll` 仅有
  `bnb_dequant_2bit`，**没有 2bit GEMV/GEMV fused**。
- 本机 **无 C++ 编译器**（`cl`/`g++`/`clang`/`zig` 均不可用，torch cpp_extension 也无编译器），
  因此无法新写 AVX2 2bit LUT fused kernel；**不虚报 2bit fused 数据**。
- 即使按 NF4 fused 的实测外推：权重字节减半（~0.5GB），但 dequant 是逐元素 LUT+dot，
  **NF4 fused 已只有 5.1GB/s（fp16 14.4、上限42.7）**，2bit 不可能把有效带宽提到 2×；且
  NF4 fused 整token ~191ms 仍慢于 fp16 239ms，未达 <100ms。
- **最终结论**：CPU 上“2bit 省一半带宽”不成立；dequant 实现吃掉带宽红利。2bit 只在
  “放得下”维度有价值；当前 `Linear2bit` 无 fused、还会 OOM，不用于推理交付。
  需要真正复核需有编译器的环境 + 新 AVX2 LUT 内核。
## 6. 2bit AVX2 LUT fused GEMV（zig 编译）实测（2026-10-23）

用 `D:\work\tools-zig` 编译 `q2bit_gemv.cpp`（AVX2，M=1，uniform 4-level，block64，不物化权重）：

| shape | 权重字节 | fused | material | speedup | fused带宽 | relL2 vs material |
|---|---|---|---|---|---|---|
| qkv 4096×2048 | 2.62MB | **2.978ms** | 110.0ms | 36.9× | **0.880GB/s** | 4.7e-7 |
| o 2048×2048 | 1.31MB | 1.499ms | 55.8ms | 37.2× | 0.874 | 4.8e-7 |
| gate 6144×2048 | 3.93MB | 4.458ms | 164.2ms | 36.8× | 0.882 | 4.7e-7 |
| up 6144×2048 | 3.93MB | 4.646ms | 163.8ms | 35.3× | 0.846 | 4.7e-7 |
| down 2048×6144 | 3.93MB | 4.512ms | 164.7ms | 36.5× | 0.871 | 8.2e-7 |
| lm_head 151936×2048 | 97.2MB | **114.4ms** | 4044ms | **35.3×** | **0.850** | 4.8e-7 |

- **整 token 估算**：28层×18.09ms + lm_head114.4 = **~621ms**，比 fp16 239ms **更慢**；有效带宽
  **0.85GB/s**，远低于 NF4 fused 5.1、fp16 14.4、上限42.7。
- 内核数值正确（relL2~5e-7），但**2bit 解包/码表计算**比 NF4 更贵；当前实现远未优化，且差距太大。
- 结论：**2bit fused 也救不了**；CPU “雷霆-2bit”不成立，dequant 实现吃掉带宽红利。
## 7. 工具链：本机 zig 可编 native DLL

`D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe`（0.14.1，自带 clang）。示例：

```powershell
& 'D:\work\tools-zig\zig\zig-x86_64-windows-0.14.1\zig.exe' c++ -O3 -mavx2 -mfma -shared `
  gt1030/q2bit_gemv.cpp -o gt1030/q2bit_gemv.dll
```

`q2bit_gemv.cpp` 即用此方式编译（ctypes 加载）。以后 2bit/NF4 kernel 不再需要 MSVC。
