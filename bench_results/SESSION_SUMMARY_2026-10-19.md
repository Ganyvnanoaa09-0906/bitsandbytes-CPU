# 自主会话总结（2026-10-19 下午，用户上课期间）

本文件汇总本轮自主推进的产出、证据与剩余事项，便于回来后快速接续。

## 交付物

1. **PyPI `bitsandbytes-cpu-fork==0.50.2.dev5`**（已发布，4 个文件：win_amd64 /
   manylinux_2_17 / manylinux_2_39 / sdist；twine PASSED，PyPI 独立 venv 重装验证）。
2. **Windows 核显训练一键包** `gt1030/dist/bitsandbytes-cpu-igpu-train-win-v0.1.0.zip`
   （1.56 MB；setup/doctor/train.bat + clblast.dll + 许可证；本机 Vega 6 实测 iGPU parity
   5.78e-3 PASS、CPU 回退 PASS）。
3. **ISA 兼容矩阵**：`isa_build.py` + `isa_matrix.sh`；x86 AVX2/generic/AVX-512 静态审计 +
   aarch64 qemu 运行时自检；C 自检从 5 项扩到 **9 项**，x86 AVX2 原生、x86 generic、
   aarch64 qemu 全部 9/9 PASS；结果 `bench_results/isa_matrix.md`。
4. **2bit/3bit 训练链路**（`q3bit_proto.py`、`q3bit_kernel.cpp`、`q3bit_gemm.cpp`、
   `q3bit_lmhead_bench.py`、`q3bit_lmhead_train.py`）：
   - 量化误差：3-bit relL2 0.22–0.28（真实 Qwen3.5-0.8B 权重），2-bit 0.51–0.72；
   - Qwen3.5-0.8B LoRA：权重 **3.202 → 0.378 GB（-88%）**、RSS 5.34 → 2.69 GB、
     端到端 CE 训练 +31% 步时（4202 vs 3206 ms）；
   - fused 3-bit GEMM 前向（lm_head）：比 fp32 matmul **快 14–22%**，读 0.104GB vs 0.947GB；
   - 反向转置融合：比 oneDNN fp32 慢 2.4×，**决定暂不替换**（记录于 §7.144）；
   - Linear 层逐层融合经逐层 + OpenMP 持久线程池实测**不划算**，决定保留"解包 + torch
     F.linear"（§7.145）。
5. **CPU MoE 基线**（`scripts/misc/moe_cpu_bench.py`，`bench_results/moe_baseline.md`）：
   - EFST 通过 HF transformers 5.16 新式**融合 3D 张量专家**实测：tiny Mixtral 2 个专家组、
     可训 4,318,016 → 65,536、17.5 ms/step、loss 10.33→10.21；
   - 合成 53.48M MoE：fp32 88.6ms/0.199GB vs 3-bit 175.9ms/0.033GB（-83%，慢 2×）；
   - 真实融合专家 3-bit 内存潜力 -89%（需按 hit 专家行按需解包才能真正提速）。
6. **csrc 新增正式内核**：`dequantize_2bit/3bit_cpu`（AVX2/NEON/标量）、
   `q3bit_gemm_cpu`（fused 前向）、`q3bit_gemm_t_cpu`（转置反向）+ pythonInterface 导出 +
   9 项自检（x86 AVX2/generic、aarch64 qemu 全绿）。

## 提交（均已 push）

```
7c740025 test: CPU MoE training baseline (real tiny Mixtral + EFST, fp32 vs 3-bit experts)
3ffa0f26 docs: decide against fusing Linear layers (per-layer + OpenMP measurements)
17f92301 feat: 3-bit training backward (transposed fused GEMM) + end-to-end CE training
b6270d83 feat(csrc): fused 3-bit dequant+GEMM (lm_head path faster than fp32) + selftest 8/8
5328cd8c test: measure tied lm_head quantized logits path (3-bit vs fp32)
240eb290 feat(csrc): 2/3-bit dequant kernels (AVX2 + NEON + scalar reference) + selftest
dda84ff6 feat: 3-bit embedding quantization + AVX2 dequant kernel (0.8B training ~= fp32 speed)
3367057c feat: 2bit/3bit frozen-base LoRA training prototype (CPU)
a3a747db test: formalize ISA compatibility matrix (x86 AVX2/generic/AVX-512 + aarch64 qemu)
6548ab92 packaging: one-click Windows package for iGPU training
0ef64c8f release: 0.50.2.dev5
d0970746 perf: real-model CPU step profile + tune_threads()
```

## 最终验证证据

- `git status`：工作区干净，分支与 origin 同步；
- 技术报告 15,392 行、审计 **0 bad chars / 0 corrupt regions**；
- C 自检：x86 AVX2 原生 **9/9 PASS**、x86 generic **9/9 PASS**、aarch64 qemu **9/9 PASS**；
- ISA 矩阵 `ISA_MATRIX_PASS`；PyPI 最新版本 = `0.50.2.dev5`（4 files）；
- 所有基准/测试只在本地磁盘与 WSL ext4 进行，未写 9P 宿主盘、未删除用户数据。

## 剩余 backlog（回来后可选）

1. **MoE 3-bit 按需解包**：patch `MixtralExperts.forward`，只解包本轮 hit 专家行；
   再评估与 KTransformers 推理侧的同模型/同内存对照（需 Linux + 其运行时）。
2. **dev6 发布**：把 csrc 3-bit 内核暴露成 Python API（`quantize_3bit/dequantize_3bit/
   gemm_3bit`）+ 打包 dev6（需 PyPI token）。
3. **视频/图像模型**支持面扩展；**iGPU 包**可加自动 CLBlast 下载/更多示例。
4. **老 glibc 并行 wheel**：devtoolset-8 工具链 或 豆包容器构建 manylinux_2_35 后传回。
5. FMA/缓存进一步压榨：需 oneDNN 级 GEMM 内核，ROI 低（§7.145 已论证）。
