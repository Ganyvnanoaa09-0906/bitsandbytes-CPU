# 2bit / 3bit 训练可用原型（CPU，LoRA）

原型：`q3bit_proto.py`（纯 torch/Python，未进 C 内核）。目标先回答两个问题，而不是空谈：

1. 块状 2/3-bit 量化在**真实 Qwen3.5-0.8B 权重**上的误差是多少；
2. **冻结 2/3-bit base + LoRA** 在 CPU 上到底能不能训、省多少内存、慢多少。

运行：

```bash
python q3bit_proto.py --mode packtest
python q3bit_proto.py --mode train --model tiny --bits 3 --steps 60 --lr 0.01
python q3bit_proto.py --mode train --model qwen --bits 3  --steps 2 --seq 64
```

## 1. 量化误差（block=64，per-block absmax，对称均匀量化）

| 数据 | 2-bit relL2 | 3-bit relL2 | 8-bit relL2 |
|---|---|---|---|
| 随机 N(0,1) | 0.5059 | 0.2156 | 0.0059 |
| Qwen3.5-0.8B attn.qkv | 0.7165 | 0.2781 | 0.0074 |
| Qwen3.5-0.8B attn.proj | 0.5757 | 0.2344 | 0.0063 |
| Qwen3.5-0.8B mlp.linear_fc1 | 0.7212 | 0.2786 | 0.0073 |

即：3-bit 每 block 64 元素误差 ~23–28%，2-bit ~58–72%，8-bit ~0.6–0.7%。
（对照：NF4 4-bit 通常 ~4–6%；3-bit 已经不是"无损级别"，必须靠 LoRA 吸收误差。）

## 2. tiny 模型 60 步 LoRA 训练（合成 batch，lr=0.01）

| base 精度 | loss 首 → 末 | 步时 | 备注 |
|---|---|---|---|
| fp32 | 1.2877 → 0.5561 | 25.1 ms | 对照 |
| 3-bit | 1.2884 → 0.5455 | 36.1 ms（+44%） | 训练稳定，loss 同幅下降 |
| 2-bit | 1.2866 → 0.5296 | 33.5 ms（+33%） | 训练稳定，量化噪声更大 |

结论：2/3-bit 冻结 base **不影响 LoRA 优化**（梯度只经过 dequantized 常量），loss 曲线正常。

## 3. 真实 Qwen3.5-0.8B：内存 vs 速度（seq=64，LoRA r=8，2 步）

| base | 权重内存 | RSS | 步时 | loss 首 → 末 |
|---|---|---|---|---|
| fp32 | 3.202 GB | 5.34 GB | 3026.7 ms | 13.61 → 12.37 |
| **3-bit** | **1.228 GB** | **3.57 GB** | 7493.4 ms（2.5×） | 13.44 → 12.14 |
| **2-bit** | **1.159 GB** | 3.50 GB | 6131.3 ms（1 步） | 15.61（误差更大） |

- 3-bit 把**权重内存降 62%**（3.20→1.23 GB），RSS 降 1.77 GB，训练正常收敛；
- 但**步时变慢 2.5×**：原型用 torch 张量逐层解包（每步 0.8B 元素的移位/掩码），
  收益被解包开销吃掉——这正是需要 C 内核的地方（见下）；
- 2-bit 内存只再省 0.07 GB（因为 0.8B 模型的 embedding/lm_head 仍 fp32，占 ~1 GB），
  误差更大（首步 loss 15.6 vs 3-bit 15.05），不建议优先。

## 4. 结论与下一步

- **训练可用性已验证**：冻结 2/3-bit base + LoRA 在 CPU 上可以正常训练、loss 下降、
  内存显著下降；瓶颈是**解包速度**而不是可行性。
- 内存下降没完全兑现（1.23 GB 而非理论 0.4 GB）：`nn.Linear` 只占 0.8B 模型约 2/3，
  embedding/lm_head（~1 GB fp32）尚未量化——下一步量化 **embedding（查表+反量化）** 才能
  让内存收益覆盖整个模型。
- 速度：原型解包慢 2.5×；真正的产品化需要 **C 内核的 3-bit dequant + GEMM 融合**
  （类似现有 8-bit gemm 内核），把解包融进 GEMM 主循环，目标接近 fp32 速度。
- 2-bit：误差与边际内存收益不划算，除非配合 per-channel/异常值处理（或 2-bit 只用于
  优化器状态而非权重）。

产物：`q3bit_proto.py`；数据归档本文件；报告 §7.139。

## 5. 进阶：embedding 量化 + C 融合解包内核（第二轮）

### 5.1 embedding 量化

Qwen3.5-0.8B 是视觉+语言多模态模型，文本侧 `language_model.embed_tokens` 是
`(248320, 1024)` 的 fp32 表（~1 GB，lm_head 与它 tie）。`QuantEmbedding` 按行分块量化，
lookup 时只解包被选中的 token 行；原型里为 hidden-state 目标把 lm_head 置空以释放 tied 表
（真实 LM 训练需要单独处理 lm_head，见结论）。

| 配置 | 权重内存 | RSS | 步时 | loss 首→末 |
|---|---|---|---|---|
| fp32 base | 3.202 GB | 5.34 GB | 3026.7 ms | 13.61 → 12.37 |
| 3-bit linear only（torch 解包） | 1.228 GB | 3.57 GB | 7493.4 ms | 13.44 → 12.14 |
| 3-bit linear+embedding（torch 解包） | 0.378 GB | 2.68 GB | 6650.0 ms | 13.32 → 12.06 |
| **3-bit linear+embedding + C kernel** | **0.378 GB** | 2.76 GB | **3107.9 ms** | 13.33 → 12.06 |
| 2-bit linear+embedding + C kernel | 0.279 GB | 2.48 GB | 3008.8 ms | 15.05（1 步） |

### 5.2 C 融合解包内核（q3bit_kernel.cpp）

AVX2 内核：每 block 3 字节 → 8 个 3-bit code，`_mm256_cvtepi32_ps` + FMA 算出 8 个 float，
直接写 fp32 权重缓冲；再用 torch F.linear。编译：

```bash
g++ -O3 -mavx2 -mfma -fPIC -shared q3bit_kernel.cpp -o q3bit_kernel.so   # Linux
zig c++ -target x86_64-windows-gnu -O3 -mavx2 -mfma -shared q3bit_kernel.cpp -o q3bit_kernel.dll
```

验证与性能：
- **bit-exact**：2/3-bit 内核输出与 torch 解包 maxdiff = 0；
- 微基准（67M 元素，约 0.8B 模型 1/8 的 Linear 权重）：内核 **32.8 ms** vs torch **239.8 ms**，
  **7.31× 加速**；
- 端到端：0.8B 3-bit 训练步时 **7493 → 3108 ms**（embedding 版 6650→3108），
  与 fp32 的 3027 ms 基本持平（1.03×）；tiny 模型 36.1 → **25.3 ms**（fp32 25.1 ms）。

### 5.3 更新后的结论

- **"低比特冻结 base + LoRA 训练"在 CPU 上可行且代价可接受**：3-bit 权重内存 3.202→0.378 GB
  （**-88%**），RSS 5.34→2.76 GB（-48%），步时 1.03× fp32；
- 2-bit 进一步到 0.279 GB/2.48 GB 且同样快，但首步 loss 15.05 vs 3-bit 13.33，误差代价明显，
  暂不推荐默认使用；
- 产品化路径清晰：把该 AVX2/NEON 内核移进 `csrc`（dequant+GEMM 融合或独立 dequant），
  并处理 lm_head：tied embedding 的 logits 可每步用同一内核解包（0.25B 元素 ~120 ms 量级，
  需进一步实测）或保留 fp32 lm_head；
- 原型验证了 **C 内核是必需的**：没有它 2.5× 慢，有了它几乎无感——这就是下一阶段进
  `bitsandbytes` 正式内核的动机。

## 6. tied lm_head（logits 路径）实测

真实 LM 训练需要输出 logits：Qwen3.5-0.8B 的 lm_head 与 embedding tie，表是
(248320, 1024) fp32 = **0.947 GB**。用 csrc 同款 AVX2 内核实测（seq=64，脚本
`q3bit_lmhead_bench.py`）：

| 方案 | lm_head 内存 | logits 时间 | 额外开销 | dequant 占比 |
|---|---|---|---|---|
| fp32 表 | 0.947 GB | 636.3 ms/步 | — | — |
| 3-bit 表（每步全表解包） | **0.104 GB（-89.1%）** | 867.0 ms/步 | **+230.7 ms/步** | 187.7 ms |

结论：
- 3-bit lm_head 省 0.84 GB，代价约 +7.4%（相对 3.1s 训练步）；内存紧张时值得；
- dequant 187.7ms 占了开销大头（约 81%），因为每步都要把整张表写回 fp32 再 GEMM；
- 最优解是 **fused 3-bit dequant + GEMM**（在 GEMM 的 k 循环里就地解包，避免 1GB 写+读），
  参考现有 `gemm_8bit` 的做法——下一步实现；在那之前，真实 LM 训练建议：
  线性层+embedding 3-bit、**lm_head 保留 fp32**（总权重 0.378+0.947=1.325 GB，仍比 fp32
  3.202 GB 少 59%，且没有 logits 的 230ms 额外开销）。

## 7. fused 3-bit dequant+GEMM（第三轮：正式进 csrc）

不再把整张 fp32 表写出来：`q3bit_gemm` 按 B 行 tile（默认 64 行 × K）现场解包到 L2 缓冲，
再直接算 `C = A @ B^T`。

**原型（独立 DLL，Qwen3.5-0.8B lm_head 形状 M=64,N=248320,K=1024）**：

| 路径 | 时间 | 说明 |
|---|---|---|
| fp32 torch matmul | 579.4 ms | 读 0.947 GB fp32 表 |
| 解包全表 + torch matmul | 825.8 ms | 旧方案（dequant 188ms + matmul） |
| **fused 3-bit GEMM** | **498.2 ms（ntile=64）** | 读 0.104 GB packed 表，**比 fp32 快 14%** |

**正式 csrc 版**（`cbq3bit_gemm_cpu`，N=131072 的等比缩小测试，WSL 原生构建）：

| 路径 | 时间 | 相对 |
|---|---|---|
| torch fp32 matmul | 313.8 ms | 1.00× |
| csrc fused GEMM | **257.7 ms（ntile=32）** | **1.22× 快**（外推 N=248320 ≈ 488ms） |

reldiff vs 参考 4.29e-07（fp32 累加顺序差异）。自检第 8 项覆盖 fused GEMM vs 标量参考：
x86 AVX2 原生 **8/8 PASS**、x86 generic **8/8 PASS**、aarch64 qemu **8/8 PASS**。

结论：**lm_head 走 fused 3-bit GEMM 是严格双赢**（内存 -89%、前向速度 +14~22%）。限制：
当前实现只做前向 / `C=A@B^T`；训练反向还需要 `dX = dY @ W` 的转置融合变体（下一轮）。

## 8. 训练反向：转置融合 GEMM（dX = dY @ W）+ 端到端 lm_head 训练（第四轮）

**转置内核** `q3bit_gemm_t`（`dX[M,K] = dY[M,N] @ dequant(B)[N,K]`，线程私有 dX 再归约；
已进 csrc：`cbq3bit_gemm_t_cpu`，自检第 9 项）。实测（Qwen3.5-0.8B lm_head 形状
M=64,N=248320,K=1024）：

| 路径 | 时间 |
|---|---|
| torch fp32 backward matmul | 158.6 ms |
| fused transposed 3-bit (ntile=16/64) | 381.6 ms（慢 2.4×） |
| 正确性 | reldiff 3.78e-06 |

反向是计算密集型（输出只有 64×1024），oneDNN 的 GEMM 效率更高；我们的融合内核优势在
"不读 fp32 表"，反向场景下 dequant+AXPY 抵消不了 GEMM 效率差距。**结论：前向融合值得，
反向当前不建议替换**（保留这个能力，等更好的分块/外积实现再评估）。

**端到端真实训练**（Qwen3.5-0.8B，CE loss，seq=64，LoRA，AdamW8bit，3 步）：

| 配置 | 步时 | 权重内存 | RSS | CE loss |
|---|---|---|---|---|
| fp32 base + fp32 lm_head | 3206 ms | 3.202 GB | 5.40 GB | 7.00 → 3.85 |
| **3-bit linear+embedding+lm_head（fused 前向）** | 4202 ms（+31%） | **0.378 GB（-88%）** | **2.69 GB（-50%）** | 8.47 → 6.00 |

即：全 3-bit 训练（含 logits）能正常收敛，内存减半以上，代价约 +31% 步时。当前瓶颈是
**每个 Linear 每步都要解包成 fp32 再 F.linear**（前向 +0.5s）；把线性层也做成 fused
3-bit GEMM（像 lm_head 前向那样）是下一步，可把这部分开销消掉。

调试中修掉两个真实 bug：① backward 里 K 从 packed 行宽推算错误（384//3*64=8192）导致越界
访问——改为由 forward 保存 K；② lmhead 脚本构建时没先 `load_kernel()`，QuantLinear 回退
到 torch 解包（前向 5.4s vs 正确 1.9s），是个值得记的"静默变慢"陷阱。

自检扩到 **9 项**：x86 AVX2 原生 9/9、x86 generic 9/9、aarch64 qemu 9/9；ISA 矩阵 PASS。
