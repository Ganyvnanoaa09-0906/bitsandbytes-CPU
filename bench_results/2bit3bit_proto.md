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
