# MoE 混合精度：EFST 热专家 + 冷专家 3-bit（2026-10-19）

脚本：`scripts/misc/moe_hybrid_bench.py`。目标：在 scaled Mixtral（HF transformers 5.16 **融合 3D 张量专家**）
上验证「EFST 校准选热专家 → hot 保持 fp32、cold 量化 3-bit 按需解包」是否能拿到比全 3-bit
更好的内存/速度折中。

## 实现

1. `MixtralConfig(hidden 512 / intermediate 1024 / 4 层 / 8 专家 / top-2 / vocab 2000)`，
   约 53.5M 参数（激活 ~6.3M/token）；
2. EFST（`top_k=2`，4 个校准 batch）为每层选出 2 个最热专家（8 选 2）；
3. 用 `HybridExperts` 替换 `MixtralExperts`：热专家保留 fp32 `Parameter`，冷专家按专家矩阵
   量化 3-bit（packed+absmax buffer）；复刻原 forward 的 one-hot gather/index_add 逻辑，
   冷专家在 dispatch 时用 AVX2 内核按需解包；
4. 冻结 base + 仅 attention LoRA 可训，AdamW8bit，3 步 CE 训练。

## 结果（seq=64，3 步）

| 变体 | 权重内存 | 相对 fp32 | 步时 | 相对 fp32 | RSS | CE loss |
|---|---|---|---|---|---|---|
| fp32 专家 | 0.205 GB | — | 89.5 ms | — | 0.63 GB | 7.7096 → 7.6869 |
| 全 3-bit 专家 | **0.038 GB** | **-81%** | 149.3 ms | +67% | 0.59 GB | 7.7170 → 7.6883 |
| 混合（2/8 热 fp32 + 6/8 冷 3-bit） | 0.080 GB | -61% | **132.8 ms** | +48% | 0.57 GB | 7.7191 → 7.6981 |

（步时有多轮波动，±8ms 量级；三次独立运行趋势一致。）

## 结论

1. **混合精度是一个可选的折中点，但不支配全 3-bit**：混合省 61% 内存、慢 48%；全 3-bit
   省 81% 内存、慢 67%。两者都在内存/速度折中前沿上——内存极度紧张选全 3-bit，想少慢一点
   选混合；混合并没有"又省又多快"。
2. 原因数据化：seq=64 时每层 8 个专家**全部会被路由命中**（router token 计数显示所有专家
   都有 token），所以"冷"专家只是命中少、不是不命中；cold 3-bit 每步仍要解包 → 省不了
   cold 解包时间。**想真正提速必须做"只解包命中 token 行"的 fused dequant+GEMM**，
   而不是按专家精度混合。
3. **顺手修了 EFST 一个真实 bug**：融合张量专家（HF 5.x Mixtral/Qwen3Next 风格）的
   `collect_expert_usage` 用 int 去查字符串 key 的计数器 → 所有计数恒为 0，
   `select_top_experts` 永远返回前 k 个专家（[0,1]），校准失效。已修复 int/str 归一化，
   加了回归用例（`scripts/test/test_efst_regressions.py` case 4），并给出真实路由计数证据。

## 下一步

- 推理侧（KTransformers 对照的正确维度）：对 hit 专家用已有 `q3bit_gemm` 做 fused
  3-bit dequant+GEMM（prefill seq=64 / decode seq=1），测内存与 token/s；
- 训练侧要提速则需 fused expert GEMM 的**反向**（`dX=dY@W`）融合，当前原型比 oneDNN 慢，
  暂不值得（§7.144）。
