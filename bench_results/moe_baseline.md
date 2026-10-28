# CPU MoE 训练基线（2026-10-19）

脚本：`scripts/misc/moe_cpu_bench.py`。范围说明：KTransformers 是**推理期**专家卸载服务器，本机没有
同台可比基线；这里建立的是**可复现的 CPU MoE 训练基线**（内存/吞吐/收敛 + EFST 集成），
供后续与推理方案对照，而不是宣称"干翻 KTransformers"。

## A. 真实 tiny Mixtral（HF 5.x 融合 3D 张量专家）+ EFST

模型：`hf-internal-testing/tiny-random-MixtralForCausalLM`（hf-mirror.com），4,318,016 参数。
transformers 5.16 的 Mixtral 专家是融合张量：`MixtralExperts.gate_up_proj [E,2I,H]`、
`down_proj [E,H,I]`，router 是自定义 `MixtralTopKRouter`——正好对应 EFST 的**张量专家**路径。

EFST（`top_k=2`, `lora=True`, r=8，4 个校准 batch）结果：

- 识别到 **2 个专家组**，可训参数 **4,318,016 → 65,536（1.5%）**；
- router 路由分布：每层平均覆盖 **4.0/4** 个专家（top-2，batch 16 tokens）；
- EFST 训练 6 步：**17.5 ms/step**，loss **10.3348 → 10.2104**，RSS 0.44 GB。

## B. 合成 Mixtral 风格 MoE（nn.Linear 专家，便于 3-bit 量化对照）

配置：hidden 512 / intermediate 1024 / 4 层 / 8 专家 / top-2，vocab 2000，seq 64，
53.48M 总参数、**~6.26M 激活参数/token**；冻结 base + attention-LoRA（33k 可训）。

| 变体 | 权重内存 | RSS | 步时 | loss |
|---|---|---|---|---|
| fp32 专家 | 0.199 GB | 0.65 GB | 88.6 ms | 7.8082 → 7.7997 |
| **3-bit 专家（csrc dequant 内核）** | **0.033 GB（-83%）** | 0.56 GB | 175.9 ms（2.0×慢） | 7.8228 → 7.8143 |

3-bit MoE 慢的原因很具体：每个专家每次被调用都要把整份权重解包成 fp32（4 层 × 8 专家 ×
2 个 Linear），解包次数远多于 dense 模型。→ MoE 场景要用 3-bit 必须做**按需融合的
dequant+GEMM**（只解包本轮 hit 的专家行），或干脆只量化冷专家。

## C. 真实 Mixtral 融合张量专家的 3-bit 内存潜力（仅内存估算）

对 `gate_up_proj` / `down_proj` 按专家矩阵做块状 3-bit 量化（block=64）：
**0.8 MB → 0.1 MB（10.9%）**——小模型比例一致，换算到真实 MoE（专家占大头）就是 -89%
专家内存潜力。限制：融合专家的 forward 走 `F.linear(current_state, gate_up_proj[expert])`，
要真正用上 3-bit 需要 patch `MixtralExperts.forward` 做按需解包（下一阶段）。

## 结论 / 下一步

- EFST 与 HF 新式融合专家布局的集成**验证通过**（这是之前只有单元测试、没有真实
  Mixtral 实例覆盖的部分）；
- CPU MoE 训练基线建立：tiny Mixtral 17.5 ms/step；53M 合成 MoE 89 ms/step（fp32）；
- 3-bit 专家对 MoE **内存收益很大（-83~89%）但当前逐专家解包太慢**；MoE 的 3-bit 必须
  按"hit 专家行"融合解包，不能照搬 dense 的逐层 dequant；
- 与 KTransformers 的对照：只能在推理侧做同模型/同内存预算的测试，且需要 Linux +
  其运行时；列入后续（当前不伪造对比数字）。
