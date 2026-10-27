# CPU GPTQ/AWQ int4 支持状态（2026-10-27）

## 已完成

- 下载并本地保存：
  - `D:\work\models\qwen05_gptq.safetensors`（Qwen2.5-0.5B-Instruct-GPTQ-Int4，459MB）
  - `D:\work\models\qwen05_awq.safetensors`（Qwen2.5-0.5B-Instruct-AWQ，731MB）
  - base `qwen05_base.safetensors`（Qwen2.5-0.5B-Instruct fp16，988MB）+ config/tokenizer
- 格式确认：
  - **GPTQ**（Qwen2.5）：`qweight [in/8,out]`（每 int32 打包 8 个 IN 的 4bit）、`qzeros [in/group,out/8]`（打包 OUT）、`scales [in/group,out]`；对称 zero point 需 `qzeros+1`（实测 7→8）。
  - **AWQ**：`qweight [in,out/8]`、`qzeros [in/group,out/8]`、`scales [in/group,out]`；官方 `autoawq` 的 `dequantize_gemm` 顺序为 `unpack_awq` + `AWQ_REVERSE_ORDER=[0,4,1,5,2,6,3,7]`，已照抄到 `gt1030/gptq_awq_cpu.py`。

## 实测（base 权重对拍，同 prompt logits）

| 格式 | 替换 Linear | weight rel（中位/最大） | logits rel | 结论 |
|---|---|---|---|---|
| GPTQ | 168 | **0.150 / 0.243** | **0.533** | 反量化方向正确，但 logits 远未 <1e-3 |
| AWQ | 168 | **0.337 / 6.19** | **0.942** | 仍未对齐（官方公式照抄后） |

- 说明：base 与量化模型本应有 4bit 量化误差；**0.533/0.942 的 logits rel 不能用“量化误差”解释**，
  更像 checkpoint/base 版本不匹配或 scale/zero 轴仍有误。
- **不宣称 #2 通过**：验收要求 logits rel<1e-3、decode M=1 比物化 ≥5×、内存≈4bit 理论；
  后两项（无 fused GPTQ/AWQ GEMV，未测）也没做。

## 阻塞/下一步

1. 找**官方 CPU dequant 参考**（autoawq/auto-gptq 源码在 `research/paper/awq_pkg/src`）或
   与 checkpoint 完全同 revision 的 base 权重，逐层对齐 weight（当前 base 可能 revision 不同）。
2. AWQ 还需核对 checkpoint 的 `qzeros` 是否存 `zero-1` 约定（`unpack_reorder_pack` 里有 `izeros-1`，
   但 `dequantize_gemm` 没有）；用单层小张量做 exact 对拍，不靠猜。
3. 实现 GPTQ/AWQ **fused GEMV**（复用 4bit 解包+dot；当前只有 NF4 `gemv_4bit`，格式不同），
   再测 decode M=1 ≥5× 与内存。

## 复现

```powershell
python gt1030/gptq_awq_cpu.py --format gptq
python gt1030/gptq_awq_cpu.py --format awq
```
日志：`research/bench_results/gptq_cpu*.log`、`awq_cpu*.log`。
## 控制量化（2026-10-27，lead 建议）

同一 `layers.0.self_attn.q_proj`，同一 group128，自己压 int4 再反量化：

| 组 | weight rel | 单层 y rel |
|---|---|---|
| control 对称（zero=8） | 0.1397 | 0.1014 |
| control 非对称（min/max zero） | 0.1137 | 0.0829 |
| **GPTQ checkpoint** | **0.1737** | **0.1265** |
| AWQ checkpoint（REVERSE_ORDER） | 0.4638 | 0.2687 |
| AWQ checkpoint（no reverse） | 1.1603 | 1.1043 |

- **GPTQ 与 control 同量级**（0.17 vs 0.14），说明 in/out 打包、qzeros+1、group 轴**基本正确**；
  剩余差是 GPTQ 实际量化误差 + base revision，不是纯解包崩。
- **AWQ 仍明显错**（0.46 vs control 0.11–0.14）；去掉 REVERSE_ORDER 更差（1.16）⇒ 问题不在
  reverse，而在 qzeros/scales/pack 约定（或 checkpoint 用了不同 zero 约定）。
- 控制组表明 lead 估的“0.02–0.04”过于乐观：**g128 对称 int4 实测 wrel≈0.14**。
- 单层 y rel GPTQ 0.126 >5e-3，仍不能算通过；全模型 logits0.53 也说明 base/量化 checkpoint
  不能直接这样对拍。
